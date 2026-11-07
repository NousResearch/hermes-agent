"""The recorder — one ffmpeg process, one state file, one window poller.

Explicit record on/off. Nothing is captured until ``start()`` is called and
nothing keeps running after ``stop()``; there is no ambient capture mode in this
plugin, by design.

State lives in a file rather than in memory because the CLI, the agent tool,
and the desktop app are three different processes that all need the same answer
to "is a recording running". The file is the registry; the PID in it is checked
for liveness rather than trusted, so a crashed ffmpeg does not leave the plugin
permanently convinced it is still recording.

Stopping is a graceful ``q`` on ffmpeg's stdin first, SIGTERM second, kill
last. That order is not politeness: an mp4 whose moov atom was never written is
unplayable and unanalyzable, so killing ffmpeg outright destroys the take it
just spent an hour capturing.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_dir

from plugins.watch import capture as cap

logger = logging.getLogger(__name__)

#: How long to wait for ffmpeg to finalize the container after 'q'. Writing the
#: moov atom on a long take takes a moment; cutting this short is how you get a
#: corrupt file at the last step.
_GRACEFUL_TIMEOUT = 8.0
_TERM_TIMEOUT = 4.0

#: Window-poll cadence. Matches the HUD's own overlay poll (1.5 s) so the two
#: agree about when an app changed, and stays cheap enough to be invisible next
#: to a game.
_POLL_INTERVAL = 1.5


def watch_dir() -> Path:
    """Where takes live. Profile-aware: two profiles must not share recordings."""
    return get_hermes_dir("workspace/watch", "watch")


def _state_path() -> Path:
    return watch_dir() / "current.json"


@dataclass
class RecordingState:
    """What is being recorded right now.

    ``timeline_path`` is JSONL, appended to by the poller thread while the
    recording runs, so a killed process still leaves a usable partial timeline.
    """

    pid: int
    started_at: float
    video_path: str
    timeline_path: str
    label: str
    fps: float
    audio: bool
    platform: str

    def to_json(self) -> dict:
        return asdict(self)


def _pid_alive(pid: int) -> bool:
    """Whether *pid* is still running.

    ``os.kill(pid, 0)`` is POSIX-only, so Windows goes through a tasklist
    query. A recording that has finished must be reported as finished on every
    host, not just the ones where the cheap check works.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return str(pid) in out.stdout
        except (OSError, subprocess.SubprocessError):
            # Unknown beats a false 'finished' — a live ffmpeg reported dead
            # would let a second recording start on top of the first.
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_state() -> Optional[RecordingState]:
    path = _state_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RecordingState(**data)
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("watch: unreadable state file (%s); treating as idle", exc)
        return None


def _write_state(state: RecordingState) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state.to_json(), indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_state() -> None:
    try:
        _state_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("watch: could not clear state (%s)", exc)


def status() -> dict:
    """Current recorder state, with a dead process reported as idle.

    Self-healing on purpose: if ffmpeg died (disk full, device unplugged), the
    stale state file is cleared here rather than blocking every future
    ``start`` until someone deletes it by hand. The partial file is still
    reported, because a take that stopped early is usually still worth watching.
    """
    state = read_state()
    if state is None:
        return {"recording": False}

    if not _pid_alive(state.pid):
        _clear_state()
        video = Path(state.video_path)
        return {
            "recording": False,
            "note": "previous recording ended unexpectedly",
            "video_path": state.video_path,
            "video_exists": video.is_file(),
            "size_bytes": video.stat().st_size if video.is_file() else 0,
        }

    elapsed = time.time() - state.started_at
    video = Path(state.video_path)
    return {
        "recording": True,
        "label": state.label,
        "elapsed_seconds": round(elapsed, 1),
        "video_path": state.video_path,
        "timeline_path": state.timeline_path,
        "size_bytes": video.stat().st_size if video.is_file() else 0,
        "audio": state.audio,
        "fps": state.fps,
    }


# ── Window poller ─────────────────────────────────────────────────────────

class _WindowPoller:
    """Append frontmost-window samples to a JSONL sidecar while recording.

    The provider is injected. On the desktop the caller supplies one backed by
    the app's own enumeration (the same ``get-windows`` path the HUD overlay
    uses); headless callers pass nothing and simply get no timeline, which is a
    degraded take rather than a failed one.
    """

    def __init__(self, path: Path, provider, started_at: float, interval: float = _POLL_INTERVAL):
        self._path = path
        self._provider = provider
        self._started_at = started_at
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._provider is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="watch-window-poll", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last: Optional[tuple[str, str]] = None
        while not self._stop.is_set():
            try:
                front = self._provider()
            except Exception as exc:  # pragma: no cover — provider is foreign
                logger.debug("watch: window provider failed (%s)", exc)
                front = None

            if front:
                key = (front.get("app", ""), front.get("title", ""))
                # Only transitions are recorded. A sample per tick would be
                # ~2,400 rows an hour of which ~40 carry information.
                if key != last and key[0]:
                    row = {
                        "at": round(time.time() - self._started_at, 2),
                        "app": key[0],
                        "title": key[1],
                    }
                    try:
                        with self._path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row) + "\n")
                    except OSError as exc:
                        logger.debug("watch: timeline write failed (%s)", exc)
                    last = key

            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


_active_poller: Optional[_WindowPoller] = None


# ── Start / stop ──────────────────────────────────────────────────────────

def start(
    *,
    label: str = "",
    fps: float = 15.0,
    audio_device: Optional[str] = None,
    screen_index: int = 1,
    region: Optional[tuple[int, int, int, int]] = None,
    window_provider=None,
    platform: Optional[str] = None,
) -> dict:
    """Begin recording. Returns a result dict; never raises for user error.

    Refuses when a recording is already live: two ffmpeg processes writing two
    files while one state slot tracks one of them loses a take, and losing a
    take you cannot re-perform is the worst failure this plugin has.
    """
    existing = status()
    if existing.get("recording"):
        return {
            "success": False,
            "error": "Already recording. Stop the current take first.",
            "status": existing,
        }

    host = platform or sys.platform
    plan = cap.capture_plan(
        host,
        fps=fps,
        audio_device=audio_device,
        screen_index=screen_index,
        display=os.environ.get("DISPLAY"),
        wayland_display=os.environ.get("WAYLAND_DISPLAY"),
        region=region,
    )

    if plan.blocked:
        return {"success": False, "error": plan.blocked}

    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in label).strip("-")
    name = f"{stamp}{'-' + slug if slug else ''}"
    directory = watch_dir()
    directory.mkdir(parents=True, exist_ok=True)
    video_path = directory / f"{name}.mp4"
    timeline_path = directory / f"{name}.timeline.jsonl"

    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    args += plan.args
    args += cap.encoder_args(audio=plan.audio)
    args += ["-movflags", "+faststart", str(video_path)]

    try:
        # stdin is a pipe because that is how ffmpeg is asked to finalize
        # cleanly ('q'). Without it there is no graceful stop at all.
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=(sys.platform != "win32"),
        )
    except FileNotFoundError:
        return {
            "success": False,
            "error": "ffmpeg not found on PATH. Install it and retry.",
        }
    except OSError as exc:
        return {"success": False, "error": f"Could not start ffmpeg: {exc}"}

    started_at = time.time()

    # A capture device that rejects the arguments fails in the first moment,
    # not at stop time. Surfacing ffmpeg's own stderr here is the difference
    # between a fixable message and an empty file discovered an hour later.
    time.sleep(0.6)
    if process.poll() is not None:
        detail = ""
        if process.stderr is not None:
            try:
                detail = (process.stderr.read() or b"").decode("utf-8", "replace").strip()
            except (OSError, ValueError):
                pass
        return {
            "success": False,
            "error": f"ffmpeg exited immediately: {detail[-500:] or 'no output'}",
        }

    state = RecordingState(
        pid=process.pid,
        started_at=started_at,
        video_path=str(video_path),
        timeline_path=str(timeline_path),
        label=label,
        fps=fps,
        audio=plan.audio,
        platform=host,
    )
    _write_state(state)

    global _active_poller
    _active_poller = _WindowPoller(timeline_path, window_provider, started_at)
    _active_poller.start()

    return {
        "success": True,
        "recording": True,
        "video_path": str(video_path),
        "timeline_path": str(timeline_path),
        "audio": plan.audio,
        "notes": plan.notes,
    }


def _graceful_stop(pid: int) -> None:
    """Ask ffmpeg to finalize, escalating only as far as needed."""
    if sys.platform == "win32":
        # No stdin handle across processes on Windows and no SIGTERM either;
        # taskkill without /F lets ffmpeg run its own shutdown path.
        subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True, timeout=15)
        deadline = time.time() + _GRACEFUL_TIMEOUT
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.3)
        if _pid_alive(pid):
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=15)
        return

    try:
        os.kill(pid, signal.SIGINT)
    except (OSError, ProcessLookupError):
        return

    deadline = time.time() + _GRACEFUL_TIMEOUT
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.2)

    if not _pid_alive(pid):
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return

    deadline = time.time() + _TERM_TIMEOUT
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.2)

    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def sidecar(video: Path, kind: str) -> Path:
    """Path to a take's sidecar file.

    ``with_name`` over ``with_suffix``: a take labelled ``set.v2`` has ``.v2``
    as its last suffix, so ``with_suffix`` would silently rewrite the label
    away and point at another take's sidecar.
    """
    return video.with_name(f"{video.stem}.{kind}")


def timeline_path_for(video: Path) -> Path:
    return sidecar(video, "timeline.jsonl")


def meta_path_for(video: Path) -> Path:
    return sidecar(video, "meta.json")


def read_meta(video: Path) -> Optional[dict]:
    """Recording metadata for a take, or None when it wasn't recorded by us.

    A file the user dropped in from Game Bar has no meta and no timeline, so
    its absence is normal rather than an error.
    """
    path = meta_path_for(video)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _probe_duration(path: Path) -> Optional[float]:
    """The take's own duration, per ffprobe.

    Wall-clock elapsed is NOT this number. A capture backend that cannot keep
    up (x11grab on a loaded box, a busy game) writes fewer frames than real
    time, and the resulting file is genuinely shorter than the session was —
    measured here at 9.1 s of video for 21.1 s of wall clock. Everything the
    model is asked about lives on the FILE's timeline, so that is the duration
    that has to be reported and the one the window track has to be mapped onto.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nk=1:nw=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def stop() -> dict:
    """End the current recording and return where the take landed."""
    state = read_state()
    if state is None:
        return {"success": False, "error": "Not recording."}

    global _active_poller
    if _active_poller is not None:
        _active_poller.stop()
        _active_poller = None

    if _pid_alive(state.pid):
        _graceful_stop(state.pid)

    _clear_state()

    video = Path(state.video_path)
    wall_duration = time.time() - state.started_at
    if not video.is_file():
        return {
            "success": False,
            "error": f"Recording stopped but no file at {state.video_path}",
        }

    video_duration = _probe_duration(video)
    timeline = Path(state.timeline_path)

    # The meta sidecar is what lets `review` map wall-clock window samples onto
    # the file's own timeline later, in a different process. Without it the
    # drift is unrecoverable — the wall duration is gone the moment this
    # function returns.
    meta = {
        "label": state.label,
        "started_at": state.started_at,
        "wall_seconds": round(wall_duration, 2),
        "video_seconds": round(video_duration, 2) if video_duration else None,
        "audio": state.audio,
        "fps": state.fps,
        "platform": state.platform,
    }
    try:
        meta_path_for(video).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("watch: could not write meta sidecar (%s)", exc)

    return {
        "success": True,
        "video_path": state.video_path,
        "timeline_path": str(timeline) if timeline.is_file() else None,
        "duration_seconds": round(video_duration if video_duration else wall_duration, 1),
        "wall_seconds": round(wall_duration, 1),
        "capture_lag": (
            round(wall_duration - video_duration, 1)
            if video_duration and wall_duration - video_duration > 1.0
            else None
        ),
        "size_bytes": video.stat().st_size,
        "audio": state.audio,
    }


def takes(limit: int = 20) -> list[dict]:
    """Recorded takes, newest first."""
    directory = watch_dir()
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        # A prepared transcode lives beside its source and is an artifact, not
        # a take — listing it invites reviewing a re-encode of a re-encode.
        if path.name.endswith(".prepared.mp4"):
            continue
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
                "has_timeline": timeline_path_for(path).is_file(),
            }
        )
        if len(rows) >= limit:
            break
    return rows
