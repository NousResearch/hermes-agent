"""The ``/watch`` slash command — the GUI-facing surface.

The CLI verbs are the full interface, but a user in the desktop app or the TUI
should not have to open a terminal to start watching. Registering a plugin slash
command puts ``/watch`` in the composer palette of every surface at once: the
desktop treats non-builtin commands as extensions and surfaces them
automatically, and the TUI and messaging gateways dispatch through the same
registry. No core file is touched.

The command is deliberately thin. Live watching is a long-running loop with
streaming output, which a slash command cannot represent — it returns one string
and finishes — so ``/watch live`` starts the loop in a background thread and
returns immediately with where to look. Everything else (status, list, cost,
stop) is a synchronous question with a short answer, which is exactly what a
slash command is good at.
"""

from __future__ import annotations

import threading
from typing import Optional

from plugins.watch import decider as dec
from plugins.watch import live as livemod
from plugins.watch import recorder as rec

_HELP = """/watch — watch the screen and judge what's happening

  /watch live [brief]     start live commentary (runs in the background)
  /watch stop             stop live watching, or finalize a recording
  /watch status           what's running right now
  /watch record [label]   start recording a take
  /watch takes            list recorded takes
  /watch replay           re-tune the last live session (free, no API call)

Live watching is quiet by design: it comments only when something changed,
it hasn't just spoken, and it isn't repeating itself."""

#: The running loop, if any. A slash command is stateless, so the handle lives
#: here — one live session per process, which matches the one screen there is.
_session: Optional[dict] = None
_lock = threading.Lock()

#: How long to wait for a start to fail before reporting success. A blocked
#: platform or a missing ffmpeg is known at once, and saying "watching" while
#: nothing is captured is the failure this feature keeps tripping over. Module
#: level so tests can drop it rather than paying it on every case.
START_GRACE_SECONDS = 2.5


def _start_live(brief: str) -> str:
    global _session

    with _lock:
        if _session is not None and _session["thread"].is_alive():
            return "Already watching. `/watch stop` first."

        stop = threading.Event()
        said: list[str] = []
        result: dict = {}

        def announce(decision) -> None:
            stamp = f"{int(decision.at) // 60}:{int(decision.at) % 60:02d}"
            said.append(f"{stamp}  {decision.text}")

        def run() -> None:
            result.update(
                livemod.run_live(
                    brief=brief,
                    on_speak=announce,
                    stop=stop,
                    policy=dec.Policy(),
                )
            )

        thread = threading.Thread(target=run, name="watch-live", daemon=True)
        _session = {"thread": thread, "stop": stop, "said": said, "result": result}
        thread.start()

    # Give the loop long enough to fail loudly. A blocked platform (Wayland) or
    # a missing ffmpeg is reported by run_live immediately, and telling the user
    # "watching" when nothing is being captured is the failure this whole
    # feature keeps tripping over.
    thread.join(timeout=START_GRACE_SECONDS)
    if result and not result.get("success"):
        with _lock:
            _session = None
        return f"Could not start watching: {result.get('error')}"

    return (
        f"Watching for: {brief}\n"
        "Comments appear as they happen. `/watch status` to check in, "
        "`/watch stop` when you're done."
    )


def _stop() -> str:
    global _session

    with _lock:
        session = _session
        _session = None

    if session is not None and session["thread"].is_alive():
        session["stop"].set()
        session["thread"].join(timeout=10.0)
        result = session["result"]
        said = session["said"]
        lines = [
            f"Stopped after {result.get('seconds', 0):.0f}s "
            f"({result.get('ticks', 0)} frames, "
            f"{result.get('model_calls', 0)} model calls, "
            f"{result.get('spoke', 0)} said)."
        ]
        if said:
            lines.append("")
            lines.extend(f"  {line}" for line in said)
        if result.get("log_path"):
            lines.append("")
            lines.append("`/watch replay` to re-tune this session for free.")
        return "\n".join(lines)

    # Not watching — maybe recording instead.
    if rec.status().get("recording"):
        stopped = rec.stop()
        if not stopped.get("success"):
            return str(stopped.get("error"))
        return (
            f"Take saved: {stopped['video_path']}\n"
            f"{stopped['duration_seconds']:.0f}s of video, "
            f"{stopped['size_bytes'] / 1048576:.1f} MB.\n"
            "Ask about it with: hermes watch review -q \"...\""
        )

    return "Not watching and not recording."


def _status() -> str:
    with _lock:
        session = _session

    lines = []
    if session is not None and session["thread"].is_alive():
        said = session["said"]
        lines.append(f"Watching live. {len(said)} comment(s) so far.")
        lines.extend(f"  {line}" for line in said[-5:])
    else:
        lines.append("Not watching.")

    recording = rec.status()
    if recording.get("recording"):
        lines.append(
            f"Recording {recording['elapsed_seconds']:.0f}s "
            f"({recording['size_bytes'] / 1048576:.1f} MB)."
        )

    return "\n".join(lines)


def _record(label: str) -> str:
    started = rec.start(label=label)
    if not started.get("success"):
        return f"Could not start recording: {started.get('error')}"

    lines = [f"Recording to {started['video_path']}"]
    if not started.get("audio"):
        lines.append("No audio (see the notes below).")
    lines.extend(f"  {note}" for note in started.get("notes", []))
    lines.append("`/watch stop` when you're done.")
    return "\n".join(lines)


def _takes() -> str:
    rows = rec.takes(limit=10)
    if not rows:
        return "No takes recorded yet. `/watch record` to make one."
    return "\n".join(
        f"  {row['size_bytes'] / 1048576:>6.1f} MB  {row['path']}"
        + ("  +timeline" if row["has_timeline"] else "")
        for row in rows
    )


def _replay() -> str:
    directory = rec.watch_dir() / "live"
    logs = sorted(
        directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    ) if directory.is_dir() else []
    if not logs:
        return "No live sessions recorded yet. `/watch live` to make one."

    path = logs[0]
    header, rows = livemod.read_log(path)
    lines = [
        f"{path.name} ({len(rows)} ticks, brief: {header.get('brief') or '—'})",
        "",
        "  threshold  calls  said",
    ]
    for threshold in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5):
        stats = livemod.replay(path, dec.Policy(min_salience=threshold))
        lines.append(
            f"  {threshold:>9.2f}  {stats['model_calls']:>5}  {stats['spoke']:>4}"
        )
    lines.append("")
    lines.append(
        "Higher threshold means it speaks less. "
        "Set it with: hermes watch live --min-salience <n>"
    )
    return "\n".join(lines)


def shutdown() -> None:
    """Stop a live loop without reporting anything.

    For session teardown, where there is nobody left to read a summary. Bounded
    join: a stuck loop must not hold the session open.
    """
    global _session

    with _lock:
        session = _session
        _session = None

    if session is not None and session["thread"].is_alive():
        session["stop"].set()
        session["thread"].join(timeout=5.0)


def handle(raw_args: str) -> str:
    """Dispatch ``/watch <subcommand> [args]``."""
    parts = (raw_args or "").strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in {"", "help", "-h", "--help"}:
        return _HELP
    if sub == "live":
        return _start_live(rest or "anything notable about how they are performing")
    if sub == "stop":
        return _stop()
    if sub == "status":
        return _status()
    if sub == "record":
        return _record(rest)
    if sub in {"takes", "list"}:
        return _takes()
    if sub == "replay":
        return _replay()

    return f"Unknown subcommand: {sub}\n\n{_HELP}"
