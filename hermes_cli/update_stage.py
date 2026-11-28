"""Stage updates to a desktop hand-off UI, if one is watching.

The desktop-update shim (scripts/desktop-update/posix.sh) renders update
progress from a status JSON file. The shim only writes stages it gates on
itself; the takeover children (_update_takeover.py, update_finish.py and the
PM sync / build stages they drive) can run for minutes with nothing on
screen. These helpers let long stages publish through the same file so the
window keeps moving.

Two discovery paths, because the status file's name embeds the shim's pid:

- ``HERMES_UPDATE_STATUS_FILE``: exported by the current posix.sh — covers
  new-shim → new-children directly.
- Marker fallback: an OLD shim (pre-env-var checkout, i.e. every old→new
  update) never exports the path, but it writes the marker whose first line
  is its own pid, and its status file is deterministic:
  ``${TMPDIR:-/tmp}/hermes-update-status.<pid>``.

Stdlib-only (importable under ``-I -S -B``) and never raises: a missing/
stale/unwritable file means no UI, which is exactly the pre-shim behavior —
a progress hiccup must never fail an update.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

STATUS_FILE_ENV = "HERMES_UPDATE_STATUS_FILE"
UI_SPAWNED_ENV = "HERMES_UPDATE_UI_ACTIVE"
MARKER_NAME = ".hermes-update-in-progress"
#: Well-known tail progress file clients poll while a CLI-driven update runs.
#: Unlike the shim's pid-suffixed file, the name is fixed so a client that was
#: never handed a path (the Desktop overlay, `hermes update --status`) can
#: still show the tail moving. Same atomic-replace discipline as the shim file.
TAIL_STAGE_NAME = "update_stage.json"
#: A tail that reports nothing for this long reads as stuck in the clients.
TAIL_STAGE_STALE_SECONDS = 5 * 60


def _process_home() -> Path:
    """The marker and the shim's log live in the PROCESS home (update_lock.update_marker_path):
    the shim resolved ``$HERMES_HOME`` or the platform default, never a profile override, and
    the platform default (sudo invoker, data-dir suffix) is not ``~/.hermes`` everywhere."""
    from hermes_constants import get_process_hermes_home
    return get_process_hermes_home()


def status_file() -> Path | None:
    """The watching shim's status file, or None when no UI is discoverable."""
    exported = os.environ.get(STATUS_FILE_ENV, "").strip()
    if exported:
        return Path(exported)
    return _status_from_marker()


def _status_from_marker() -> Path | None:
    """Derive the shim's status file from the update marker's owner pid.

    The marker's first line is the hand-off shim's ``$$`` (update_lock.py
    adopts rather than rewrites it, so it still names the shim here). The
    shim's status file lives beside it, pid-suffixed. No marker (plain CLI
    update) or no matching file → no UI.
    """
    try:
        pid = int((_process_home() / MARKER_NAME).read_text(encoding="utf-8-sig")
                  .splitlines()[0].strip())
    except (OSError, ValueError, IndexError):
        return None
    if pid <= 0:
        return None
    # The shim writes beside ${TMPDIR:-/tmp}; gettempdir() resolves the same way.
    candidate = Path(tempfile.gettempdir()) / f"hermes-update-status.{pid}"
    return candidate if candidate.is_file() else None


def publish_stage(message: str) -> None:
    """Report a running stage to the watching UI. Best-effort, never raises.

    Only ever writes ``status=running``: terminal states belong to the shim,
    which publishes them after this process has exited. The atomic replace
    mirrors the shim's write_status — a polling reader must never observe a
    half-written file.
    """
    path = status_file()
    if path is None:
        return
    payload = json.dumps({"status": "running", "message": message})
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".update-stage-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except OSError:
        pass


def tail_stage_file() -> Path | None:
    """The well-known tail progress file, or None when the home is unwritable.

    Same root as the update receipts (the ROOT home, never a profile's), so
    every client knows where to look without being handed a path.
    """
    try:
        from hermes_constants import get_default_hermes_root

        logs = get_default_hermes_root() / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        return logs / TAIL_STAGE_NAME
    except Exception:
        # Never raises: exotic roots (None from a stubbed home, unmounted
        # drives) mean no UI, not a failed update.
        return None


def _utcnow_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def publish_tail_stage(message: str, *, status: str = "running") -> None:
    """Report tail progress to polling clients. Best-effort, never raises.

    Writes ``{status, message, at}`` with an atomic replace; the ``at``
    timestamp is the heartbeat — a reader compares it against
    :data:`TAIL_STAGE_STALE_SECONDS` to tell "working" from "wedged".
    Terminal states (``status="done"``) carry no staleness warning.
    A missing/unwritable file degrades to the pre-client behavior (a silent
    tail), never to a failed update.
    """
    path = tail_stage_file()
    if path is None:
        return
    payload = json.dumps({"status": status, "message": message, "at": _utcnow_iso()})
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".update-tail-stage-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except OSError:
        pass


def clear_tail_stage() -> None:
    """Remove the tail progress file at the end of a run. Best-effort, never raises."""
    path = tail_stage_file()
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def read_tail_stage() -> dict | None:
    """The last published tail stage with its age, or None.

    Returns ``{"status", "message", "at", "age_seconds"}``; a missing or
    unparseable file is None (a tail that never published, or a finished run
    that cleared it). Never raises.
    """
    import datetime

    path = tail_stage_file()
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    age: float | None = None
    try:
        at = datetime.datetime.fromisoformat(str(data.get("at", "")))
        if at.tzinfo is None:
            at = at.replace(tzinfo=datetime.UTC)
        age = (datetime.datetime.now(datetime.UTC) - at).total_seconds()
    except ValueError:
        pass
    return {
        "status": data.get("status", "running"),
        "message": data.get("message", ""),
        "at": data.get("at"),
        "age_seconds": age,
    }


def _ui_present_in_log(log_text: str) -> bool:
    """True when THIS hand-off run started a browser/panel window.

    Reads the shim's own log lines from the tail (the run's shim logs 'shim:
    app window …' or 'shim: status panel pid=…' when a renderer exists,
    '… skipping UI' when it does not). Unparseable → False: spawning a
    duplicate panel is cosmetic; leaving a Safari user with no UI at all is
    the bug being fixed.
    """
    tail = "\n".join(log_text.splitlines()[-40:])
    return "app window" in tail or "status panel pid=" in tail


def ensure_panel(update_root: Path) -> None:
    """Pop the native status panel when no renderer is up (macOS, old→new).

    An old shim has no panel of its own and skipped its browser window on a
    non-Chromium default — its whole update then runs invisible. The freshly
    pulled tree ships update-panel.js, so the takeover child can
    start it against the discovered status file. Best-effort end to end: any
    failure leaves the update running exactly as UI-less as before.
    """
    if os.environ.get(UI_SPAWNED_ENV) or os.name != "posix" or os.uname().sysname != "Darwin":
        return
    if status_file() is None:
        return
    panel = update_root / "scripts" / "desktop-update" / "update-panel.js"
    if not panel.is_file():
        return
    try:
        log = _process_home() / "logs" / "desktop-update-handoff.log"
        if log.is_file() and _ui_present_in_log(log.read_text(encoding="utf-8-sig", errors="replace")[-8000:]):
            return
        import subprocess

        # Own session + devnull stdio: the panel must outlive this child the
        # same way the shim-spawned UI outlives the shim's stages, and it
        # self-exits when the shim publishes a terminal state.
        subprocess.Popen(
            ["/usr/bin/osascript", "-l", "JavaScript", str(panel), str(status_file())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            # detached-spawn: ok — update progress panel, self-exits on the terminal state
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        os.environ[UI_SPAWNED_ENV] = "1"  # inherited: never spawn twice per chain
    except OSError:
        pass
