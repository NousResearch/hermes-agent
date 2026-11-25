"""Reset the Bot Desktop lease between tests: no listeners in this process, no lease files under the
current HERMES_HOME."""
import contextlib

from tools.bot_desktop import lease


def reset_lease() -> None:
    with lease._lock:
        lease._listeners.clear()
    p = lease._path(None)
    for f in (p, p.with_suffix(".lock"), p.with_suffix(".json.tmp")):
        with contextlib.suppress(OSError):
            f.unlink()
