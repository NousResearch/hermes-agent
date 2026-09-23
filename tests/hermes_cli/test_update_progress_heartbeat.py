"""A quiet update step must stay visible to the Desktop idle watchdog.

The Windows hand-off cancels ``hermes update`` (exit 124) after 600s with no
stdout/stderr and no ``logs/update.log`` growth. npm installs with
``--progress=false`` produce neither while they are still working.
"""
import io
import sys
import threading

from hermes_constants import get_hermes_home
from hermes_cli.update_cmd import _update_progress_heartbeat


def test_heartbeat_prints_while_the_step_is_still_running(monkeypatch):
    """The tick has to land before the step returns, or the watchdog already fired."""
    import builtins

    seen = threading.Event()
    real_print = builtins.print

    def _print(*args, **kwargs):
        if args and str(args[0]).startswith("still ("):
            seen.set()
        real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", _print)
    with _update_progress_heartbeat("still ({elapsed}s)", interval_seconds=0.05):
        assert seen.wait(5), "no progress line while the step was still inside the heartbeat"


def test_unwrapped_stdout_grows_update_log_before_the_step_returns(monkeypatch, tmp_path):
    """Without the stdout mirror, the tick still has to grow the file the watchdog stats."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    log_path = get_hermes_home() / "logs" / "update.log"
    found = threading.Event()

    def _watch() -> None:
        while not found.is_set():
            try:
                text = log_path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            if "still (" in text:
                found.set()
                return
            if found.wait(0.05):
                return

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        with _update_progress_heartbeat("still ({elapsed}s)", interval_seconds=0.05):
            assert found.wait(5), "update.log did not grow while the quiet step was still running"
    finally:
        found.set()
        watcher.join(timeout=2)
