"""The CLI's startup housekeeping (curator pass, skill sync) must never hold the prompt.

A due weekly curator pass on a large library snapshotted and pruned the skills tree for
six minutes on the main thread, between the banner and the input box.
"""

import threading
import time
from unittest.mock import MagicMock

import agent.curator as curator_mod


def test_startup_maintenance_returns_while_curator_still_running(monkeypatch):
    from cli import HermesCLI

    release, entered = threading.Event(), threading.Event()
    seen = {}

    def _slow_curator(**_kw):
        seen["thread"] = threading.current_thread()
        entered.set()
        release.wait(10)

    monkeypatch.setattr(curator_mod, "maybe_run_curator", _slow_curator)
    obj = HermesCLI.__new__(HermesCLI)
    obj._console_print = MagicMock()
    started = time.monotonic()
    try:
        obj._tui_startup_background_maintenance()
        elapsed = time.monotonic() - started
        assert entered.wait(5), "curator pass never started"
    finally:
        release.set()
    assert elapsed < 5, f"startup maintenance held the caller for {elapsed:.1f}s"
    assert seen.get("thread") is not threading.main_thread()
