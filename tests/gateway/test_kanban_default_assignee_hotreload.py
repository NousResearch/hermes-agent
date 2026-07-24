"""Regression test for #55446 — kanban.default_assignee hot-reload.

Verifies that the dispatcher re-reads ``kanban.default_assignee`` from
config on each dispatch tick instead of capturing it once at startup.
"""
import asyncio
import inspect
import time

import pytest


def test_dispatcher_rereads_default_assignee_on_each_tick(monkeypatch, tmp_path):
    """Config changes to default_assignee take effect on the next tick."""
    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb

    # --- state: config starts with empty default_assignee ---
    cfg_state = {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "default_assignee": "",
        }
    }
    captured_assignees: list = []
    tick_count = {"dispatch": 0}

    monkeypatch.setattr(_cfg_mod, "load_config", lambda: cfg_state)
    monkeypatch.setattr(
        _kb, "list_boards",
        lambda include_archived=False: [{"slug": _kb.DEFAULT_BOARD}],
    )
    monkeypatch.setattr(
        _kb, "read_board_metadata",
        lambda slug: {"slug": slug},
    )
    monkeypatch.setattr(
        _kb, "kanban_db_path",
        lambda board=None: tmp_path / "kanban.db",
    )
    # Stub out reap_worker_zombies so it doesn't interfere
    monkeypatch.setattr(_kb, "reap_worker_zombies", lambda: [])

    real_dispatch = _kb.dispatch_once

    def _capture_dispatch(conn, **kwargs):
        captured_assignees.append(kwargs.get("default_assignee"))
        return None

    monkeypatch.setattr(_kb, "dispatch_once", _capture_dispatch)

    # --- time travel ---
    real_monotonic = time.monotonic
    time_values = iter([1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0])

    def _monotonic_for_dispatcher():
        caller = inspect.currentframe().f_back  # type: ignore[union-attr]
        code = caller.f_code if caller is not None else None
        filename = code.co_filename if code is not None else ""
        if filename.endswith("gateway/kanban_watchers.py"):
            return next(time_values, 1005.0)
        return real_monotonic()

    monkeypatch.setattr(
        "gateway.kanban_watchers.time.monotonic", _monotonic_for_dispatcher
    )

    real_to_thread = asyncio.to_thread

    async def _to_thread(fn, *args, **kwargs):
        fn_name = getattr(fn, "__name__", "")
        result = await real_to_thread(fn, *args, **kwargs)
        # Only count _tick_once calls (the main dispatch).
        if fn_name == "_tick_once":
            tick_count["dispatch"] += 1
            # After the first dispatch tick, change config.
            if tick_count["dispatch"] == 1:
                cfg_state["kanban"]["default_assignee"] = "indigo"
            # After the second dispatch tick, stop the loop.
            if tick_count["dispatch"] >= 2:
                runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _sleep)

    runner = object.__new__(GatewayRunner)
    runner._running = True

    asyncio.run(
        asyncio.wait_for(
            runner._kanban_dispatcher_watcher(),
            timeout=5.0,
        )
    )

    # First tick: empty string → None (startup value).
    # Second tick: config changed to "indigo" → re-read picks it up.
    assert len(captured_assignees) == 2, (
        f"Expected 2 dispatch_once calls, got {len(captured_assignees)}"
    )
    assert captured_assignees[0] is None, (
        f"First tick should use startup value (None), got {captured_assignees[0]!r}"
    )
    assert captured_assignees[1] == "indigo", (
        f"Second tick should re-read config ('indigo'), got {captured_assignees[1]!r}"
    )


def test_dispatcher_assignee_fallback_on_config_failure(monkeypatch, tmp_path):
    """When config reload fails, the startup default_assignee is preserved."""
    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb

    call_count = {"n": 0}
    captured_assignees: list = []

    def _load_config_with_failure():
        call_count["n"] += 1
        if call_count["n"] <= 1:
            # First call (startup): return valid config with an assignee.
            return {
                "kanban": {
                    "dispatch_in_gateway": True,
                    "dispatch_interval_seconds": 1,
                    "default_assignee": "startup-user",
                }
            }
        # Subsequent calls (tick re-read): simulate config failure.
        raise OSError("config file vanished")

    monkeypatch.setattr(_cfg_mod, "load_config", _load_config_with_failure)
    monkeypatch.setattr(
        _kb, "list_boards",
        lambda include_archived=False: [{"slug": _kb.DEFAULT_BOARD}],
    )
    monkeypatch.setattr(
        _kb, "read_board_metadata",
        lambda slug: {"slug": slug},
    )
    monkeypatch.setattr(
        _kb, "kanban_db_path",
        lambda board=None: tmp_path / "kanban.db",
    )
    monkeypatch.setattr(_kb, "reap_worker_zombies", lambda: [])

    def _capture_dispatch(conn, **kwargs):
        captured_assignees.append(kwargs.get("default_assignee"))
        return None

    monkeypatch.setattr(_kb, "dispatch_once", _capture_dispatch)

    real_monotonic = time.monotonic
    time_values = iter([1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0])

    def _monotonic_for_dispatcher():
        caller = inspect.currentframe().f_back  # type: ignore[union-attr]
        code = caller.f_code if caller is not None else None
        filename = code.co_filename if code is not None else ""
        if filename.endswith("gateway/kanban_watchers.py"):
            return next(time_values, 1005.0)
        return real_monotonic()

    monkeypatch.setattr(
        "gateway.kanban_watchers.time.monotonic", _monotonic_for_dispatcher
    )

    tick_count = {"dispatch": 0}
    real_to_thread = asyncio.to_thread

    async def _to_thread(fn, *args, **kwargs):
        fn_name = getattr(fn, "__name__", "")
        result = await real_to_thread(fn, *args, **kwargs)
        if fn_name == "_tick_once":
            tick_count["dispatch"] += 1
            if tick_count["dispatch"] >= 2:
                runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _sleep)

    runner = object.__new__(GatewayRunner)
    runner._running = True

    asyncio.run(
        asyncio.wait_for(
            runner._kanban_dispatcher_watcher(),
            timeout=5.0,
        )
    )

    assert len(captured_assignees) == 2
    # First tick: startup value.
    assert captured_assignees[0] == "startup-user"
    # Second tick: config failed → fallback to startup value.
    assert captured_assignees[1] == "startup-user"
