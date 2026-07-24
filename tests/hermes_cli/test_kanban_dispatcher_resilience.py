from __future__ import annotations

import asyncio
import inspect
import logging
import sqlite3
import time
from types import SimpleNamespace


class _DummyConn:
    def close(self) -> None:
        pass


def _install_gateway_dispatcher_harness(
    monkeypatch,
    tmp_path,
    *,
    dispatch_once,
    stop_after_ticks: int,
    monotonic_values: list[float] | None = None,
    after_tick=None,
):
    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb

    db_path = tmp_path / "kanban.db"
    runner = object.__new__(GatewayRunner)
    runner._running = True
    calls = {"ticks": 0}

    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
            }
        },
    )
    monkeypatch.setattr(
        _kb,
        "list_boards",
        lambda include_archived=False: [{"slug": _kb.DEFAULT_BOARD}],
    )
    monkeypatch.setattr(_kb, "read_board_metadata", lambda slug: {"slug": slug})
    monkeypatch.setattr(_kb, "kanban_db_path", lambda board=None: db_path)
    monkeypatch.setattr(_kb, "connect", lambda *args, **kwargs: _DummyConn())
    monkeypatch.setattr(_kb, "dispatch_once", dispatch_once)

    async def _to_thread(fn, *args, **kwargs):
        name = getattr(fn, "__name__", "")
        if name == "reap_worker_zombies":
            return []
        if name == "_ready_nonempty":
            return False
        result = fn(*args, **kwargs)
        if name == "_tick_once":
            calls["ticks"] += 1
            if after_tick is not None:
                after_tick(calls["ticks"])
            if calls["ticks"] >= stop_after_ticks:
                runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)

    if monotonic_values is not None:
        values = iter(monotonic_values)
        fallback = monotonic_values[-1]
        real_monotonic = time.monotonic

        def _monotonic_for_gateway_dispatcher():
            caller = inspect.currentframe().f_back  # type: ignore[union-attr]
            code = caller.f_code if caller is not None else None
            filename = code.co_filename if code is not None else ""
            function_name = code.co_name if code is not None else ""
            if filename.replace("\\", "/").endswith(
                "gateway/kanban_watchers.py"
            ) and function_name in {
                "_record_confirmed_corrupt_board",
                "_tick_once_for_board",
            }:
                return next(values, fallback)
            return real_monotonic()

        monkeypatch.setattr(
            "gateway.kanban_watchers.time.monotonic",
            _monotonic_for_gateway_dispatcher,
        )

    return runner, db_path, calls


def _run_watcher(runner) -> None:
    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3.0))


def test_gateway_dispatcher_quick_check_ok_does_not_latch(
    monkeypatch, tmp_path, caplog
):
    dispatch_calls = {"count": 0}

    def _dispatch_once(*args, **kwargs):
        dispatch_calls["count"] += 1
        raise sqlite3.DatabaseError("database disk image is malformed")

    runner, db_path, calls = _install_gateway_dispatcher_harness(
        monkeypatch,
        tmp_path,
        dispatch_once=_dispatch_once,
        stop_after_ticks=2,
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE marker(id INTEGER)")

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_watcher(runner)

    messages = [record.getMessage() for record in caplog.records]
    assert calls["ticks"] == 2
    assert dispatch_calls["count"] == 2
    assert not any("not a valid SQLite database" in msg for msg in messages)
    assert sum("read-only quick_check passed" in msg for msg in messages) == 2


def test_gateway_dispatcher_confirmed_corruption_doubles_backoff(
    monkeypatch, tmp_path, caplog
):
    dispatch_calls = {"count": 0}

    def _dispatch_once(*args, **kwargs):
        dispatch_calls["count"] += 1
        raise sqlite3.DatabaseError("database disk image is malformed")

    runner, db_path, calls = _install_gateway_dispatcher_harness(
        monkeypatch,
        tmp_path,
        dispatch_once=_dispatch_once,
        stop_after_ticks=3,
        monotonic_values=[1000.0, 1029.0, 1030.0, 1030.0],
    )
    db_path.write_text("not sqlite", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _run_watcher(runner)

    messages = [record.getMessage() for record in caplog.records]
    assert calls["ticks"] == 3
    assert dispatch_calls["count"] == 2
    assert any("for 30s" in msg for msg in messages)
    assert any("for 60s" in msg for msg in messages)
    assert any("after 30s corrupt-board backoff" in msg for msg in messages)


def test_gateway_dispatcher_fingerprint_change_retries_and_resets_backoff(
    monkeypatch, tmp_path, caplog
):
    dispatch_calls = {"count": 0}

    def _dispatch_once(*args, **kwargs):
        dispatch_calls["count"] += 1
        raise sqlite3.DatabaseError("database disk image is malformed")

    def _after_tick(tick: int) -> None:
        if tick == 1:
            db_path.write_text("not sqlite but changed", encoding="utf-8")

    runner, db_path, calls = _install_gateway_dispatcher_harness(
        monkeypatch,
        tmp_path,
        dispatch_once=_dispatch_once,
        stop_after_ticks=2,
        monotonic_values=[1000.0, 1001.0, 1001.0],
        after_tick=_after_tick,
    )
    db_path.write_text("not sqlite", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _run_watcher(runner)

    messages = [record.getMessage() for record in caplog.records]
    assert calls["ticks"] == 2
    assert dispatch_calls["count"] == 2
    assert sum("for 30s" in msg for msg in messages) == 2
    assert not any("for 60s" in msg for msg in messages)
    assert any("database changed; retrying dispatch" in msg for msg in messages)


def test_gateway_dispatcher_success_clears_disabled_board_state(
    monkeypatch, tmp_path, caplog
):
    dispatch_calls = {"count": 0}

    def _dispatch_once(*args, **kwargs):
        dispatch_calls["count"] += 1
        if dispatch_calls["count"] in {1, 3}:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return SimpleNamespace(spawned=[])

    runner, db_path, calls = _install_gateway_dispatcher_harness(
        monkeypatch,
        tmp_path,
        dispatch_once=_dispatch_once,
        stop_after_ticks=3,
        monotonic_values=[1000.0, 1030.0, 1031.0],
    )
    db_path.write_text("not sqlite", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _run_watcher(runner)

    messages = [record.getMessage() for record in caplog.records]
    assert calls["ticks"] == 3
    assert dispatch_calls["count"] == 3
    assert sum("for 30s" in msg for msg in messages) == 2
    assert not any("for 60s" in msg for msg in messages)
