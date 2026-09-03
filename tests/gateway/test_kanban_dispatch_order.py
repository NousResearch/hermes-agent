"""The dispatcher must claim ready work before awaited decomposition."""

import asyncio


def test_ready_dispatch_precedes_auto_decompose(monkeypatch, tmp_path):
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_decompose as decomp
    import hermes_cli.config as config_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    calls = []

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": True,
                "auto_decompose_per_tick": 1,
            }
        },
    )
    monkeypatch.setattr(
        kb,
        "list_boards",
        lambda include_archived=False: [{"slug": "default"}],
    )
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(
        kb, "dispatch_once", lambda *args, **kwargs: calls.append("dispatch")
    )
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: False)
    monkeypatch.setattr(decomp, "list_triage_ids", lambda: ["task_exact_head"])

    def _decompose(*args, **kwargs):
        calls.append("decompose")
        runner._running = False
        return decomp.DecomposeOutcome("task_exact_head", False, "test stop")

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(decomp, "decompose_task", _decompose)
    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)

    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3.0))

    assert calls[:2] == ["dispatch", "decompose"]
