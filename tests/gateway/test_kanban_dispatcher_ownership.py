import asyncio

from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


def _reset_kanban_env(monkeypatch, home):
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DISPATCH_IN_GATEWAY",
    ):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()


def test_dispatchable_boards_for_profile_includes_owned_project_only(tmp_path, monkeypatch):
    _reset_kanban_env(monkeypatch, tmp_path)
    kb.create_board("owned", dispatch_owner="coordinator_one")
    kb.create_board("other", dispatch_owner="coordinator_two")

    boards = kb.dispatchable_boards_for_profile("coordinator_one")

    assert [b["slug"] for b in boards] == ["owned"]


def test_dispatchable_boards_for_default_profile_skips_project_boards(tmp_path, monkeypatch):
    _reset_kanban_env(monkeypatch, tmp_path)
    kb.create_board("project", dispatch_owner="coordinator_project")

    boards = kb.dispatchable_boards_for_profile("default")

    assert [b["slug"] for b in boards] == ["default"]


def test_dispatchable_boards_missing_ownership_safe_skips_named_project(tmp_path, monkeypatch):
    _reset_kanban_env(monkeypatch, tmp_path)
    kb.create_board("legacy-project")

    boards = kb.dispatchable_boards_for_profile("coordinator_one")

    assert [b["slug"] for b in boards] == []


def test_dispatchable_boards_default_board_owner_overrides_legacy_default(tmp_path, monkeypatch):
    _reset_kanban_env(monkeypatch, tmp_path)
    kb.write_board_metadata("default", dispatch_owner="coordinator_admin")

    assert [b["slug"] for b in kb.dispatchable_boards_for_profile("default")] == []
    assert [b["slug"] for b in kb.dispatchable_boards_for_profile("coordinator_admin")] == ["default"]


async def _run_one_dispatcher_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_dispatcher_watcher()


def test_gateway_dispatcher_dispatches_only_boards_owned_by_active_profile(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    active_home = root / "profiles" / "coordinator_one"
    active_home.mkdir(parents=True)
    _reset_kanban_env(monkeypatch, active_home)
    kb.create_board("owned", dispatch_owner="coordinator_one")
    kb.create_board("other", dispatch_owner="coordinator_two")
    kb.create_board("missing-owner")

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1, "auto_decompose": False}},
    )
    dispatched = []

    def fake_dispatch_once(conn, *, board, **kwargs):
        dispatched.append(board)
        return None

    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True

    asyncio.run(_run_one_dispatcher_tick(monkeypatch, runner))

    assert dispatched == ["owned"]


def test_gateway_dispatcher_coordinator_gateways_do_not_fight_each_other(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    kb_home = root / "profiles" / "coordinator_one"
    kb_home.mkdir(parents=True)
    _reset_kanban_env(monkeypatch, kb_home)
    kb.create_board("one", dispatch_owner="coordinator_one")
    kb.create_board("two", dispatch_owner="coordinator_two")

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1, "auto_decompose": False}},
    )
    dispatched = []

    def fake_dispatch_once(conn, *, board, **kwargs):
        dispatched.append(board)
        return None

    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True

    asyncio.run(_run_one_dispatcher_tick(monkeypatch, runner))

    assert dispatched == ["one"]
