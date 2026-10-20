"""A single-profile gateway that starts multiplexing in place for a newly created profile must keep
the DEFAULT profile's own work running: keys injected only into the process env (docker -e, systemd
Environment=, op run) have no .env line, so every launch-home scope binds the launch secret scope
(env frozen at activation + .env), and the launch profile's live MCP connections stay resolvable."""
import asyncio
import types

import pytest

import hermes_constants
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.status import flush_runtime_status, write_runtime_status
from hermes_cli import gateway_migrate as gm
from hermes_cli.gateway_multiplex_mode import SINGLE_PROFILE_REASON


@pytest.fixture
def standalone_host(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    (home / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(gm, "_installed_services", lambda _home: [])
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda _home: None)
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    # Provider key injected by the service manager (docker -e / systemd Environment=), no .env line.
    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-env-key")
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False))
    runner._running = True
    runner.session_runtime_descriptor = {"state": "ready", "served_profiles": [], "capabilities": []}
    runner.session_ticket_store = None
    write_runtime_status(gateway_state="running", multiplex_standalone_reason=SINGLE_PROFILE_REASON,
                         served_profiles=[])
    flush_runtime_status()
    yield runner, home
    from gateway.runtime_ownership import process_ownership
    for profile in (home / "profiles").iterdir():
        process_ownership.release(profile.resolve())


def _mkprofile(home, name):
    d = home / "profiles" / name
    d.mkdir()
    (d / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    (d / ".env").write_text("OPENROUTER_API_KEY=bot-key\n", encoding="utf-8")
    return d


def _read():
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        return get_secret("OPENROUTER_API_KEY")
    except UnscopedSecretError:
        return "UnscopedSecretError"


@pytest.mark.asyncio
async def test_default_turn_in_flight_across_the_flip_keeps_its_credentials(standalone_host):
    runner, home = standalone_host
    started, release, seen = asyncio.Event(), asyncio.Event(), []

    async def fake_handle(event):
        seen.append(_read())
        started.set()
        await release.wait()          # a tool call / provider retry later in the same turn
        seen.append(_read())

    runner._handle_message = fake_handle
    handler = runner._primary_message_handler()     # what the adapter holds before activation
    turn = asyncio.create_task(handler(types.SimpleNamespace(source=None)))
    await started.wait()
    _mkprofile(home, "daily-news-bot")
    result = await runner.reconcile_served_profiles(reason="control-socket")
    assert result["added"] == ["daily-news-bot"]
    release.set()
    await turn
    assert seen == ["launch-env-key", "launch-env-key"], seen


@pytest.mark.asyncio
async def test_default_turn_after_the_flip_keeps_env_injected_credentials(standalone_host):
    runner, home = standalone_host
    seen = []

    async def fake_handle(event):
        seen.append(_read())

    runner._handle_message = fake_handle
    runner.adapters = {"fake": types.SimpleNamespace(
        set_message_handler=lambda h: setattr(runner, "_rewired", h),
        set_busy_session_handler=lambda h: None, set_platform_event_handler=lambda h: None)}
    _mkprofile(home, "daily-news-bot")
    await runner.reconcile_served_profiles(reason="control-socket")
    src = types.SimpleNamespace(platform=None, profile=None, chat_id="1", user_id="u")
    runner._admit_primary_source = lambda source, default_home: default_home
    await runner._rewired(types.SimpleNamespace(source=src))
    assert seen == ["launch-env-key"], seen


@pytest.mark.asyncio
async def test_default_profiles_mcp_connections_survive_the_flip(standalone_host):
    from tools import mcp_tool as core
    from tools.mcp_tool_handlers import _make_check_fn
    from tools.mcp_tool_scope import _server_key
    runner, home = standalone_host
    key = _server_key("github")                      # single-profile discovery keys by bare name
    core._servers[key] = types.SimpleNamespace(session=object(), _is_recycled_stdio=lambda: False)
    try:
        check = _make_check_fn("github")
        assert check()
        _mkprofile(home, "daily-news-bot")
        await runner.reconcile_served_profiles(reason="control-socket")
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(home):           # the re-wired primary handler's scope
            assert check(), "default profile's live MCP server vanished after the flip"
    finally:
        core._servers.pop(key, None)


def test_default_cron_fire_and_kanban_tick_after_the_flip_keep_env_injected_credentials(standalone_host):
    from cron.scheduler import _fire_secret_scope
    from gateway.kanban_watchers_dispatcher import _default_profile_secret_scope
    runner, home = standalone_host
    _mkprofile(home, "daily-news-bot")
    asyncio.run(runner.reconcile_served_profiles(reason="control-socket"))
    with _fire_secret_scope():
        assert _read() == "launch-env-key"
    with _default_profile_secret_scope():
        assert _read() == "launch-env-key"


@pytest.mark.asyncio
async def test_activation_does_not_run_once_shutdown_began(standalone_host):
    from agent.secret_scope import is_multiplex_active
    runner, home = standalone_host
    _mkprofile(home, "daily-news-bot")
    runner._running = False            # stop() flips this first
    await runner.reconcile_served_profiles(reason="control-socket")
    assert not is_multiplex_active(), "a shutting-down gateway armed fail-closed secrets"


@pytest.mark.asyncio
async def test_concurrent_rescans_keep_the_new_profiles_reservation(standalone_host, monkeypatch):
    """A watcher tick and a creator's rescan both start the switch. The slower preflight resuming
    while the faster rescan is still serving the bot must not reset the reservation to just the
    default home (routing and cron would then miss a bot whose authority is live)."""
    from gateway import run_profile_reconcile as rpr
    runner, home = standalone_host
    preflights = []
    housekeeping = runner._run_housekeeping_in_executor

    async def run_housekeeping(func, *args):
        if func is rpr._added_profile_blocker:
            preflights.append(func)
            if len(preflights) > 1:
                await reserved_bot.wait()  # resumes while the first rescan is mid-serve
                return None
        return await housekeeping(func, *args)

    serve = runner._serve_profile_runtime
    reserved_bot = asyncio.Event()

    async def slow_serve(name, path):
        reason = await serve(name, path)
        reserved_bot.set()
        for _ in range(20):          # adapters still connecting: the other rescan gets the loop
            await asyncio.sleep(0)
        return reason

    runner._run_housekeeping_in_executor = run_housekeeping
    runner._serve_profile_runtime = slow_serve
    bot = _mkprofile(home, "daily-news-bot").resolve()
    await asyncio.gather(runner.reconcile_served_profiles(reason="control-socket"),
                         runner.reconcile_served_profiles(reason="watcher"))
    reserved = {name: path for name, path in runner.config._runtime_profile_homes}
    assert reserved.get("daily-news-bot") == bot, reserved


@pytest.mark.asyncio
async def test_activation_rechecks_shutdown_after_the_preflight(standalone_host, monkeypatch):
    from agent.secret_scope import is_multiplex_active
    from gateway import run_profile_reconcile as rpr
    runner, home = standalone_host
    monkeypatch.setattr(rpr, "_added_profile_blocker", lambda _launch: setattr(runner, "_running", False))
    _mkprofile(home, "daily-news-bot")
    await runner.reconcile_served_profiles(reason="control-socket")
    assert not is_multiplex_active(), "stop() began during the preflight, yet fail-closed secrets were armed"
