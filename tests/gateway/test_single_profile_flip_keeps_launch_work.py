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
        for stale in (key, (hermes_constants.hermes_home_key(home), key)):
            core._servers.pop(stale, None)
            core._server_scope_keys.pop(stale, None)
            core._server_tool_scopes.pop(stale, None)


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


async def _native_turn_across_activation(tmp_path, profile_home):
    """Drive a Desktop/TUI ``prompt.submit`` through ``AuthorityConnection.dispatch`` on a standalone
    owner and switch the process to multi-profile hosting while the turn is running."""
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting

    started, release, seen = asyncio.Event(), asyncio.Event(), []

    async def answer(event):
        seen.append(_read())
        started.set()
        await release.wait()
        seen.append(_read())
        return "ok"

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = types.SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                                   config=GatewayConfig(multiplex_profiles=False),
                                   _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(profile_home), instance_id="owner")
    store._db.create_session("s", source="telegram")
    authority.sessions["s"] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id="c"), "s")
    connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
    try:
        await connection.dispatch({"id": 1, "method": "session.resume", "params": {"session_id": "s"}})
        reply = await connection.dispatch({"id": 2, "method": "prompt.submit", "params": {
            "session_id": "s", "submission_id": "x", "text": "hi"}})
        assert reply.get("result", {}).get("status") == "queued", reply
        await asyncio.wait_for(started.wait(), 10)
        activate_multi_profile_hosting()            # a second profile switched the gateway mid-turn
        release.set()
        await authority.sessions["s"].task
    finally:
        await connection.close()
    return seen


@pytest.mark.asyncio
async def test_native_launch_turn_in_flight_across_activation_keeps_its_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-env-key")
    seen = await _native_turn_across_activation(tmp_path, tmp_path)
    assert seen == ["launch-env-key", "launch-env-key"], seen


@pytest.mark.asyncio
async def test_native_named_profile_turn_never_reads_the_launch_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-env-key")
    bot = tmp_path / "profiles" / "bot"
    bot.mkdir(parents=True)
    seen = await _native_turn_across_activation(tmp_path, bot)
    assert seen[1] == "UnscopedSecretError", seen


@pytest.mark.asyncio
async def test_default_profiles_mcp_server_is_still_owned_after_the_flip(standalone_host):
    """After the switch the default profile's live server reads connected, a config prune and
    /reload-mcp (both select by owner) stop it, and its tools leave the registry."""
    from tools import mcp_tool as core
    from tools.mcp_tool_discovery import _adopt_server, get_mcp_status
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool_registration import _register_server_tools
    from tools.registry import registry
    from gateway.run import _profile_runtime_scope
    from hermes_constants import hermes_home_key
    runner, home = standalone_host
    cfg = {"url": "https://mcp.example/gh"}
    closed = []

    async def _shutdown():
        closed.append("github")

    tool = types.SimpleNamespace(name="t", description="d", inputSchema={"type": "object", "properties": {}},
                                 annotations=None)
    srv = types.SimpleNamespace(name="github", session=object(), _config=cfg, _tools=[tool], tool_timeout=30,
                                initialize_result=None, _registered_tool_names=[], _sampling=None,
                                _is_recycled_stdio=lambda: False, shutdown=_shutdown)
    _adopt_server("github", srv)                    # single-profile discovery: bare key, owner None
    srv._registered_tool_names = _register_server_tools("github", srv, cfg)
    try:
        _mkprofile(home, "daily-news-bot")
        await runner.reconcile_served_profiles(reason="control-socket")
        with _profile_runtime_scope(home):
            from tools.mcp_tool import _mcp_registry_scope
            scope = _mcp_registry_scope()
            assert get_mcp_status({"github": cfg})[0]["status"] == "connected"
            assert registry.get_entry("mcp__github__t") is not None
            from tools import mcp_tool_loop as loop
            loop._ensure_mcp_loop()
            try:
                shutdown_mcp_servers(scope=scope)   # /reload-mcp's teardown for this profile
            finally:
                loop._stop_mcp_loop()
            srv_tools = list(srv._registered_tool_names)
            from tools.mcp_tool_registration import _deregister_mcp_tool_all_scopes
            for name in srv_tools:                  # the task's own _deregister_tools on shutdown
                _deregister_mcp_tool_all_scopes(srv, name)
        assert closed == ["github"], "the reload skipped the default profile's server"
        assert (hermes_home_key(home), "github") not in core._servers
        assert registry.get_entry("mcp__github__t", scope=hermes_home_key(home)) is None
        assert "mcp__github__t" not in registry._tools
    finally:
        for key in [k for k in core._servers if (k[1] if isinstance(k, tuple) else k) == "github"]:
            core._servers.pop(key, None)
            core._server_scope_keys.pop(key, None)
            core._server_tool_scopes.pop(key, None)
        for slot in (None, hermes_home_key(home)):
            registry.deregister("mcp__github__t", scope=slot)
