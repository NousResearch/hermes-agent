"""Real discovery and dispatch receipts for session-aware native slash callbacks."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ensure_current_event_loop():
    """Override the shared sync-loop fixture: dispatch owns its asyncio.run loops,
    and gateway tests use pytest-asyncio. An implicit policy loop is unused here
    and can be orphaned when asyncio.run clears the current loop.
    """
    yield


def install_plugin(home):
    plugin = home / "plugins" / "owner-probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: owner-probe\nversion: 1.0.0\n")
    (home / "config.yaml").write_text("plugins:\n  enabled: [owner-probe]\n")
    (plugin / "__init__.py").write_text('''
import asyncio
import json
from hermes_constants import get_hermes_home

def register(ctx):
    registered_home = str(get_hermes_home())
    async def owner(raw, **kwargs):
        await asyncio.sleep(0.01)
        return json.dumps(dict(raw=raw, registered_home=registered_home,
                               active_home=str(get_hermes_home()), **kwargs), default=str)
    ctx.register_command("owner-probe", owner)
    ctx.register_command("owner-legacy", lambda raw: "legacy:" + raw)
''')


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
@pytest.mark.parametrize("source,lazy", [("desktop", False), ("tui", True)])
def test_desktop_dispatch_owns_each_profile_and_session_through_async_thread(tmp_path, monkeypatch, method, source, lazy):
    from hermes_constants import get_hermes_home
    from tui_gateway import server

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ambient = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    sessions = {}
    for name in ("a", "b"):
        home = ambient / "profiles" / name
        install_plugin(home)
        sessions[name] = {
            "profile_home": str(home), "session_key": "stored-" + name,
            "source": source, "agent": None if lazy else SimpleNamespace(session_id="compressed-" + name),
            "running": False,
        }
        monkeypatch.setitem(server._sessions, "runtime-" + name, sessions[name])

    def dispatch(name):
        async def in_running_loop():
            params = {"session_id": "runtime-" + name, "command": "/owner-probe On",
                      "name": "owner-probe", "arg": "On"}
            return server._methods[method]("request-" + name, params)
        return asyncio.run(in_running_loop())

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(dispatch, ["a", "b", "a", "b"]))
    for name, result in zip(["a", "b", "a", "b"], results):
        assert "result" in result, result
        payload = json.loads(result["result"]["output"])
        home = sessions[name]["profile_home"]
        assert payload == {
            "raw": "On", "registered_home": home, "active_home": home,
            "session_id": ("stored-" if lazy else "compressed-") + name, "task_id": "stored-" + name,
            "runtime_session_id": "runtime-" + name, "stored_session_id": "stored-" + name,
            "profile": name, "hermes_home": home, "surface": source,
        }
        legacy = server._methods[method]("legacy", {
            "session_id": "runtime-" + name, "command": "/owner-legacy Exact Case",
            "name": "owner-legacy", "arg": "Exact Case"})
        assert legacy["result"]["output"] == "legacy:Exact Case"
    assert get_hermes_home() == ambient


def test_desktop_dispatch_rejects_stale_owner_instead_of_using_ambient_profile(tmp_path, monkeypatch):
    from tui_gateway import server

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    install_plugin(home)
    response = server._methods["command.dispatch"](
        "stale", {"session_id": "not-a-runtime", "name": "owner-probe", "arg": "on"})
    assert response.get("error", {}).get("code") == 4001, response


def test_cli_process_command_uses_its_own_agent_not_last_active_session(tmp_path, monkeypatch, capsys):
    from cli import HermesCLI

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    install_plugin(home)
    clients = []
    for name in ("a", "b"):
        client = object.__new__(HermesCLI)
        client.config = {}
        client.session_id = "stored-" + name
        client.agent = SimpleNamespace(session_id="conversation-" + name, _current_task_id="task-" + name)
        clients.append(client)
    for client in [*clients, clients[0]]:
        assert client.process_command("/owner-probe Status") is True
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload == {
            "raw": "Status", "registered_home": str(home), "active_home": str(home),
            "session_id": client.agent.session_id, "task_id": client.agent._current_task_id,
            "stored_session_id": client.session_id, "runtime_session_id": None,
            "profile": "default", "hermes_home": str(home), "surface": "cli",
        }
    assert clients[0].process_command("/owner-legacy Exact Case") is True
    assert capsys.readouterr().out.strip() == "legacy:Exact Case"


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex", [False, True])
async def test_gateway_dispatch_resolves_durable_owner_with_real_session_store(tmp_path, monkeypatch, multiplex):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    names = ("a", "b") if multiplex else (None,)
    homes = {name: home / "profiles" / name if name else home for name in names}
    for target in homes.values():
        install_plugin(target)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    runner.session_store = SessionStore(home / "sessions", runner.config)
    runner._draining = False
    sources = [SessionSource(platform=Platform.TELEGRAM, chat_id="chat-" + str(name),
                             user_id="u", chat_type="dm", profile=name) for name in names]

    async def dispatch(source):
        event = MessageEvent(text="/owner_probe On", source=source, message_id="m")
        handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(event, source, "owner_probe")
        assert handled is True
        assert command == "owner_probe"
        assert isinstance(result, str)
        payload = json.loads(result)
        entry = await runner.async_session_store.lookup_by_session_key(runner._session_key_for_source(source))
        assert entry is not None
        target = str(homes[source.profile])
        assert payload == {
            "raw": "On", "registered_home": target, "active_home": target,
            "session_id": entry.session_id, "task_id": entry.session_id,
            "stored_session_id": entry.session_id, "runtime_session_id": None,
            "profile": source.profile or "default", "hermes_home": target, "surface": "gateway",
        }
        return entry.session_id

    try:
        first = await asyncio.gather(*(dispatch(source) for source in sources))
        second = await asyncio.gather(*(dispatch(source) for source in sources))
        assert first == second
        assert len(set(first)) == len(sources)
        assert get_hermes_home() == home
    finally:
        for db in runner.session_store._db_handles.values():
            db.close()
