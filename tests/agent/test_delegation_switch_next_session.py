"""``/delegation <backend>`` saves for the next session; ``--now`` rebuilds this session's tools.

A session routes on the backend its ``delegate_task`` schema was built from, so routing and schema switch
together. Real AIAgent turns against a recording fake provider, a real delegation backend plugin discovered from
a temp HERMES_HOME and the real session DB: each test reads the schema Hermes actually sent and the backend a
task with no choice routes on.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home
from tests.tools.test_delegation_backends import OPTIONS, PROBE_PLUGIN, REPO_PLUGINS


@pytest.fixture
def options_backend(monkeypatch):
    """``delegation.backend: probe-options`` (adds a per-task ``option``) on a fake provider."""
    import hermes_yaml as yaml
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    home = Path(os.environ["HERMES_HOME"])
    plugin_dir = home / "plugins" / "probe-options"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "probe-options", "version": "1.0"}))
    (plugin_dir / "__init__.py").write_text(PROBE_PLUGIN)
    bundled = home.parent / "bundled"
    bundled.mkdir()
    (bundled / "delegation").symlink_to(REPO_PLUGINS / "delegation", target_is_directory=True)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    srv = FakeLLMServer([]).__enter__()
    extra = yaml.safe_dump({"plugins": {"enabled": ["probe-options"]}, "delegation": {
        "backend": "probe-options", "backends": {"probe-options": {"options": OPTIONS}}}})
    write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + extra)
    manager = PluginManager()
    manager.discover_and_load()
    yield srv, home
    manager.unload()
    srv.__exit__(None, None, None)


def _agent(srv, home, session_id):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=home / "state.db")
    db.ensure_session(session_id, source="cli")  # opened before its first turn, as the gateway does
    return AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                   session_db=db, session_id=session_id, quiet_mode=True, platform="cli")


def _sent_task_fields(srv) -> set[str]:
    """The per-task fields of ``delegate_task`` in the last main request."""
    tool = next(t for t in srv.main_requests()[-1]["tools"] if t["function"]["name"] == "delegate_task")
    return set(tool["function"]["parameters"]["properties"]["tasks"]["items"]["properties"])


def _routed_on(agent) -> str:
    """The backend a task naming no option routes on for ``agent``'s session."""
    from tools.delegate_tool_config import _load_config
    from tools.delegate_tool_routes import resolve_task_routes
    routes, error = resolve_task_routes([{"goal": "a plain task"}], _load_config(), agent)
    assert error is None
    return routes[0].backend


def _turn(agent, text, history):
    return agent.run_conversation(text, conversation_history=history, task_id="t")["messages"]


def test_a_now_switch_rebuilds_this_sessions_delegate_task_and_routing_for_the_new_backend(options_backend):
    from agent.delegation_backends import set_global_backend

    srv, home = options_backend
    agent = _agent(srv, home, "chat-now")
    history = _turn(agent, "hello", [])
    assert "option" in _sent_task_fields(srv) and _routed_on(agent) == "probe-options"
    agent.close()

    # The gateway's next turn is a fresh agent: it restores the session's pinned tools, then takes the request.
    set_global_backend("default", session_key="chat-now", apply_now=True)
    fresh = _agent(srv, home, "chat-now")
    history = _turn(fresh, "and now", history)
    assert "option" not in _sent_task_fields(srv) and _routed_on(fresh) == "default"
    fresh.close()

    # The rebuilt tools and their backend are the session's pin now, whatever config.yaml says by then.
    set_global_backend("probe-options")
    resumed = _agent(srv, home, "chat-now")
    _turn(resumed, "still here", history)
    assert "option" not in _sent_task_fields(srv) and _routed_on(resumed) == "default"
    resumed.close()


def test_a_plain_switch_changes_nothing_in_this_session_and_new_sessions_take_the_saved_backend(options_backend):
    import hermes_yaml as yaml
    from agent.delegation_backends import set_global_backend

    srv, home = options_backend
    agent = _agent(srv, home, "chat-plain")
    history = _turn(agent, "hello", [])
    before = srv.main_requests()[-1]["tools"]

    set_global_backend("default", session_key="chat-plain")
    assert yaml.safe_load((home / "config.yaml").read_text())["delegation"]["backend"] == "default"
    history = _turn(agent, "and now", history)
    assert srv.main_requests()[-1]["tools"] == before  # the cached tool list is untouched
    assert _routed_on(agent) == "probe-options"  # and so is the backend it routes on
    agent.close()
    resumed = _agent(srv, home, "chat-plain")  # a fresh agent on the same session restores both
    _turn(resumed, "still here", history)
    assert srv.main_requests()[-1]["tools"] == before and _routed_on(resumed) == "probe-options"
    resumed.close()

    new = _agent(srv, home, "chat-plain-new")  # /new
    _turn(new, "hello again", [])
    assert "option" not in _sent_task_fields(srv) and _routed_on(new) == "default"
    new.close()
