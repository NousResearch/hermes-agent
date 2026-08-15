import json

import pytest

from acp_adapter import session as acp_session
from hermes_cli import config as config_module
from tools import terminal_tool as tt


class _Agent:
    model = "test-model"
    provider = "test-provider"
    base_url = None
    api_mode = None


class _MemoryDb:
    def __init__(self):
        self.sessions = {}
        self.messages = {}

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def create_session(self, *, session_id, source, model, model_config):
        self.sessions[session_id] = {
            "id": session_id,
            "source": source,
            "model": model,
            "model_config": json.dumps(model_config),
        }

    def update_session_meta(self, session_id, model_config, model):
        self.sessions[session_id]["model_config"] = model_config
        self.sessions[session_id]["model"] = model

    def replace_messages(self, session_id, history, **_kwargs):
        self.messages[session_id] = list(history)

    def get_messages_as_conversation(self, session_id, **_kwargs):
        return list(self.messages.get(session_id, []))

    def delete_session(self, session_id):
        return self.sessions.pop(session_id, None) is not None


def _route():
    return {
        "backend": "ssh",
        "host": "workspace.example",
        "user": "developer",
        "port": 2222,
        "key": "~/.ssh/workspace",
        "sync": False,
    }


def _isolate_task_state(monkeypatch):
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(tt, "_session_cwd", {})


def test_acp_workspace_route_defaults_to_no_sync():
    route = acp_session._normalize_acp_workspace_route({
        "backend": "ssh",
        "host": "workspace.example",
        "user": "developer",
    })
    assert route is not None
    assert route["sync"] is False


def test_invalid_configured_workspace_route_fails_closed(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "read_raw_config",
        lambda: {
            "acp": {
                "workspace": {
                    "backend": "ssh",
                    "host": "workspace.example",
                }
            }
        },
    )

    with pytest.raises(ValueError, match="Invalid acp.workspace"):
        acp_session._configured_acp_workspace_route()


def test_invalid_persisted_workspace_route_fails_closed():
    db = _MemoryDb()
    db.create_session(
        session_id="invalid-route",
        source="acp",
        model="test-model",
        model_config={
            "cwd": "/workspace/project",
            "workspace_route": {
                "backend": "ssh",
                "host": "workspace.example",
            },
        },
    )
    manager = acp_session.SessionManager(agent_factory=_Agent, db=db)

    with pytest.raises(ValueError, match="Invalid persisted ACP workspace route"):
        manager.get_session("invalid-route")


def test_acp_session_persists_configured_workspace_route_before_agent_creation(
    monkeypatch,
):
    _isolate_task_state(monkeypatch)
    db = _MemoryDb()
    route = _route()
    monkeypatch.setattr(acp_session, "_configured_acp_workspace_route", lambda: route)
    observed = {}

    def agent_factory():
        observed.update(tt.resolve_task_overrides(next(iter(tt._task_env_overrides))))
        return _Agent()

    manager = acp_session.SessionManager(agent_factory=agent_factory, db=db)
    state = manager.create_session(cwd="/workspace/project")

    assert state.workspace_route == route
    assert observed == {
        "cwd": "/workspace/project",
        "env_type": "ssh",
        "ssh_host": "workspace.example",
        "ssh_user": "developer",
        "ssh_port": 2222,
        "ssh_key": "~/.ssh/workspace",
        "ssh_sync": False,
    }
    persisted = json.loads(db.sessions[state.session_id]["model_config"])
    assert persisted["workspace_route"] == route


def test_acp_session_restores_persisted_workspace_route_before_agent_creation(
    monkeypatch,
):
    _isolate_task_state(monkeypatch)
    db = _MemoryDb()
    session_id = "persisted-acp-session"
    route = _route()
    db.sessions[session_id] = {
        "id": session_id,
        "source": "acp",
        "model": "test-model",
        "model_config": json.dumps({
            "cwd": "/workspace/project",
            "workspace_route": route,
        }),
    }
    observed = {}

    def agent_factory():
        observed.update(tt.resolve_task_overrides(session_id))
        return _Agent()

    manager = acp_session.SessionManager(agent_factory=agent_factory, db=db)
    state = manager.get_session(session_id)

    assert state is not None
    assert state.workspace_route == route
    assert observed["env_type"] == "ssh"
    assert observed["ssh_sync"] is False
