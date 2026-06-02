"""Tests for Workspace, Session, EntryEvent models."""

from agent.managed_agents.workspace import (
    Workspace,
    DEFAULT_WORKSPACE_ID,
    get_workspace,
    put_workspace,
)
from agent.managed_agents.session import (
    Session,
    DEFAULT_SESSION_ID,
    get_session,
    put_session,
    resolve_session,
)
from agent.managed_agents.entry_event import EntryEvent


def test_workspace_default():
    ws = Workspace.make_default()
    assert ws.workspace_id == DEFAULT_WORKSPACE_ID
    assert ws.name == "Local"
    assert ws.entrypoint == "cli"


def test_workspace_round_trip():
    ws = Workspace(workspace_id="ws-1", name="test", entrypoint="discord", external_source_id="guild-123")
    d = ws.to_dict()
    ws2 = Workspace.from_dict(d)
    assert ws2.workspace_id == "ws-1"
    assert ws2.entrypoint == "discord"
    assert ws2.external_source_id == "guild-123"


def test_workspace_put_get():
    ws = Workspace(workspace_id="ws-test", name="Test")
    put_workspace(ws)
    loaded = get_workspace("ws-test")
    assert loaded is not None
    assert loaded.name == "Test"


def test_session_default():
    s = Session.make_default()
    assert s.session_id == DEFAULT_SESSION_ID
    assert s.workspace_id == DEFAULT_WORKSPACE_ID
    assert s.entrypoint == "cli"


def test_session_round_trip():
    s = Session(session_id="s-1", workspace_id="ws-1", name="test-channel", entrypoint="discord", external_channel_id="ch-456")
    d = s.to_dict()
    s2 = Session.from_dict(d)
    assert s2.session_id == "s-1"
    assert s2.external_channel_id == "ch-456"


def test_session_put_get():
    s = Session(session_id="s-test", name="Test Session")
    put_session(s)
    loaded = get_session("s-test")
    assert loaded is not None
    assert loaded.name == "Test Session"


def test_resolve_session_found():
    s = Session(session_id="resolved-1", name="resolved")
    put_session(s)
    result = resolve_session("resolved-1")
    assert result.session_id == "resolved-1"


def test_resolve_session_missing_falls_back():
    result = resolve_session("nonexistent")
    assert result.session_id == DEFAULT_SESSION_ID


def test_resolve_session_none_falls_back():
    result = resolve_session(None)
    assert result.session_id == DEFAULT_SESSION_ID


def test_entry_event_round_trip():
    ev = EntryEvent(
        event_id="ev-1",
        entrypoint="feishu",
        external_user_id="user-1",
        message="hello",
        intent="status_check",
    )
    d = ev.to_dict()
    ev2 = EntryEvent.from_dict(d)
    assert ev2.event_id == "ev-1"
    assert ev2.entrypoint == "feishu"
    assert ev2.message == "hello"
    assert ev2.intent == "status_check"


def test_entry_event_defaults():
    ev = EntryEvent(event_id="ev-defaults", entrypoint="web", message="ping")
    assert ev.workspace_id == DEFAULT_WORKSPACE_ID
    assert ev.session_id == DEFAULT_SESSION_ID
    assert ev.external_channel_id is None


def test_workspace_has_created_at():
    ws = Workspace(workspace_id="ts-1", name="ts")
    assert ws.created_at, "created_at should be auto-populated"


def test_session_has_created_at():
    s = Session(session_id="ts-s", name="ts-session")
    assert s.created_at, "created_at should be auto-populated"
    assert s.updated_at, "updated_at should be auto-populated"
