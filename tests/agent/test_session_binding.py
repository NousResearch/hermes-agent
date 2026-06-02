"""Tests for Session Binding store."""

from agent.managed_agents.session_binding import (
    get_binding,
    put_binding,
    resolve_binding,
    _binding_key,
)
from agent.managed_agents.workspace import DEFAULT_WORKSPACE_ID
from agent.managed_agents.session import DEFAULT_SESSION_ID


def test_binding_key_generation():
    key = _binding_key("discord", "ch-123", "th-456")
    assert key == "discord:ch-123:th-456"


def test_binding_put_get():
    put_binding("discord", "ch-1", "th-1", "ws-discord", "s-discord")
    result = get_binding("discord", "ch-1", "th-1")
    assert result is not None
    assert result == ("ws-discord", "s-discord")


def test_binding_missing_returns_none():
    result = get_binding("feishu", "nonexistent", None)
    assert result is None


def test_resolve_binding_mapped():
    put_binding("feishu", "thread-1", None, "ws-feishu", "s-feishu")
    ws, s = resolve_binding("feishu", "thread-1", None)
    assert ws == "ws-feishu"
    assert s == "s-feishu"


def test_resolve_binding_unmapped_falls_back():
    ws, s = resolve_binding("cli", None, None)
    assert ws == DEFAULT_WORKSPACE_ID
    assert s == DEFAULT_SESSION_ID
