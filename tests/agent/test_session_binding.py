"""Tests for Session Binding store."""

import pytest

from agent.managed_agents.session_binding import (
    get_binding,
    put_binding,
    resolve_binding,
    resolve_binding_with_source,
    get_binding_value,
    SessionBindingValue,
    _binding_key,
    _binding_values,
    _bindings_lock,
)
from agent.managed_agents.workspace import DEFAULT_WORKSPACE_ID
from agent.managed_agents.session import DEFAULT_SESSION_ID


@pytest.fixture(autouse=True)
def _clear_bindings():
    """Clear binding stores between tests."""
    with _bindings_lock:
        _binding_values.clear()
    yield
    with _bindings_lock:
        _binding_values.clear()


# ---------------------------------------------------------------------------
# Legacy tuple API
# ---------------------------------------------------------------------------

def test_binding_key_generation():
    key = _binding_key("discord", "ch-123", "th-456")
    assert key == "discord:ch-123:th-456"


def test_binding_key_with_none_fields():
    key = _binding_key("feishu", "oc_abc", None)
    assert key == "feishu:oc_abc:"


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


# ---------------------------------------------------------------------------
# SessionBindingValue (v2.10)
# ---------------------------------------------------------------------------

def test_binding_value_new_format():
    """put_binding with source writes SessionBindingValue correctly."""
    put_binding("feishu", "oc_card", None, "ws-1", "ses-1", source="card")
    val = get_binding_value("feishu", "oc_card", None)
    assert val is not None
    assert val.workspace_id == "ws-1"
    assert val.session_id == "ses-1"
    assert val.source == "card"
    assert val.created_at != ""


def test_binding_value_default_source():
    """put_binding without source defaults to 'default'."""
    put_binding("feishu", "oc_default", None, "ws-2", "ses-2")
    val = get_binding_value("feishu", "oc_default", None)
    assert val is not None
    assert val.source == "default"


def test_binding_value_thread_source():
    """put_binding with source='thread' records thread-derived binding."""
    put_binding("feishu", "oc_grp", "om_t1", "ws-3", "ses-3", source="thread")
    val = get_binding_value("feishu", "oc_grp", "om_t1")
    assert val is not None
    assert val.source == "thread"


def test_binding_value_alias_source():
    """put_binding with source='alias' records alias-derived binding."""
    put_binding("feishu", "oc_alias", None, "ws-4", "ses-4", source="alias")
    val = get_binding_value("feishu", "oc_alias", None)
    assert val is not None
    assert val.source == "alias"


def test_resolve_binding_with_source_found():
    """resolve_binding_with_source returns SessionBindingValue when found."""
    put_binding("feishu", "oc_resolve", None, "ws-5", "ses-5", source="card")
    val = resolve_binding_with_source("feishu", "oc_resolve", None)
    assert val is not None
    assert val.source == "card"


def test_resolve_binding_with_source_not_found():
    """resolve_binding_with_source returns None when no binding exists."""
    val = resolve_binding_with_source("feishu", "nonexistent", None)
    assert val is None


def test_binding_value_from_dict_list():
    """SessionBindingValue.from_dict reads old 2-tuple format."""
    val = SessionBindingValue.from_dict(["ws-old", "ses-old"])
    assert val.workspace_id == "ws-old"
    assert val.session_id == "ses-old"
    assert val.source == "default"
    assert val.created_at == ""


def test_binding_value_from_dict_full():
    """SessionBindingValue.from_dict reads full dict format."""
    val = SessionBindingValue.from_dict({
        "workspace_id": "ws-full",
        "session_id": "ses-full",
        "source": "card",
        "created_at": "2025-01-01T00:00:00Z",
    })
    assert val.workspace_id == "ws-full"
    assert val.session_id == "ses-full"
    assert val.source == "card"
    assert val.created_at == "2025-01-01T00:00:00Z"


def test_binding_value_to_dict_roundtrip():
    """SessionBindingValue to_dict/from_dict roundtrip is stable."""
    original = SessionBindingValue(
        workspace_id="ws-rt",
        session_id="ses-rt",
        source="thread",
        created_at="2025-06-01T12:00:00Z",
    )
    d = original.to_dict()
    restored = SessionBindingValue.from_dict(d)
    assert restored.workspace_id == "ws-rt"
    assert restored.session_id == "ses-rt"
    assert restored.source == "thread"
    assert restored.created_at == "2025-06-01T12:00:00Z"


def test_card_priority_over_thread():
    """Card binding takes priority over thread binding for same key."""
    # Thread binding
    put_binding("feishu", "oc_priority", "om_t1", "ws-thread", "ses-thread", source="thread")
    # Card binding overrides
    put_binding("feishu", "oc_priority", "om_t1", "ws-card", "ses-card", source="card")
    val = get_binding_value("feishu", "oc_priority", "om_t1")
    assert val is not None
    assert val.session_id == "ses-card"
    assert val.source == "card"
