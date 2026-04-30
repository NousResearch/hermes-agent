"""Tests for the LINE source allowlist (user / group / room filtering)."""
import pytest

from gateway.platforms.line import is_allowed


@pytest.mark.parametrize(
    "cfg, user_id, expected",
    [
        ({"users": ["U1", "U2"], "groups": [], "rooms": []}, "U1", True),
        ({"users": ["U1"], "groups": [], "rooms": []}, "U999", False),
    ],
    ids=["in_list", "not_in_list"],
)
def test_is_allowed_user(cfg, user_id, expected):
    event = {"source": {"type": "user", "userId": user_id}}
    assert is_allowed(event, cfg) is expected


def test_group_in_allowlist():
    cfg = {"users": [], "groups": ["Cabc"], "rooms": []}
    event = {"source": {"type": "group", "groupId": "Cabc", "userId": "U1"}}
    assert is_allowed(event, cfg) is True


def test_room_in_allowlist():
    cfg = {"users": [], "groups": [], "rooms": ["Rxyz"]}
    event = {"source": {"type": "room", "roomId": "Rxyz", "userId": "U1"}}
    assert is_allowed(event, cfg) is True


def test_unknown_source_type_denied():
    cfg = {"users": [], "groups": [], "rooms": []}
    event = {"source": {"type": "weird", "id": "?"}}
    assert is_allowed(event, cfg) is False


def test_empty_allowlists_deny_all():
    cfg = {"users": [], "groups": [], "rooms": []}
    for src in [{"type": "user", "userId": "U1"},
                {"type": "group", "groupId": "C1", "userId": "U1"},
                {"type": "room", "roomId": "R1", "userId": "U1"}]:
        assert is_allowed({"source": src}, cfg) is False


def test_source_type_present_but_id_missing_denied():
    cfg = {"users": ["U1"], "groups": ["C1"], "rooms": ["R1"]}
    assert is_allowed({"source": {"type": "user"}}, cfg) is False
    assert is_allowed({"source": {"type": "group"}}, cfg) is False
    assert is_allowed({"source": {"type": "room"}}, cfg) is False


def test_missing_source_key_denied():
    cfg = {"users": ["U1"], "groups": [], "rooms": []}
    assert is_allowed({}, cfg) is False


def test_allow_all_users_env_resolved_in_adapter(monkeypatch):
    """LINE_ALLOW_ALL_USERS=true resolves to _allow_all_sources=True at __init__
    (debug-only escape hatch — bypass enforced at the dispatch call site)."""
    from gateway.platforms.line import LineAdapter
    from tests.gateway.conftest import make_line_platform_config
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_ALLOW_ALL_USERS", "true")
    adapter = LineAdapter(make_line_platform_config(token="t"))
    assert adapter._allow_all_sources is True


def test_allow_all_users_env_unset_resolves_false(monkeypatch):
    """When LINE_ALLOW_ALL_USERS is unset/false, the bypass is disabled."""
    from gateway.platforms.line import LineAdapter
    from tests.gateway.conftest import make_line_platform_config
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.delenv("LINE_ALLOW_ALL_USERS", raising=False)
    adapter = LineAdapter(make_line_platform_config(token="t"))
    assert adapter._allow_all_sources is False
