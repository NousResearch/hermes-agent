"""Tests for gateway tool permission tiers (issue #527)."""

from __future__ import annotations

from gateway.tool_tiers import (
    ADMIN,
    GUEST,
    OWNER,
    USER,
    can_use,
    normalize_role,
    refine_allowed_tools,
    resolve_role,
)


def test_unconfigured_install_is_unrestricted() -> None:
    """No role configured means every tool is allowed: existing installs unaffected."""
    assert can_use(None, "terminal") is True
    assert can_use(None, "write_file") is True
    assert can_use(None, "anything_new") is True


def test_owner_can_use_everything_including_new_tools() -> None:
    assert can_use(OWNER, "terminal") is True
    assert can_use(OWNER, "write_file") is True
    assert can_use(OWNER, "a_tool_added_next_year") is True


def test_admin_runs_agent_tools_but_not_raw_shell() -> None:
    assert can_use(ADMIN, "cron") is True
    assert can_use(ADMIN, "sessions") is True
    assert can_use(ADMIN, "terminal") is False
    assert can_use(ADMIN, "write_file") is False


def test_user_gets_read_tools_but_not_writes() -> None:
    assert can_use(USER, "web_search") is True
    assert can_use(USER, "read_file") is True
    assert can_use(USER, "write_file") is False
    assert can_use(USER, "terminal") is False


def test_guest_keeps_only_the_safe_floor() -> None:
    assert can_use(GUEST, "help") is True
    assert can_use(GUEST, "whoami") is True
    assert can_use(GUEST, "web_search") is False
    assert can_use(GUEST, "read_file") is False


def test_unknown_role_fails_closed() -> None:
    assert can_use("wizard", "help") is False
    assert can_use("wizard", "terminal") is False


def test_normalize_role_accepts_aliases() -> None:
    assert normalize_role("Administrator") == ADMIN
    assert normalize_role(" member ") == USER
    assert normalize_role("visitor") == GUEST
    assert normalize_role(None) is None
    assert normalize_role("") is None


def test_resolve_role_uses_user_entry_then_default() -> None:
    roles = {"discord:123": GUEST, "telegram:456": ADMIN}
    assert resolve_role("discord:123", roles, default=USER) == GUEST
    assert resolve_role("telegram:456", roles, default=USER) == ADMIN
    assert resolve_role("discord:999", roles, default=USER) == USER
    assert resolve_role("discord:999", roles) is None


def test_refine_tools_preserves_order_for_owner() -> None:
    tools = ["terminal", "write_file", "web_search", "read_file"]
    assert refine_allowed_tools(OWNER, tools) == tools
    assert refine_allowed_tools(None, tools) == tools


def test_refine_tools_filters_for_guest() -> None:
    tools = ["help", "web_search", "whoami", "terminal"]
    assert refine_allowed_tools(GUEST, tools) == ["help", "whoami"]
