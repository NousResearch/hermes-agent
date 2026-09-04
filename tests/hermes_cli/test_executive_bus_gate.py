"""Tests for Executive Bus availability gating."""

from __future__ import annotations


def test_executive_bus_gate_requires_explicit_platform_enable(monkeypatch):
    from tools.registry import invalidate_check_fn_cache
    from hermes_cli import config as config_mod
    from hermes_cli.executive_bus_gate import executive_bus_enabled_for_current_context

    monkeypatch.setenv("HERMES_PLATFORM", "discord")
    monkeypatch.setattr(config_mod, "load_config", lambda: {"platform_toolsets": {"discord": ["terminal"]}})
    invalidate_check_fn_cache()

    assert executive_bus_enabled_for_current_context() is False

    monkeypatch.setattr(config_mod, "load_config", lambda: {"platform_toolsets": {"discord": ["terminal", "executive_bus"]}})
    invalidate_check_fn_cache()

    assert executive_bus_enabled_for_current_context() is True


def test_executive_bus_gate_is_platform_scoped(monkeypatch):
    from tools.registry import invalidate_check_fn_cache
    from hermes_cli import config as config_mod
    from hermes_cli.executive_bus_gate import executive_bus_enabled_for_current_context

    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"platform_toolsets": {"discord": ["executive_bus"], "telegram": ["terminal"]}},
    )
    invalidate_check_fn_cache()

    assert executive_bus_enabled_for_current_context() is False


def test_executive_bus_tools_hidden_until_gate_passes(monkeypatch):
    from tools.registry import invalidate_check_fn_cache, registry
    from hermes_cli import config as config_mod
    import tools.capability_registry_tool  # noqa: F401 - registers tool
    import tools.profile_delegation_tool  # noqa: F401 - registers tool

    requested = {"find_capability", "delegate_to_profile"}
    monkeypatch.setenv("HERMES_PLATFORM", "cli")
    monkeypatch.setattr(config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["terminal"]}})
    invalidate_check_fn_cache()

    hidden = registry.get_definitions(requested, quiet=True)
    assert hidden == []

    monkeypatch.setattr(config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["terminal", "executive_bus"]}})
    invalidate_check_fn_cache()

    visible = registry.get_definitions(requested, quiet=True)
    assert {entry["function"]["name"] for entry in visible} == requested
