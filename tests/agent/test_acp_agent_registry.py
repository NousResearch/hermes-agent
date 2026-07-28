"""Unit tests for the ACP agent registry (issue #5257).

Only ``copilot`` ships built into core — every other ACP agent (Claude
Code, Codex CLI, community agents) registers itself from a
``plugins/model-providers/<name>-acp/`` plugin via
:func:`register_acp_agent`. These tests cover the mechanism generically
(registration, override, resolution order, command fallbacks, env
stripping) using synthetic entries, plus copilot's pre-existing legacy
contract, rather than asserting specific third-party agents are built in.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agent.acp_agent_registry import (
    ACP_AGENT_REGISTRY,
    ACPAgentEntry,
    agent_display_name,
    agent_install_hint,
    is_acp_agent_available,
    known_agents,
    normalize_agent_name,
    register_acp_agent,
    resolve_agent_launch,
)


@pytest.fixture
def synthetic_agent():
    """Register a throwaway agent for the duration of a test, then remove it.

    Standing in for a real plugin's ``register_acp_agent()`` call — the
    registry doesn't care whether the entry came from a plugin import or a
    test, so this exercises the exact same code path.
    """
    name = "synthetic-test-agent"
    entry = ACPAgentEntry(
        command="synthetic-acp",
        command_fallbacks=("synthetic-acp-legacy",),
        args=("--acp",),
        env_unset=("SYNTHETIC_SESSION_MARKER",),
        display_name="Synthetic Test Agent",
        install_hint="npm install -g synthetic-acp",
    )
    register_acp_agent(name, entry)
    try:
        yield name, entry
    finally:
        ACP_AGENT_REGISTRY.pop(name, None)


def test_known_agents_include_builtin_copilot():
    assert "copilot" in known_agents()


def test_builtin_copilot_resolution():
    with patch("agent.acp_agent_registry.shutil.which", return_value=None):
        assert resolve_agent_launch("copilot") == ("copilot", ["--acp", "--stdio"])


def test_register_acp_agent_round_trip(synthetic_agent):
    name, entry = synthetic_agent
    assert name in known_agents()
    assert is_acp_agent_available(name)
    with patch("agent.acp_agent_registry.shutil.which", return_value=None):
        assert resolve_agent_launch(name) == (entry.command, list(entry.args))


def test_register_acp_agent_overrides_existing_entry():
    """Last-writer-wins — a plugin re-registering a name replaces the entry.

    Mirrors the ``providers.register_provider`` override contract: a user
    plugin under ``$HERMES_HOME/plugins/model-providers/`` can replace a
    bundled agent entry without editing this file.
    """
    name = "override-test-agent"
    register_acp_agent(name, ACPAgentEntry(command="first-command"))
    try:
        register_acp_agent(name, ACPAgentEntry(command="second-command"))
        with patch("agent.acp_agent_registry.shutil.which", return_value=None):
            assert resolve_agent_launch(name) == ("second-command", [])
    finally:
        ACP_AGENT_REGISTRY.pop(name, None)


def test_command_falls_back_to_alternate_bin_when_only_it_is_installed(synthetic_agent):
    name, entry = synthetic_agent
    fallback = entry.command_fallbacks[0]

    def _only_fallback(candidate):
        return f"/usr/bin/{fallback}" if candidate == fallback else None

    with patch("agent.acp_agent_registry.shutil.which", side_effect=_only_fallback):
        assert resolve_agent_launch(name) == (fallback, list(entry.args))


def test_agent_env_unset_declares_configured_markers(synthetic_agent):
    name, entry = synthetic_agent
    from agent.acp_agent_registry import agent_env_unset

    assert agent_env_unset(name) == entry.env_unset
    assert agent_env_unset("copilot") == ()
    assert agent_env_unset("unknown-agent") == ()


def test_normalization_is_case_and_whitespace_insensitive(synthetic_agent):
    name, _entry = synthetic_agent
    assert normalize_agent_name(f"  {name.upper()} ") == name
    with patch("agent.acp_agent_registry.shutil.which", return_value=None):
        assert resolve_agent_launch(name.upper()) == resolve_agent_launch(name)


def test_generic_env_override_is_shlex_split_full_command():
    with patch.dict(os.environ, {"HERMES_ACP_CLINE_COMMAND": "npx cline-acp --stdio"}, clear=False):
        assert resolve_agent_launch("cline") == ("npx", ["cline-acp", "--stdio"])
        assert is_acp_agent_available("cline")


def test_generic_env_override_beats_registry():
    with patch.dict(os.environ, {"HERMES_ACP_COPILOT_COMMAND": "/opt/custom/copilot --flag"}, clear=False):
        assert resolve_agent_launch("copilot") == ("/opt/custom/copilot", ["--flag"])


def test_legacy_copilot_env_vars_are_command_path_only():
    with patch.dict(
        os.environ,
        {"HERMES_COPILOT_ACP_COMMAND": "/usr/local/bin/copilot"},
        clear=False,
    ):
        command, args = resolve_agent_launch("copilot")
        assert command == "/usr/local/bin/copilot"
        # default args preserved — the legacy var never carried args
        assert args == ["--acp", "--stdio"]

    with patch.dict(os.environ, {"COPILOT_CLI_PATH": "/opt/bin/copilot"}, clear=False):
        command, _ = resolve_agent_launch("copilot")
        assert command == "/opt/bin/copilot"


def test_legacy_copilot_args_env_var_replaces_args():
    with patch.dict(os.environ, {"HERMES_COPILOT_ACP_ARGS": "--acp"}, clear=False):
        assert resolve_agent_launch("copilot") == ("copilot", ["--acp"])


def test_generic_args_env_var_replaces_args(synthetic_agent):
    name, _entry = synthetic_agent
    env_key = f"HERMES_ACP_{name.upper().replace('-', '_')}_ARGS"
    with patch.dict(os.environ, {env_key: "--acp --sandbox"}, clear=False):
        assert resolve_agent_launch(name) == ("synthetic-acp", ["--acp", "--sandbox"])


def test_unknown_agent_without_override_raises_with_hint():
    with pytest.raises(ValueError) as excinfo:
        resolve_agent_launch("definitely-not-an-agent")
    message = str(excinfo.value)
    assert "HERMES_ACP_DEFINITELY_NOT_AN_AGENT_COMMAND" in message
    assert not is_acp_agent_available("definitely-not-an-agent")


def test_display_name_and_install_hint(synthetic_agent):
    name, entry = synthetic_agent
    assert agent_display_name(name) == entry.display_name
    assert entry.install_hint in agent_install_hint(name)
    # Unknown agents get a generic-but-actionable hint
    assert "HERMES_ACP_MYSTERY_COMMAND" in agent_install_hint("mystery")


def test_registry_entries_have_display_names_and_hints():
    for name, entry in ACP_AGENT_REGISTRY.items():
        assert entry.command, name
        assert entry.display_name, name
        assert entry.install_hint, name


def test_discovery_picks_up_plugin_registered_agents(monkeypatch):
    """``_discover_agents`` imports provider plugins once, lazily.

    A plugin's ``__init__.py`` calls :func:`register_acp_agent` as a side
    effect of being imported by :func:`providers.list_providers`; this
    simulates that without needing a real plugin directory on disk.
    """
    import agent.acp_agent_registry as registry_mod

    monkeypatch.setattr(registry_mod, "_agents_discovered", False)
    name = "discovered-test-agent"

    def _fake_list_providers():
        register_acp_agent(name, ACPAgentEntry(command="discovered-acp"))
        return []

    with patch("providers.list_providers", side_effect=_fake_list_providers):
        try:
            assert name in known_agents()
        finally:
            ACP_AGENT_REGISTRY.pop(name, None)
