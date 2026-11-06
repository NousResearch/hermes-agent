"""Regression test for issue #85622.

When an external memory provider is configured (``memory.provider`` set to a
non-empty value) but ``memory_enabled: false`` is in the config (e.g., left
over from the blank-slate setup wizard), the built-in ``MemoryStore`` was never
created.  The system-prompt renderer checks ``agent._memory_store`` before
injecting ``MEMORY.md`` / ``USER.md``, so the built-in frozen snapshot was
silently dropped from every new-chat system prompt — contradicting the
documented "additive, never replacing" contract.

The fix resolves the effective store flags in ``tools/memory_tool.py``
(``resolve_builtin_memory_store_flags``), forcing ``memory_enabled`` /
``user_profile_enabled`` to ``True`` when a provider is configured, ensuring
the built-in store is always created and injected alongside the external
provider block.
"""

import pytest

from run_agent import AIAgent


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _write_config(hermes_home, memory_section):
    """Write a minimal config.yaml with the given memory section."""
    import hermes_yaml as yaml

    hermes_home.mkdir(parents=True, exist_ok=True)
    config = {"model": {"default": "test-model", "provider": "openrouter"}}
    config["memory"] = memory_section
    with open(hermes_home / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)


def _make_agent(monkeypatch, tmp_path, memory_section, **extra):
    """Create an AIAgent with the given memory config section."""
    hermes_home = tmp_path / "hm"
    _write_config(hermes_home, memory_section)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", dict)
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", _FakeOpenAI)
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *args, **kwargs: 204_800,
    )
    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        **extra,
    )


class TestProviderAdditiveBuiltinMemory:
    """Verify built-in memory is active when a provider is configured."""

    def test_provider_configured_forces_memory_enabled_true(self, monkeypatch, tmp_path):
        """When memory.provider is set and memory_enabled is false, the
        built-in store is still created and _memory_enabled is True."""
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            {"memory_enabled": False, "user_profile_enabled": False, "provider": "fake"},
        )
        assert agent._memory_enabled is True, (
            "memory_enabled must be forced True when a provider is configured "
            "(additive contract, issue #85622)"
        )
        assert agent._memory_store is not None, (
            "built-in MemoryStore must be created when a provider is configured "
            "even if memory_enabled was false in config (issue #85622)"
        )

    def test_provider_configured_forces_user_profile_enabled_true(self, monkeypatch, tmp_path):
        """When memory.provider is set and user_profile_enabled is false,
        user_profile_enabled is forced True."""
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            {"memory_enabled": False, "user_profile_enabled": False, "provider": "fake"},
        )
        assert agent._user_profile_enabled is True, (
            "user_profile_enabled must be forced True when a provider is "
            "configured (additive contract, issue #85622)"
        )

    def test_no_provider_respects_memory_enabled_false(self, monkeypatch, tmp_path):
        """Without a provider, memory_enabled: false is respected — the
        built-in store is NOT created.  This verifies the fix doesn't
        over-trigger when no provider is configured."""
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            {"memory_enabled": False, "user_profile_enabled": False, "provider": ""},
        )
        assert agent._memory_enabled is False, (
            "memory_enabled must stay False when no provider is configured"
        )
        assert agent._memory_store is None, (
            "built-in MemoryStore must NOT be created when memory_enabled is "
            "false and no provider is configured"
        )

    def test_no_provider_respects_memory_enabled_true(self, monkeypatch, tmp_path):
        """Without a provider, memory_enabled: true creates the store as usual."""
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            {"memory_enabled": True, "user_profile_enabled": True, "provider": ""},
        )
        assert agent._memory_enabled is True
        assert agent._memory_store is not None

    @pytest.mark.parametrize(("memory_enabled", "expected_logs"), [(False, 1), (True, 0)])
    def test_provider_override_logging_matches_memory_transition(
        self, monkeypatch, tmp_path, caplog, memory_enabled, expected_logs,
    ):
        """Capture INFO and assert exactly one log on the false-to-true transition."""
        import logging

        caplog.set_level(logging.INFO, logger="tools.memory_tool")
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            {"memory_enabled": memory_enabled, "user_profile_enabled": True, "provider": "fake"},
        )
        override_logs = [
            r for r in caplog.records
            if "enabling built-in MEMORY.md injection" in r.getMessage()
        ]
        assert len(override_logs) == expected_logs
        assert all(record.levelno == logging.INFO for record in override_logs)
        assert agent._memory_enabled is True

    def test_empty_provider_string_does_not_force(self, monkeypatch, tmp_path):
        """An empty/whitespace provider string must not trigger the override."""
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            {"memory_enabled": False, "user_profile_enabled": False, "provider": "  "},
        )
        assert agent._memory_enabled is False, (
            "whitespace-only provider must not force memory_enabled"
        )
        assert agent._memory_store is None

    def test_subagent_checkpoint_gate_beats_provider_override(self, monkeypatch, tmp_path):
        """``memory_agent_context="subagent"`` keeps MEMORY.md/USER.md out of a
        checkpoint-gate fork's prompt even when a provider is configured —
        the upstream gate wins over the additive override."""
        agent = _make_agent(
            monkeypatch,
            tmp_path,
            {"memory_enabled": False, "user_profile_enabled": False, "provider": "fake"},
            memory_agent_context="subagent",
        )
        assert agent._memory_enabled is False
        assert agent._user_profile_enabled is False
        assert agent._memory_store is None
