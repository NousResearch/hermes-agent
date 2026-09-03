"""Regression tests for CLI prefill config key compatibility."""

from __future__ import annotations

import cli


def test_resolve_prefill_messages_file_uses_top_level(monkeypatch):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)

    assert cli._resolve_prefill_messages_file(
        {
            "prefill_messages_file": "top.json",
            "agent": {"prefill_messages_file": "legacy.json"},
        }
    ) == "top.json"


def test_resolve_prefill_messages_file_accepts_legacy_agent_key(monkeypatch):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)

    assert cli._resolve_prefill_messages_file(
        {"agent": {"prefill_messages_file": "legacy.json"}}
    ) == "legacy.json"


def test_resolve_prefill_messages_file_prefers_env(monkeypatch):
    monkeypatch.setenv("HERMES_PREFILL_MESSAGES_FILE", "env.json")

    assert cli._resolve_prefill_messages_file(
        {
            "prefill_messages_file": "top.json",
            "agent": {"prefill_messages_file": "legacy.json"},
        }
    ) == "env.json"


def test_cli_mixin_init_agent_loads_prefill(monkeypatch, tmp_path):
    import json
    from unittest.mock import MagicMock
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    prefill_data = [{"role": "system", "content": "test"}]
    prefill_file = tmp_path / "prefill.json"
    prefill_file.write_text(json.dumps(prefill_data), encoding="utf-8")

    class DummyCLI(MagicMock, CLIAgentSetupMixin):
        pass

    dummy = DummyCLI()
    dummy.agent = None
    dummy.system_prompt = None
    dummy.api_key = "test"
    dummy.base_url = "http://localhost"
    dummy.provider = "custom"
    dummy.api_mode = "openai"
    dummy.acp_command = None
    dummy.acp_args = None
    dummy.model = "mock"
    dummy.max_tokens = 100
    dummy.max_turns = 10
    dummy.enabled_toolsets = []
    dummy.disabled_toolsets = []
    dummy.verbose = False
    dummy.reasoning_config = None
    dummy.service_tier = None
    dummy._providers_only = None
    dummy._providers_ignore = None
    dummy._providers_order = None
    dummy._provider_sort = None
    dummy._provider_require_params = None
    dummy._provider_data_collection = None
    dummy._openrouter_min_coding_score = None
    dummy.session_id = "test_sess"
    dummy._session_db = None
    dummy._resumed = False
    dummy.conversation_history = []
    dummy.prefill_messages = None
    dummy._ensure_runtime_credentials = MagicMock(return_value=True)

    captured_kwargs = {}
    def mock_agent(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"prefill_messages_file": str(prefill_file)})
    monkeypatch.setattr("cli.AIAgent", mock_agent)

    success = dummy._init_agent()
    assert success is True
    assert dummy.prefill_messages == prefill_data
    assert captured_kwargs.get("prefill_messages") == prefill_data


