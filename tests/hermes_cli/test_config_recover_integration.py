"""Integration tests for the config-recovery flow with secret redaction.

These exercise the full _config_recover CLI function and the TUI RPC
recover_status / recover_write round-trip, mocking only the external
LLM API call.  The goal is to verify the wiring: that secrets are
redacted before the prompt, preserved through the LLM response, and
restored before hitting disk.
"""

import os
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml


@pytest.fixture
def isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir and clear the parse-failed flag."""
    env = {"HERMES_HOME": str(tmp_path), "HERMES_CONFIG_PARSE_FAILED": "1"}
    with patch.dict(os.environ, env, clear=False):
        os.environ["HERMES_CONFIG_PARSE_FAILED"] = "1"
        yield tmp_path
    os.environ.pop("HERMES_CONFIG_PARSE_FAILED", None)


@pytest.fixture
def broken_config_with_secrets(isolated_hermes_home):
    """Write a config.yaml.broken that contains inline secrets."""
    home = isolated_hermes_home
    broken_content = (
        "model:\n"
        "  default: openrouter/anthropic/claude-sonnet-4\n"
        "providers:\n"
        "  openrouter:\n"
        "    api_key: sk-or-v1-abc123def456ghi789jkl012\n"
        "    name: OpenRouter\n"
        "mcp_servers:\n"
        "  github:\n"
        "    token: ghp_abc123def456ghi789jkl012mno345\n"
        "    url: https://api.github.com\n"
        "_config_version: 33\n"
    )
    (home / "config.yaml.broken").write_text(broken_content, encoding="utf-8")
    (home / "config.yaml.error").write_text(
        "while parsing a block collection\n"
        "expected <block end>, but found '?'",
        encoding="utf-8",
    )
    # Also write the broken content as the live config.yaml so the
    # parse-failed guard scenario is realistic.
    (home / "config.yaml").write_text(broken_content, encoding="utf-8")
    return broken_content


class TestConfigRecoverCLIIntegration:
    """Full _config_recover flow: redact → mock LLM → restore → validate."""

    def test_cli_recover_redacts_before_llm_restores_after(
        self, isolated_hermes_home, broken_config_with_secrets
    ):
        """Secrets must not appear in the LLM prompt, and must be present in the written config."""
        from hermes_cli.config import _config_recover, get_hermes_home

        home = isolated_hermes_home
        captured_prompt = {}

        def fake_llm_create(model, messages, **kwargs):
            """Mock OpenAI client.chat.completions.create."""
            prompt_text = messages[0]["content"]
            captured_prompt["text"] = prompt_text

            # The prompt should contain placeholders, not real secrets.
            # Return the redacted YAML unchanged (simulating the LLM
            # fixing syntax but preserving placeholders).
            # Extract the YAML between the fences.
            import re
            yaml_match = re.search(r"```yaml\n(.*?)```", prompt_text, re.DOTALL)
            if yaml_match:
                repaired = yaml_match.group(1).strip()
            else:
                repaired = prompt_text

            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = repaired
            return response

        args = types.SimpleNamespace(model="openai/gpt-4o")

        # Patch the OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_llm_create

        with patch("openai.OpenAI", return_value=mock_client):
            # Patch credential resolution to avoid needing real keys
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-dummy"}):
                _config_recover(args)

        # 1. The prompt must NOT contain real secrets
        prompt = captured_prompt["text"]
        assert "sk-or-v1-abc123def456ghi789jkl012" not in prompt, (
            "OpenRouter API key leaked into the LLM prompt"
        )
        assert "ghp_abc123def456ghi789jkl012mno345" not in prompt, (
            "GitHub token leaked into the LLM prompt"
        )
        assert "__REDACTED_" in prompt, (
            "Expected placeholders in the prompt but found none"
        )

        # 2. The written config must have real secrets restored
        written = (home / "config.yaml").read_text(encoding="utf-8")
        assert "sk-or-v1-abc123def456ghi789jkl012" in written, (
            "OpenRouter API key was not restored after LLM repair"
        )
        assert "ghp_abc123def456ghi789jkl012mno345" in written, (
            "GitHub token was not restored after LLM repair"
        )
        assert "__REDACTED_" not in written, (
            "Placeholders survived into the written config"
        )

    def test_cli_recover_prompt_includes_placeholder_instruction(
        self, isolated_hermes_home, broken_config_with_secrets
    ):
        """The prompt must instruct the LLM to preserve placeholders."""
        from hermes_cli.config import _config_recover

        captured = {}

        def fake_create(model, messages, **kwargs):
            captured["prompt"] = messages[0]["content"]
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "model:\n  default: test\n"
            return response

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        args = types.SimpleNamespace(model="openai/gpt-4o")
        with patch("openai.OpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-dummy"}):
                _config_recover(args)

        prompt = captured["prompt"]
        assert "placeholder" in prompt.lower() or "REDACTED" in prompt, (
            "Prompt doesn't mention placeholders or redaction"
        )


class TestConfigRecoverTUIRPCIntegration:
    """TUI RPC round-trip: recover_status (redact) → recover_write (restore)."""

    def test_recover_status_returns_redacted_yaml(
        self, isolated_hermes_home, broken_config_with_secrets
    ):
        """recover_status must not include real secrets in its response."""
        from tui_gateway.server import _methods

        status_fn = _methods.get("config.recover_status")
        assert status_fn is not None, "config.recover_status not registered"

        with patch("tui_gateway.server._hermes_home", isolated_hermes_home):
            result = status_fn(1, {})
        assert result["result"]["needs_recovery"] is True

        raw_config = result["result"]["raw_config"]
        assert "sk-or-v1-abc123def456ghi789jkl012" not in raw_config, (
            "OpenRouter key leaked in recover_status response"
        )
        assert "ghp_abc123def456ghi789jkl012mno345" not in raw_config, (
            "GitHub token leaked in recover_status response"
        )
        assert "__REDACTED_" in raw_config

    def test_recover_write_restores_secrets_before_writing(
        self, isolated_hermes_home, broken_config_with_secrets
    ):
        """recover_write must restore placeholders from the mapping before writing."""
        from tui_gateway.server import _methods, _recover_secret_map

        # Call recover_status first to populate the mapping
        status_fn = _methods.get("config.recover_status")
        with patch("tui_gateway.server._hermes_home", isolated_hermes_home):
            status_result = status_fn(1, {})
        redacted_yaml = status_result["result"]["raw_config"]

        # Verify the mapping was stashed
        assert "mapping" in _recover_secret_map
        assert len(_recover_secret_map["mapping"]) > 0

        # Now simulate the agent returning the redacted YAML unchanged
        # (it "fixed" the syntax but preserved placeholders)
        write_fn = _methods.get("config.recover_write")
        write_result = write_fn(1, {"content": redacted_yaml})

        assert write_result["result"]["written"] is True

        # The file on disk must have real secrets, not placeholders
        written = (isolated_hermes_home / "config.yaml").read_text(encoding="utf-8")
        assert "sk-or-v1-abc123def456ghi789jkl012" in written, (
            "Secret not restored in recover_write output"
        )
        assert "__REDACTED_" not in written, (
            "Placeholders survived into written config"
        )

    def test_recover_write_without_prior_status_does_not_restore(
        self, isolated_hermes_home, broken_config_with_secrets
    ):
        """If recover_write is called without a prior recover_status (no mapping),
        the content is written as-is."""
        from tui_gateway.server import _methods, _recover_secret_map

        # Clear any stale mapping
        _recover_secret_map.clear()

        write_fn = _methods.get("config.recover_write")
        # Write YAML without placeholders
        clean_yaml = (
            "model:\n"
            "  default: openrouter/test\n"
            "_config_version: 33\n"
        )
        result = write_fn(1, {"content": clean_yaml})
        assert result["result"]["written"] is True
        assert "__REDACTED_" not in (isolated_hermes_home / "config.yaml").read_text()
