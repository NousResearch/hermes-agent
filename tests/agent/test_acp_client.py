"""Tests for the ACPClient (acpx-backed)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from agent.acp_client import (
    ACPClient,
    _format_messages_as_prompt,
    _render_content,
    extract_agent_from_url,
)


# ---------------------------------------------------------------------------
# extract_agent_from_url
# ---------------------------------------------------------------------------

class TestExtractAgentFromUrl:
    def test_claude(self):
        assert extract_agent_from_url("acp://claude") == "claude"

    def test_codex(self):
        assert extract_agent_from_url("acp://codex") == "codex"

    def test_copilot(self):
        assert extract_agent_from_url("acp://copilot") == "copilot"

    def test_non_acp_url(self):
        assert extract_agent_from_url("https://api.openai.com") is None

    def test_empty(self):
        assert extract_agent_from_url("") is None

    def test_bare_prefix(self):
        assert extract_agent_from_url("acp://") is None


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

class TestFormatMessages:
    def test_basic_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = _format_messages_as_prompt(msgs, agent_name="claude")
        assert "User:\nHello" in result

    def test_system_and_user(self):
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = _format_messages_as_prompt(msgs)
        assert "System:\nBe helpful" in result
        assert "User:\nHi" in result

    def test_model_hint(self):
        result = _format_messages_as_prompt([], model="gpt-5")
        assert "gpt-5" in result

    def test_empty_messages(self):
        result = _format_messages_as_prompt([])
        assert result == ""


class TestRenderContent:
    def test_string(self):
        assert _render_content("hello") == "hello"

    def test_none(self):
        assert _render_content(None) == ""

    def test_dict_with_text(self):
        assert _render_content({"text": "hi"}) == "hi"

    def test_list_of_parts(self):
        parts = [{"text": "a"}, {"text": "b"}]
        assert _render_content(parts) == "a\nb"


# ---------------------------------------------------------------------------
# ACPClient init
# ---------------------------------------------------------------------------

class TestACPClientInit:
    def test_known_agent(self):
        client = ACPClient(agent_name="gemini")
        assert client.agent_name == "gemini"
        assert client.base_url == "acp://gemini"

    def test_agent_from_base_url(self):
        client = ACPClient(base_url="acp://codex")
        assert client.agent_name == "codex"

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown ACP agent"):
            ACPClient(agent_name="nonexistent-xyz")

    def test_custom_command(self):
        client = ACPClient(agent_name="custom", acp_command="/bin/my-agent --acp")
        assert client.agent_name == "custom"

    def test_close_idempotent(self):
        client = ACPClient(agent_name="gemini")
        client.close()
        client.close()  # should not raise

    def test_all_agents_resolve_to_acpx(self):
        """All built-in agents route through acpx."""
        from agent.acp_agent_registry import list_agents
        for agent in list_agents():
            client = ACPClient(agent_name=agent)
            assert "acpx" in " ".join(client._acp_argv), f"{agent} doesn't use acpx"


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

class TestCopilotBackwardsCompat:
    def test_import(self):
        from agent.copilot_acp_client import CopilotACPClient, ACP_MARKER_BASE_URL
        assert ACP_MARKER_BASE_URL == "acp://copilot"

    def test_creates_copilot_agent(self):
        from agent.copilot_acp_client import CopilotACPClient
        client = CopilotACPClient()
        assert client.agent_name == "copilot"
        assert client.base_url == "acp://copilot"


# ---------------------------------------------------------------------------
# acpx subprocess flow (mocked)
# ---------------------------------------------------------------------------

class TestAcpxFlow:
    def test_successful_response(self):
        """acpx returns NDJSON session update events."""
        ndjson_output = "\n".join([
            json.dumps({"method": "session/update", "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "Hello "}}}}),
            json.dumps({"method": "session/update", "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "world!"}}}}),
        ]) + "\n"

        proc = mock.MagicMock()
        proc.stdout = iter(ndjson_output.splitlines(keepends=True))
        proc.stderr = iter([])
        proc.stdin = mock.MagicMock()
        proc.wait.return_value = 0
        proc.returncode = 0

        with mock.patch("subprocess.Popen", return_value=proc):
            client = ACPClient(agent_name="gemini")
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gemini-acp",
            )
            assert result.choices[0].message.content == "Hello world!"
            assert result.model == "gemini-acp"

    def test_agent_not_found_error(self):
        """FileNotFoundError from Popen gives clear error."""
        with mock.patch("subprocess.Popen", side_effect=FileNotFoundError("not found")):
            client = ACPClient(agent_name="gemini")
            with pytest.raises(RuntimeError, match="Could not start ACP agent"):
                client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hi"}],
                )

    def test_nonzero_exit_with_stderr(self):
        """Process failure surfaces stderr."""
        proc = mock.MagicMock()
        proc.stdout = iter([])
        proc.stderr = iter(["Error: auth failed\n"])
        proc.stdin = mock.MagicMock()
        proc.wait.return_value = 1
        proc.returncode = 1

        with mock.patch("subprocess.Popen", return_value=proc):
            client = ACPClient(agent_name="gemini")
            with pytest.raises(RuntimeError, match="auth failed"):
                client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hi"}],
                )

    def test_auth_error_detected(self):
        """401 in stderr is flagged as auth failure."""
        proc = mock.MagicMock()
        proc.stdout = iter([])
        proc.stderr = iter(["HTTP 401: authentication_error\n"])
        proc.stdin = mock.MagicMock()
        proc.wait.return_value = 1
        proc.returncode = 1

        with mock.patch("subprocess.Popen", return_value=proc):
            client = ACPClient(agent_name="claude")
            with pytest.raises(RuntimeError, match="authentication failed"):
                client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hi"}],
                )

    def test_command_includes_exec_and_approve(self):
        """Verify acpx is called with exec and --approve-all."""
        proc = mock.MagicMock()
        proc.stdout = iter([])
        proc.stderr = iter([])
        proc.stdin = mock.MagicMock()
        proc.wait.return_value = 0
        proc.returncode = 0

        with mock.patch("subprocess.Popen", return_value=proc) as mock_popen:
            client = ACPClient(agent_name="gemini")
            client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
            )
            call_args = mock_popen.call_args[0][0]
            assert "exec" in call_args
            assert "--approve-all" in call_args


# ---------------------------------------------------------------------------
# Provider overlay wiring
# ---------------------------------------------------------------------------

class TestProviderOverlays:
    def test_acp_providers_registered(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        expected_agents = [
            "claude-acp", "codex-acp", "gemini-acp", "cursor-acp",
            "kiro-acp", "kilocode-acp", "opencode-acp", "kimi-acp",
            "qwen-acp", "cline-acp", "amp-acp", "droid-acp", "iflow-acp",
            "copilot-acp",
        ]
        for provider_id in expected_agents:
            assert provider_id in HERMES_OVERLAYS, f"Missing overlay: {provider_id}"
            overlay = HERMES_OVERLAYS[provider_id]
            assert overlay.auth_type == "external_process"
            agent_name = provider_id.removesuffix("-acp")
            assert overlay.base_url_override == f"acp://{agent_name}"

    def test_agent_name_derivation_from_provider(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        for provider_id in ["claude-acp", "codex-acp", "gemini-acp"]:
            overlay = HERMES_OVERLAYS[provider_id]
            agent = extract_agent_from_url(overlay.base_url_override)
            assert agent == provider_id.removesuffix("-acp")
