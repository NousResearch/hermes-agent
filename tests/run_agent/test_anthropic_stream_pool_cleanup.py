"""Regression test for the stale-stream pool-cleanup misfiring on Anthropic sessions.

Bug: ``_replace_primary_openai_client`` was called unconditionally by all three
stream-cleanup paths in ``chat_completion_helpers``.  When the active
``api_mode`` is ``"anthropic_messages"`` the agent uses ``_anthropic_client``
and ``self.client`` is ``None``; ``_client_kwargs`` is an empty dict ``{}``.
The unconditional ``OpenAI(**{})`` call inside the rebuild then raises
``AuthenticationError`` / ``ValueError`` with the message
  "The api_key client option must be set … by setting the OPENAI_API_KEY
   environment variable"
even though the provider is Anthropic and the correct key is present.

Fix: ``_replace_primary_openai_client`` now short-circuits on
``api_mode == "anthropic_messages"`` and calls ``_rebuild_anthropic_client``
instead, which knows how to build the correct client type.

See: reeve errors.log 2026-06-23 (task t_2ac53e81)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


@pytest.fixture
def anthropic_agent():
    """Minimal AIAgent wired for provider=anthropic / api_mode=anthropic_messages.

    We patch ``run_agent.OpenAI`` so the initial client construction
    (chat_completions path) doesn't need a real key, then immediately flip the
    agent into Anthropic-native mode by setting the relevant attributes
    directly — the same state reached after a switch_model() to anthropic.
    """
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="sk-openai-placeholder",
            base_url="https://api.openai.com/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    # Simulate the state after switch_model() → anthropic:
    a.provider = "anthropic"
    a.api_mode = "anthropic_messages"
    a.client = None
    a._client_kwargs = {}          # empty — no OpenAI creds, by design
    a._anthropic_api_key = "sk-ant-correct-key"
    a._anthropic_base_url = "https://api.anthropic.com"
    a._anthropic_client = MagicMock()
    a._is_anthropic_oauth = False
    return a


class TestReplaceClientAnthropicNative:
    """_replace_primary_openai_client must not attempt OpenAI construction on
    Anthropic-native sessions."""

    def test_does_not_call_openai_constructor(self, anthropic_agent):
        """OpenAI() must never be invoked when api_mode=anthropic_messages."""
        with (
            patch("run_agent.OpenAI") as mock_openai,
            patch.object(
                anthropic_agent,
                "_rebuild_anthropic_client",
                return_value=None,
            ),
        ):
            result = anthropic_agent._replace_primary_openai_client(
                reason="stale_stream_pool_cleanup"
            )

        mock_openai.assert_not_called()
        assert result is True

    def test_calls_rebuild_anthropic_client(self, anthropic_agent):
        """The Anthropic rebuild path must be called instead."""
        with patch.object(
            anthropic_agent,
            "_rebuild_anthropic_client",
        ) as mock_rebuild:
            result = anthropic_agent._replace_primary_openai_client(
                reason="stream_retry_pool_cleanup"
            )

        mock_rebuild.assert_called_once()
        assert result is True

    def test_returns_false_and_logs_warning_on_rebuild_failure(
        self, anthropic_agent, caplog
    ):
        """If _rebuild_anthropic_client raises, return False and emit the right
        warning — not an OPENAI_API_KEY error."""
        import logging

        with (
            patch.object(
                anthropic_agent,
                "_rebuild_anthropic_client",
                side_effect=ValueError("Anthropic API key missing"),
            ),
            caplog.at_level(logging.WARNING, logger="run_agent"),
        ):
            result = anthropic_agent._replace_primary_openai_client(
                reason="stale_stream_pool_cleanup"
            )

        assert result is False
        # Warning must name the Anthropic client, not the OpenAI one.
        assert any(
            "Anthropic client" in r.message for r in caplog.records
        ), f"Expected 'Anthropic client' in warning; got: {[r.message for r in caplog.records]}"
        # The misleading OPENAI_API_KEY phrase must NOT appear.
        assert not any(
            "OpenAI client" in r.message for r in caplog.records
        ), "Unexpected 'OpenAI client' warning fired for Anthropic provider"

    def test_openai_path_still_works_for_openai_provider(self, anthropic_agent):
        """Regression guard: the OpenAI rebuild path still fires for non-Anthropic
        sessions (i.e. the guard only skips on anthropic_messages)."""
        anthropic_agent.api_mode = "chat_completions"
        anthropic_agent.provider = "openai"
        anthropic_agent._client_kwargs = {"api_key": "sk-openai-real", "base_url": "https://api.openai.com/v1"}
        fake_new_client = MagicMock()
        fake_new_client._client = MagicMock(is_closed=False)

        with patch.object(
            anthropic_agent,
            "_create_openai_client",
            return_value=fake_new_client,
        ) as mock_create:
            result = anthropic_agent._replace_primary_openai_client(
                reason="stale_stream_pool_cleanup"
            )

        mock_create.assert_called_once()
        assert result is True
