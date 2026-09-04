"""Regression tests: fallback entries on host-mandated single-protocol
endpoints (Kimi Coding api.kimi.com/coding) must resolve to the correct
api_mode, not the OpenAI chat_completions default.

Bug: ``try_activate_fallback()`` determined api_mode with its own provider
list and never consulted ``host_mandated_api_mode()``. A named
``kimi-coding`` fallback entry (no explicit base_url) resolved its client
to ``https://api.kimi.com/coding/v1`` (via ``_to_openai_base_url()``), the
/anthropic-suffix and api.anthropic.com checks did not match, and the swap
drove the session over the OpenAI wire — POST /coding/chat/completions →
HTTP 404 on every attempt (6 retries), then the chain skipped the entry.
The primary path was unaffected (it consults host_mandated_api_mode()).
"""

from unittest.mock import MagicMock, patch

from agent import chat_completion_helpers
from run_agent import AIAgent

from tests.run_agent.test_provider_fallback import _make_agent, _mock_client


class TestKimiFallbackApiMode:
    """kimi-coding fallback must land on anthropic_messages."""

    @staticmethod
    def _kimi_fb_client():
        # What resolve_provider_client() returns for kimi-coding after
        # _to_openai_base_url() rewrites /coding → /coding/v1.
        return _mock_client(base_url="https://api.kimi.com/coding/v1")

    def test_resolved_client_url_consults_host_mandated_mode(self):
        """Post-resolve pass: .../coding/v1 must map to anthropic_messages."""
        agent = _make_agent(
            fallback_model=[{"provider": "kimi-coding", "model": "kimi-k3"}]
        )
        agent._fallback_chain = [
            {"provider": "kimi-coding", "model": "kimi-k3"}
        ]

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(self._kimi_fb_client(), "kimi-k3"),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.provider == "kimi-coding"
        assert agent.api_mode == "anthropic_messages"

    def test_pre_resolve_hint_consults_host_mandated_mode(self):
        """Pre-resolve pass: explicit base_url hint .../coding wins too."""
        agent = _make_agent(
            fallback_model=[
                {
                    "provider": "kimi-coding",
                    "model": "kimi-k3",
                    "base_url": "https://api.kimi.com/coding",
                }
            ]
        )
        agent._fallback_chain = [
            {
                "provider": "kimi-coding",
                "model": "kimi-k3",
                "base_url": "https://api.kimi.com/coding",
            }
        ]

        captured = {}

        def fake_resolve(provider, model=None, raw_codex=True,
                         explicit_base_url=None, explicit_api_key=None,
                         api_mode=None):
            captured["api_mode"] = api_mode
            return _mock_client(base_url="https://api.kimi.com/coding/v1"), model

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=fake_resolve,
        ):
            assert agent._try_activate_fallback() is True

        assert captured["api_mode"] == "anthropic_messages"
        assert agent.api_mode == "anthropic_messages"

    def test_explicit_api_mode_entry_not_overridden(self):
        """An explicit fb.api_mode (even chat_completions) stays pinned."""
        agent = _make_agent(
            fallback_model=[
                {
                    "provider": "kimi-coding",
                    "model": "kimi-k3",
                    "api_mode": "chat_completions",
                }
            ]
        )
        agent._fallback_chain = [
            {
                "provider": "kimi-coding",
                "model": "kimi-k3",
                "api_mode": "chat_completions",
            }
        ]

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(self._kimi_fb_client(), "kimi-k3"),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "chat_completions"

    def test_openai_wire_provider_unchanged(self):
        """Regression guard: plain OpenAI-wire fallbacks stay chat_completions."""
        agent = _make_agent(
            fallback_model=[{"provider": "deepseek", "model": "deepseek-chat"}]
        )
        agent._fallback_chain = [
            {"provider": "deepseek", "model": "deepseek-chat"}
        ]

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(
                _mock_client(base_url="https://api.deepseek.com/v1"),
                "deepseek-chat",
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "chat_completions"
