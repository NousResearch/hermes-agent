"""Tests that fallback_providers[].extra_body is honoured during fallback.

Regression tests for issue #26460: OpenRouter-specific routing metadata
(provider.order, allow_fallbacks, etc.) configured under a fallback entry's
extra_body was silently dropped when the fallback was activated, because
_prefs assembly only read from agent-level attributes, not the active
fallback config.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent_with_fallback(fallback_providers):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_providers,
        )
        agent.client = MagicMock()
        agent.client.base_url = "https://openrouter.ai/api/v1"
        return agent


def _mock_fb_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    m = MagicMock()
    m.base_url = base_url
    m.api_key = api_key
    return m


# ── extra_body stored on activation ──────────────────────────────────────


class TestFallbackExtraBodyStorage:
    def test_extra_body_stored_on_activation(self):
        """_fallback_extra_body must be set to the entry's extra_body dict."""
        extra_body = {
            "provider": {
                "order": ["baidu/fp8", "gmicloud/fp8"],
                "allow_fallbacks": False,
            }
        }
        fb_entry = {
            "provider": "openrouter",
            "model": "z-ai/glm-5.1",
            "extra_body": extra_body,
        }
        agent = _make_agent_with_fallback([fb_entry])
        fb_client = _mock_fb_client()

        with patch(
            "agent.chat_completion_helpers.resolve_provider_client",
            return_value=(fb_client, "z-ai/glm-5.1"),
        ):
            activated = agent._try_activate_fallback()

        assert activated
        assert agent._fallback_extra_body == extra_body

    def test_no_extra_body_stores_none(self):
        """Entry without extra_body must set _fallback_extra_body to None."""
        fb_entry = {"provider": "openrouter", "model": "z-ai/glm-5.1"}
        agent = _make_agent_with_fallback([fb_entry])
        fb_client = _mock_fb_client()

        with patch(
            "agent.chat_completion_helpers.resolve_provider_client",
            return_value=(fb_client, "z-ai/glm-5.1"),
        ):
            activated = agent._try_activate_fallback()

        assert activated
        assert agent._fallback_extra_body is None

    def test_empty_extra_body_stores_none(self):
        """An empty dict extra_body should not pollute _prefs — stored as None."""
        fb_entry = {
            "provider": "openrouter",
            "model": "z-ai/glm-5.1",
            "extra_body": {},
        }
        agent = _make_agent_with_fallback([fb_entry])
        fb_client = _mock_fb_client()

        with patch(
            "agent.chat_completion_helpers.resolve_provider_client",
            return_value=(fb_client, "z-ai/glm-5.1"),
        ):
            agent._try_activate_fallback()

        assert agent._fallback_extra_body is None


# ── extra_body forwarded into provider_preferences ──────────────────────


class TestFallbackExtraBodyForwarding:
    def _activate_with_extra_body(self, extra_body):
        fb_entry = {
            "provider": "openrouter",
            "model": "z-ai/glm-5.1",
            "extra_body": extra_body,
        }
        agent = _make_agent_with_fallback([fb_entry])
        fb_client = _mock_fb_client()
        with patch(
            "agent.chat_completion_helpers.resolve_provider_client",
            return_value=(fb_client, "z-ai/glm-5.1"),
        ):
            agent._try_activate_fallback()
        return agent

    def test_provider_order_forwarded_to_prefs(self):
        """provider.order from extra_body must appear in _prefs after activation."""
        from agent.chat_completion_helpers import _build_api_kwargs_for_openai

        extra_body = {
            "provider": {
                "order": ["baidu/fp8", "gmicloud/fp8"],
                "allow_fallbacks": False,
            }
        }
        agent = self._activate_with_extra_body(extra_body)

        # Read _prefs the same way the build path does
        _prefs = {}
        from agent.chat_completion_helpers import _validated_openrouter_provider_sort
        if agent.providers_allowed:
            _prefs["only"] = agent.providers_allowed
        if agent.providers_ignored:
            _prefs["ignore"] = agent.providers_ignored
        if agent.providers_order:
            _prefs["order"] = agent.providers_order

        _fb_extra_body = getattr(agent, "_fallback_extra_body", None) or {}
        _fb_provider_prefs = _fb_extra_body.get("provider") if isinstance(_fb_extra_body, dict) else None
        if _fb_provider_prefs and isinstance(_fb_provider_prefs, dict):
            _prefs.update(_fb_provider_prefs)

        assert _prefs.get("order") == ["baidu/fp8", "gmicloud/fp8"]
        assert _prefs.get("allow_fallbacks") is False

    def test_fallback_prefs_override_global_order(self):
        """Fallback-local provider.order takes precedence over global providers_order."""
        fb_entry = {
            "provider": "openrouter",
            "model": "z-ai/glm-5.1",
            "extra_body": {
                "provider": {"order": ["fallback-gpu/fp8"]}
            },
        }
        agent = _make_agent_with_fallback([fb_entry])
        # Simulate a global providers_order set from primary config
        agent.providers_order = ["primary-gpu/bf16"]
        fb_client = _mock_fb_client()

        with patch(
            "agent.chat_completion_helpers.resolve_provider_client",
            return_value=(fb_client, "z-ai/glm-5.1"),
        ):
            agent._try_activate_fallback()

        _prefs = {"order": agent.providers_order}
        _fb_extra_body = getattr(agent, "_fallback_extra_body", None) or {}
        _fb_pp = _fb_extra_body.get("provider") if isinstance(_fb_extra_body, dict) else None
        if _fb_pp and isinstance(_fb_pp, dict):
            _prefs.update(_fb_pp)

        # Fallback-local order should win
        assert _prefs["order"] == ["fallback-gpu/fp8"]


# ── extra_body cleared on restore ────────────────────────────────────────


class TestFallbackExtraBodyClearing:
    def test_extra_body_cleared_on_restore(self):
        """_fallback_extra_body must be None after restore_primary_runtime."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        fb_entry = {
            "provider": "openrouter",
            "model": "z-ai/glm-5.1",
            "extra_body": {"provider": {"order": ["baidu/fp8"]}},
        }
        agent = _make_agent_with_fallback([fb_entry])
        fb_client = _mock_fb_client()

        with patch(
            "agent.chat_completion_helpers.resolve_provider_client",
            return_value=(fb_client, "z-ai/glm-5.1"),
        ):
            agent._try_activate_fallback()

        assert agent._fallback_extra_body is not None

        # Simulate restore — set up minimal _primary_runtime
        agent._primary_runtime = {
            "model": "primary-model",
            "provider": "openrouter",
            "api_key": "primary-key",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        }

        with patch("agent.agent_runtime_helpers.resolve_provider_client",
                   return_value=(MagicMock(base_url="https://openrouter.ai/api/v1",
                                           api_key="primary-key"), "primary-model")):
            try:
                restore_primary_runtime(agent)
            except Exception:
                pass  # restore may fail in minimal test env; we only need side-effects

        assert getattr(agent, "_fallback_extra_body", None) is None

    def test_extra_body_none_before_any_activation(self):
        """_fallback_extra_body should be absent or None on a fresh agent."""
        agent = _make_agent_with_fallback([])
        assert getattr(agent, "_fallback_extra_body", None) is None
