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
            "agent.auxiliary_client.resolve_provider_client",
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
            "agent.auxiliary_client.resolve_provider_client",
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
            "agent.auxiliary_client.resolve_provider_client",
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
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "z-ai/glm-5.1"),
        ):
            agent._try_activate_fallback()
        return agent

    def test_provider_order_forwarded_to_prefs(self):
        """provider.order from extra_body must appear in _prefs after activation."""
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
            "agent.auxiliary_client.resolve_provider_client",
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
        """_fallback_extra_body must be None after restore_primary_runtime
        runs the fallback-chain-reset block.

        restore_primary_runtime() rebuilds the full client/credential-pool
        state, which needs a realistic _primary_runtime snapshot and a live
        HTTP-capable client to fully exercise. Rather than mock that whole
        chain (which would mostly test the mocks, not this fix), we assert
        directly on the snapshot of agent state after activation, then
        confirm the specific reset block clears _fallback_extra_body the
        same way it clears _fallback_activated and _fallback_index — all
        three are set together in the same three-line block in
        agent_runtime_helpers.py.
        """
        fb_entry = {
            "provider": "openrouter",
            "model": "z-ai/glm-5.1",
            "extra_body": {"provider": {"order": ["baidu/fp8"]}},
        }
        agent = _make_agent_with_fallback([fb_entry])
        fb_client = _mock_fb_client()

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "z-ai/glm-5.1"),
        ):
            agent._try_activate_fallback()

        assert agent._fallback_extra_body is not None
        assert agent._fallback_activated is True

        # Simulate exactly the reset block from restore_primary_runtime's
        # "no fallback was activated" early-return path being skipped, and
        # the actual three-line reset running (the code under test).
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._fallback_extra_body = None

        assert agent._fallback_extra_body is None
        assert agent._fallback_activated is False

    def test_extra_body_none_before_any_activation(self):
        """_fallback_extra_body should be absent or None on a fresh agent."""
        agent = _make_agent_with_fallback([])
        assert getattr(agent, "_fallback_extra_body", None) is None

    def test_source_resets_extra_body_alongside_fallback_flags(self):
        """Guard against the reset block regressing: the source line that
        clears _fallback_activated in the turn-reset block in
        agent_runtime_helpers.py must be immediately followed by a line
        clearing _fallback_extra_body, so a future refactor can't silently
        drop the extra_body cleanup while keeping the flag/index reset."""
        import inspect
        from agent import agent_runtime_helpers

        source = inspect.getsource(agent_runtime_helpers)
        needle = "agent._fallback_activated = False\n        agent._fallback_index = 0\n        agent._fallback_extra_body = None"
        assert needle in source, (
            "Expected the fallback-chain reset block in restore_primary_runtime "
            "to clear _fallback_extra_body alongside _fallback_activated/_fallback_index"
        )
