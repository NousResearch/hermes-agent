"""Regression tests for OpenCode /v1 stripping during /model switch.

When switching to an Anthropic-routed OpenCode model mid-session (e.g.
``/model minimax-m2.7`` on opencode-go, or ``/model claude-sonnet-4-6``
on opencode-zen), the resolved base_url must have its trailing ``/v1``
stripped before being handed to the Anthropic SDK.

Without the strip, the SDK prepends its own ``/v1/messages`` path and
requests hit ``https://opencode.ai/zen/go/v1/v1/messages`` — a double
``/v1`` that returns OpenCode's website 404 page with HTML body.

``hermes_cli.runtime_provider.resolve_runtime_provider`` already strips
``/v1`` at fresh agent init (PR #4918), but the ``/model`` mid-session
switch path in ``hermes_cli.model_switch.switch_model`` was missing the
same logic — these tests guard against that regression.
"""

from unittest.mock import patch

import pytest

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _run_opencode_switch(
    raw_input: str,
    current_provider: str,
    current_model: str,
    current_base_url: str,
    explicit_provider: str = "",
    runtime_base_url: str = "",
):
    """Run switch_model with OpenCode mocks and return the result.

    runtime_base_url defaults to current_base_url; tests can override it
    to simulate the credential resolver returning a base_url different
    from the session's current one.
    """
    effective_runtime_base = runtime_base_url or current_base_url
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "sk-opencode-fake",
                "base_url": effective_runtime_base,
                "api_mode": "chat_completions",
            },
        ),
        patch(
            "hermes_cli.models.validate_requested_model",
            return_value=_MOCK_VALIDATION,
        ),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            current_api_key="sk-opencode-fake",
            explicit_provider=explicit_provider,
        )


class TestOpenCodeGoV1Strip:
    """OpenCode Go: ``/model minimax-*`` must strip /v1."""

    def test_switch_to_minimax_m27_strips_v1(self):
        """GLM-5 → MiniMax-M2.7: base_url loses trailing /v1."""
        result = _run_opencode_switch(
            raw_input="minimax-m2.7",
            current_provider="opencode-go",
            current_model="glm-5",
            current_base_url="https://opencode.ai/zen/go/v1",
        )

        assert result.success, f"switch_model failed: {result.error_message}"
        assert result.api_mode == "anthropic_messages"
        assert result.base_url == "https://opencode.ai/zen/go", (
            f"Expected /v1 stripped for anthropic_messages; got {result.base_url}"
        )

    def test_switch_to_minimax_m25_strips_v1(self):
        """Same behavior for M2.5."""
        result = _run_opencode_switch(
            raw_input="minimax-m2.5",
            current_provider="opencode-go",
            current_model="kimi-k2.5",
            current_base_url="https://opencode.ai/zen/go/v1",
        )

        assert result.success
        assert result.api_mode == "anthropic_messages"
        assert result.base_url == "https://opencode.ai/zen/go"

    def test_switch_to_glm_leaves_v1_intact(self):
        """OpenAI-compatible models (GLM, Kimi, MiMo) keep /v1."""
        result = _run_opencode_switch(
            raw_input="glm-5.1",
            current_provider="opencode-go",
            current_model="minimax-m2.7",
            current_base_url="https://opencode.ai/zen/go",  # stripped from previous Anthropic model
            runtime_base_url="https://opencode.ai/zen/go/v1",
        )

        assert result.success
        assert result.api_mode == "chat_completions"
        assert result.base_url == "https://opencode.ai/zen/go/v1", (
            f"chat_completions must keep /v1; got {result.base_url}"
        )

    def test_switch_to_kimi_leaves_v1_intact(self):
        result = _run_opencode_switch(
            raw_input="kimi-k2.5",
            current_provider="opencode-go",
            current_model="glm-5",
            current_base_url="https://opencode.ai/zen/go/v1",
        )

        assert result.success
        assert result.api_mode == "chat_completions"
        assert result.base_url == "https://opencode.ai/zen/go/v1"

    def test_trailing_slash_also_stripped(self):
        """``/v1/`` with trailing slash is also stripped cleanly."""
        result = _run_opencode_switch(
            raw_input="minimax-m2.7",
            current_provider="opencode-go",
            current_model="glm-5",
            current_base_url="https://opencode.ai/zen/go/v1/",
        )

        assert result.success
        assert result.api_mode == "anthropic_messages"
        assert result.base_url == "https://opencode.ai/zen/go"


class TestOpenCodeZenV1Strip:
    """OpenCode Zen: ``/model claude-*`` must strip /v1."""

    def test_switch_to_claude_sonnet_strips_v1(self):
        """Gemini → Claude on opencode-zen: /v1 stripped."""
        result = _run_opencode_switch(
            raw_input="claude-sonnet-4-6",
            current_provider="opencode-zen",
            current_model="gemini-3-flash",
            current_base_url="https://opencode.ai/zen/v1",
        )

        assert result.success
        assert result.api_mode == "anthropic_messages"
        assert result.base_url == "https://opencode.ai/zen"

    def test_switch_to_gemini_leaves_v1_intact(self):
        """Gemini on opencode-zen stays on chat_completions with /v1."""
        result = _run_opencode_switch(
            raw_input="gemini-3-flash",
            current_provider="opencode-zen",
            current_model="claude-sonnet-4-6",
            current_base_url="https://opencode.ai/zen",  # stripped from previous Claude
            runtime_base_url="https://opencode.ai/zen/v1",
        )

        assert result.success
        assert result.api_mode == "chat_completions"
        assert result.base_url == "https://opencode.ai/zen/v1"

    def test_switch_to_gpt_uses_codex_responses_keeps_v1(self):
        """GPT on opencode-zen uses codex_responses api_mode — /v1 kept."""
        result = _run_opencode_switch(
            raw_input="gpt-5.4",
            current_provider="opencode-zen",
            current_model="claude-sonnet-4-6",
            current_base_url="https://opencode.ai/zen",
            runtime_base_url="https://opencode.ai/zen/v1",
        )

        assert result.success
        assert result.api_mode == "codex_responses"
        assert result.base_url == "https://opencode.ai/zen/v1"


class TestAgentSwitchModelDefenseInDepth:
    """run_agent.AIAgent.switch_model() also strips /v1 as defense-in-depth."""

    def test_agent_switch_model_strips_v1_for_anthropic_messages(self):
        """Even if a caller hands in a /v1 URL, the agent strips it."""
        from run_agent import AIAgent

        # Build a bare agent instance without running __init__; we only want
        # to exercise switch_model's base_url normalization logic.
        agent = AIAgent.__new__(AIAgent)
        agent.model = "glm-5"
        agent.provider = "opencode-go"
        agent.base_url = "https://opencode.ai/zen/go/v1"
        agent.api_key = "sk-opencode-fake"
        agent.api_mode = "chat_completions"
        agent._client_kwargs = {}

        # Intercept the expensive client rebuild — we only need to verify
        # that base_url was normalized before it reached the Anthropic
        # client factory.
        captured = {}

        def _fake_build_anthropic_client(api_key, base_url, **kwargs):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            return object()  # placeholder client — no real calls expected

        # The downstream cache/plumbing touches a bunch of private state
        # that wasn't initialized above; we don't want to rebuild the full
        # runtime for this single assertion, so short-circuit after the
        # strip by raising inside the stubbed factory.
        class _Sentinel(Exception):
            pass

        def _raise_after_capture(api_key, base_url, **kwargs):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            raise _Sentinel("strip verified")

        with patch(
            "agent.anthropic_adapter.build_anthropic_client",
            side_effect=_raise_after_capture,
        ), patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=""), patch(
            "agent.anthropic_adapter._is_oauth_token", return_value=False
        ):
            with pytest.raises(_Sentinel):
                agent.switch_model(
                    new_model="minimax-m2.7",
                    new_provider="opencode-go",
                    api_key="sk-opencode-fake",
                    base_url="https://opencode.ai/zen/go/v1",
                    api_mode="anthropic_messages",
                )

        assert captured.get("base_url") == "https://opencode.ai/zen/go", (
            f"agent.switch_model did not strip /v1; passed {captured.get('base_url')} "
            "to build_anthropic_client"
        )
