"""RED-phase regression tests for #51903.

When a sub-agent is spawned via delegate_task with
delegation.reasoning_effort: none, the resolved child_reasoning is
{"enabled": False}. The chat_completions transport MUST honor that
and not emit extra_body.reasoning, otherwise non-thinking providers
return HTTP 400.
"""
import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


class TestLegacyPathHonorsReasoningDisabled:
    """Issue #51903: legacy kwargs path (no provider_profile) must not
    emit extra_body.reasoning when reasoning_config disables thinking."""

    def test_reasoning_disabled_omits_extra_body_reasoning(self, transport):
        """The primary reproducer: a Groq-routed sub-agent with
        delegation.reasoning_effort: none must not emit
        extra_body.reasoning at all."""
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            supports_reasoning=True,
            reasoning_config={"enabled": False},
        )
        # Bug: extra_body.reasoning is {"enabled": True, "effort": "medium"}
        # Fix: must not be present
        assert "reasoning" not in kw.get("extra_body", {}), (
            f"reasoning_config={{'enabled': False}} must suppress "
            f"extra_body.reasoning, got {kw.get('extra_body', {}).get('reasoning')!r}"
        )

    def test_reasoning_effort_none_omits_extra_body_reasoning(self, transport):
        """The exact delegation config from the issue: reasoning_config
        resolved from delegation.reasoning_effort: 'none' is
        {'enabled': False}. The transport must suppress the
        extra_body.reasoning emission."""
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            supports_reasoning=True,
            reasoning_config={"enabled": False},
        )
        assert "reasoning" not in kw.get("extra_body", {})

    def test_no_reasoning_config_keeps_default_medium(self, transport):
        """Regression guard: when no reasoning_config is set, the
        default behavior (emitting medium) is preserved."""
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o",
            messages=msgs,
            supports_reasoning=True,
            reasoning_config=None,
        )
        assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_no_supports_reasoning_omits_extra_body_reasoning(self, transport):
        """Regression guard: if supports_reasoning is False, no
        extra_body.reasoning is emitted even with reasoning_config set
        (existing behavior)."""
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o",
            messages=msgs,
            supports_reasoning=False,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert "reasoning" not in kw.get("extra_body", {})

    def test_github_models_with_disabled_reasoning_still_omits(self, transport):
        """GitHub Models path: when reasoning is disabled, the
        github_reasoning_extra must not be emitted either."""
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o",
            messages=msgs,
            supports_reasoning=True,
            is_github_models=True,
            github_reasoning_extra={"enabled": True, "effort": "high"},
            reasoning_config={"enabled": False},
        )
        # The fix should make the github_models branch also check
        # reasoning_config["enabled"] is False. Currently it doesn't.
        assert "reasoning" not in kw.get("extra_body", {}), (
            "GitHub Models path must also honor reasoning_config.enabled=False"
        )
