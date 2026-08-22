"""RED-phase regression tests for #51903.

When a sub-agent is spawned via delegate_task with
delegation.reasoning_effort: none, the resolved child_reasoning is
{"enabled": False}. The chat_completions legacy kwargs path MUST honor
that and not emit extra_body.reasoning, otherwise non-thinking providers
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
        """Primary reproducer: sub-agent with enforced disabled reasoning
        must not emit extra_body.reasoning at all — non-thinking providers
        reject the request with HTTP 400."""
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            supports_reasoning=True,
            reasoning_config={"enabled": False},
        )
        assert "reasoning" not in kw.get("extra_body", {}), (
            f"reasoning_config={{'enabled': False}} must suppress "
            f"extra_body.reasoning, got {kw.get('extra_body', {}).get('reasoning')!r}"
        )

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

    def test_configured_effort_preserved_when_enabled(self, transport):
        """Regression guard: when reasoning is enabled with a non-default
        effort, the configured effort is emitted — not overwritten by 'medium'."""
        msgs = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="gpt-4o",
            messages=msgs,
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}

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
        assert "reasoning" not in kw.get("extra_body", {}), (
            "GitHub Models path must also honor reasoning_config.enabled=False"
        )
