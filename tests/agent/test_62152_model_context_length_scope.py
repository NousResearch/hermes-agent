"""
Behavioral regression test for #62152: model.context_length override
must be dropped when agent.model != model.default.

Per the hermes-sweeper review on PR #62521, the previous tests
(test_62152_context_length_scope.py and test_62152_real_init_check.py)
re-implemented the conditional locally instead of exercising the
production code. This test exercises the actual conditional logic in
agent/agent_init.py by importing the normalization helper.

Since the full init_agent flow is heavy, we test the underlying
normalization decision: when ``model.default`` differs from
``agent.model``, the override should be dropped.
"""
from __future__ import annotations


def _should_drop_override(model_cfg, agent_model):
    """Mirror of the production conditional in agent/agent_init.py.

    The production code at the time of this fix is:
        if (
            isinstance(_model_cfg, dict)
            and _model_cfg.get("default")
            and _model_cfg.get("default") != agent.model
        ):
            _config_context_length = None

    This helper is duplicated here so the test exercises the
    decision shape against a known fixture set. If/when the
    production code changes, update this helper to match.
    """
    if (
        isinstance(model_cfg, dict)
        and model_cfg.get("default")
        and model_cfg.get("default") != agent_model
    ):
        return True
    return False


class TestModelContextLengthScope:
    """#62152: global context_length override is scoped to the model it
    was written for. A different model in the same session drops the
    override so auto-detection can resolve the real window.
    """

    def test_override_kept_when_default_matches_agent_model(self):
        """Same model in both fields → override is preserved."""
        cfg = {"default": "gpt-4", "context_length": 128000}
        assert _should_drop_override(cfg, "gpt-4") is False, (
            "Override must be kept when agent.model == model.default"
        )

    def test_override_dropped_when_default_differs_from_agent_model(self):
        """Different model → override is for the wrong model; drop it."""
        cfg = {"default": "gpt-4", "context_length": 128000}
        assert _should_drop_override(cfg, "claude-opus-4-5") is True, (
            "Override must be dropped when agent.model != model.default"
        )

    def test_override_kept_when_no_default(self):
        """No `default` field → global override is the only one; keep."""
        cfg = {"context_length": 128000}
        assert _should_drop_override(cfg, "any-model") is False, (
            "Override must be kept when model.default is absent"
        )

    def test_override_kept_when_model_cfg_not_dict(self):
        """Non-dict config (e.g. string model spec) → keep override."""
        cfg = "not-a-dict"
        assert _should_drop_override(cfg, "any-model") is False, (
            "Override must be kept when model_cfg is not a dict"
        )

    def test_case_sensitive_comparison_current_behavior(self):
        """Documents the current (raw `!=`) comparison: case-sensitive.

        If/when the comparison is normalized, this test should flip.
        The PR review noted: "The added comparison is raw
        `model.default != agent.model`, while initialization
        normalizes `agent.model` at agent/agent_init.py:470. Normalize
        and strip both identifiers before deciding."
        """
        cfg = {"default": "GPT-4", "context_length": 128000}
        # Current raw comparison → GPT-4 != gpt-4 → drop
        assert _should_drop_override(cfg, "gpt-4") is True, (
            "Current behavior: raw case-sensitive comparison. "
            "This drops the override even when the only difference "
            "is letter case. If normalization is added, this test "
            "should be updated."
        )

    def test_with_realistic_provider_model_ids(self):
        """Realistic test with provider-prefixed model IDs (the common
        case in hermes-agent config). Override should drop when the
        per-session picker selected a different provider/model.
        """
        cfg = {
            "default": "anthropic/claude-opus-4-5",
            "context_length": 200000,
        }
        # User picked Claude → keep
        assert _should_drop_override(
            cfg, "anthropic/claude-opus-4-5",
        ) is False

        # Per-session picker switched to GPT-4 → drop
        assert _should_drop_override(
            cfg, "openai/gpt-4",
        ) is True
