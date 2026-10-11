"""Wire-scoped ``supports_vision_tool_messages=False`` veto for OpenCode Go (#132837).

The Go relay's tool-content rejections are chat-completions-era (#104731 422, #47026 400);
its responses/codex wire accepts image parts inside ``function_call_output`` (verified live
2026-10-04). The veto must lift on that wire and hold on chat_completions/anthropic_messages,
while unscoped vetoes (xiaomi/MiMo) stay unconditional.
"""

from unittest.mock import patch

from providers import routed_model_rejects_vision_tool_messages


class TestOpenCodeGoVetoWires:
    def test_responses_wire_models_lift_the_veto(self):
        """gpt-/grok-/muse-spark route through codex_responses: native tool-result vision
        must be admitted instead of forcing the aux-LLM text downgrade."""
        assert routed_model_rejects_vision_tool_messages("opencode-go", "muse-spark-1.3-contributor") is False
        assert routed_model_rejects_vision_tool_messages("opencode-go", "gpt-5.2-codex") is False
        assert routed_model_rejects_vision_tool_messages("opencode-go", "grok-4.1") is False

    def test_chat_completions_models_keep_the_veto(self):
        """Unprefixed ids fall back to the chat wire where the original 422/400s live."""
        assert routed_model_rejects_vision_tool_messages("opencode-go", "glm-5") is True

    def test_anthropic_wire_models_keep_the_veto(self):
        """minimax-/qwen route through anthropic_messages; no positive evidence for image
        parts there, so the veto stays (fail closed)."""
        assert routed_model_rejects_vision_tool_messages("opencode-go", "minimax-m2.5") is True
        assert routed_model_rejects_vision_tool_messages("opencode-go", "qwen4-max") is True

    def test_family_prefixed_id_is_normalized_before_wire_derivation(self):
        assert routed_model_rejects_vision_tool_messages("opencode-go", "opencode-go/muse-spark-1.3") is False
        assert routed_model_rejects_vision_tool_messages("opencode-go", "opencode-go/glm-5") is True

    def test_aggregator_target_lifts_only_on_the_codex_wire(self):
        """The aggregator-targeted leg of the predicate must honour the same wire scope."""
        assert routed_model_rejects_vision_tool_messages("openrouter", "opencode-go/muse-spark-1.3") is False
        assert routed_model_rejects_vision_tool_messages("openrouter", "opencode-go/glm-5") is True

    def test_unscoped_vetoes_stay_unconditional(self):
        """Providers without ``vision_tool_messages_veto_wires`` (xiaomi/MiMo, #89981) must
        not regress: their veto ignores wire derivation entirely."""
        assert routed_model_rejects_vision_tool_messages("xiaomi", "mimo-v2.5") is True
        assert routed_model_rejects_vision_tool_messages("openrouter", "xiaomi/mimo-v2.5") is True


class TestVisionToolsConsumers:
    def test_accepts_tool_result_images_on_codex_wire(self):
        """End-to-end gate: a vision-capable catalog entry on the codex wire must pass
        ``_accepts_tool_result_images`` (native fast path re-opens)."""
        from tools.vision_tools import _accepts_tool_result_images
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert _accepts_tool_result_images("opencode-go", "muse-spark-1.3-contributor", {}) is True
            assert _accepts_tool_result_images("opencode-go", "glm-5", {}) is False
