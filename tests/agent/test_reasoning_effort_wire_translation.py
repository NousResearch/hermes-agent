"""Wire translation for Hermes' extended reasoning-effort vocabulary (#89503).

Hermes' internal effort set extends the wire vocabulary with ``ultra`` (the
/reasoning command documents none..xhigh|max|ultra). OpenAI-compatible wires —
OpenRouter chief among them — accept exactly max|xhigh|high|medium|low|minimal|
none and reject the extension with HTTP 400:

    reasoning.effort: Invalid option: expected one of "max"|"xhigh"|"high"|
    "medium"|"low"|"minimal"|"none"

An ``ultra`` configured while the default model was Anthropic worked (the
Anthropic adapter maps its own levels), but the moment a per-job override
pinned an OpenRouter model the extension leaked through the OpenAI-compatible
transport untranslated and every call failed. ``_reasoning_config_for_model``
is the wire-compat chokepoint for this transport: it must cap the extension
for every model, not just the one vendor prefix that happened to be fixed
first.
"""

from agent.transports.chat_completions import _reasoning_config_for_model


class TestUltraEffortWireTranslation:
    def test_ultra_maps_to_max_for_any_model(self):
        """The extension level caps at the wire vocabulary for every model —
        including the OpenRouter vendor prefixes a per-job override pins
        (#89503's nvidia/ case) and models with no vendor prefix at all."""
        for model in (
            "nvidia/nemotron-3.5-lightning:free",
            "deepseek/deepseek-v4",
            "qwen/qwen3.5-coder",
            "some-internal-model",
        ):
            out = _reasoning_config_for_model(
                model, {"enabled": True, "effort": "ultra"}
            )
            assert out == {"enabled": True, "effort": "max"}, model

    def test_gpt_56_ultra_still_maps(self):
        """The original pre-existing mapping (gpt-5.6 + ultra → max) is
        preserved by the generalized one."""
        out = _reasoning_config_for_model(
            "gpt-5.6", {"enabled": True, "effort": "ultra"}
        )
        assert out == {"enabled": True, "effort": "max"}

    def test_wire_native_levels_pass_through_untouched(self):
        for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            cfg = {"enabled": True, "effort": level}
            assert _reasoning_config_for_model("any/model", cfg) == cfg

    def test_non_dict_and_none_pass_through(self):
        assert _reasoning_config_for_model("m", None) is None
        assert _reasoning_config_for_model("m", "not-a-dict") == "not-a-dict"
