"""Display contracts for Bedrock IDs in the Telegram ``/model`` picker (#94986).

Two invariants about being able to *pick* a model, not about looks: every label a
page renders is distinct (identical labels make the choice a coin flip), and no
label is ever blank or a rewritten ID.
"""

import pytest

from plugins.platforms.telegram.model_picker_display import (
    group_models_by_vendor,
    model_button_labels,
    pack_rows,
    routing_legend,
)

# Shapes a real ``provider_model_ids("bedrock")`` listing mixes: the same model
# advertised as a bare foundation-model ID and behind one or more routing
# namespaces, plus vendors whose model names contain dots of their own.
BEDROCK_SAMPLE = [
    "global.anthropic.claude-opus-5",
    "us.anthropic.claude-opus-5",
    "anthropic.claude-opus-5",
    "us.anthropic.claude-sonnet-4-6",
    "amazon.nova-lite-v1:0",
    "us.amazon.nova-lite-v1:0",
    "openai.gpt-5.6-terra",
    "us.openai.gpt-5.6-terra",
    "zai.glm-4.7",
    "moonshot.kimi-k2-thinking",
    "moonshotai.kimi-k2.5",
]


class TestBedrockPickerLabels:
    def test_labels_stay_distinct_and_only_carry_the_namespace_that_disambiguates(self):
        """The routing namespace is the differentiator when a model is advertised
        several times and pure noise when it is not. A bare ID never borrows a
        namespace it does not carry — lending it the configured region would make
        it identical to its regional twin — so the legend names that shape in the
        message body instead, where it costs no button width."""
        labels = model_button_labels(BEDROCK_SAMPLE)
        assert len(set(labels)) == len(BEDROCK_SAMPLE), labels

        # Distinctness must hold per vendor page too: that is the list the picker
        # renders after the drill-down, where labels are recomputed.
        for group in group_models_by_vendor(BEDROCK_SAMPLE):
            scoped = model_button_labels([BEDROCK_SAMPLE[i] for i in group["indices"]])
            assert len(set(scoped)) == len(scoped), f"{group['label']}: {scoped}"

        assert model_button_labels(["us.anthropic.claude-sonnet-4-6"]) == ["sonnet-4-6"]
        assert model_button_labels(
            ["global.anthropic.claude-opus-5", "us.anthropic.claude-opus-5", "anthropic.claude-opus-5"]
        ) == ["G: opus-5", "us: opus-5", "opus-5"]

        legend = routing_legend(
            ["global.anthropic.claude-opus-5", "us.anthropic.claude-opus-5", "anthropic.claude-opus-5"],
            "eu")
        assert "G: = global" in legend and "us: = us" in legend and "eu" in legend
        # Nothing to explain when no bare ID competes with a routed twin.
        assert routing_legend(["us.anthropic.claude-opus-5", "us.anthropic.claude-sonnet-5"]) == ""

    @pytest.mark.parametrize(
        "model_id",
        ["gpt-4o-mini", "mistral-large-latest", "global.anthropic", "anthropic.", "us.anthropic.",
         # ``claude-`` is stripped inside the Anthropic page, so this ID's whole
         # display name is the prefix — the shape that empties a label.
         "anthropic.claude-", "us.anthropic.claude-"],
    )
    def test_never_renders_an_empty_or_rewritten_label(self, model_id):
        """Telegram rejects blank button text and fails the whole message, so a
        non-Bedrock or degenerate ID must fall back to itself, verbatim."""
        assert model_button_labels([model_id]) == [model_id]

    def test_a_label_too_wide_for_a_column_gets_its_own_row(self):
        """Two columns halve the readable width, which is what ellipsized
        ``✓ AWS Bedrock (1…`` and hid the model count. Row packing must preserve
        order, since callbacks are positional."""
        labels = ["short", "✓ AWS Bedrock (154) — long enough to clip", "b", "c"]
        rows = pack_rows(labels)

        assert [i for row in rows for i in row] == list(range(len(labels)))
        wide_row = next(row for row in rows if 1 in row)
        assert wide_row == [1], rows
        assert max(len(row) for row in rows) <= 2
