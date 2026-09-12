"""Selectability invariants that survive truncation and unknown vendors (#94986).

Two contracts the label/grouping pair must keep for the picker to be usable, both
about *reaching an exact model*, not about looks:

* whatever the caller renders, the labels of one page stay pairwise distinct
  **after** the width clamp — two buttons ellipsized to the same text make the
  choice a coin flip, which is the original symptom;
* the vendor drill-down partitions the list it is given, so no advertised ID
  becomes unreachable just because its vendor segment is unknown to Hermes.

The vendor step is Bedrock-shaped-ID specific, so the tests also pin that a
provider whose IDs carry no known vendor keeps the original flat flow.
"""

from collections import Counter

import pytest

from plugins.platforms.telegram.model_picker_display import (
    group_models_by_vendor,
    model_button_labels,
)


class TestLabelsStayDistinctAfterTruncation:
    def test_long_ids_sharing_a_prefix_do_not_collapse_to_one_label(self):
        """Bedrock ships IDs whose first ~38 characters are identical (dated
        preview profiles, Stability's ``stable-image-*`` family). Clamping them
        to the button width without checking the result re-creates the very
        collision the label work exists to remove."""
        models = [
            "us.stability.stable-image-control-structure-preview-20260101-v1:0",
            "us.stability.stable-image-control-structure-preview-20260202-v1:0",
            "us.stability.stable-image-control-structure-preview-20260303-v1:0",
        ]

        labels = model_button_labels(models)

        assert len(set(labels)) == len(models), labels
        # The clamp must still do its job: no label may exceed the button width.
        assert all(len(label) <= 38 for label in labels), labels

    def test_truncation_keeps_the_distinguishing_tail_not_just_the_head(self):
        """A head-only clamp drops exactly the part that differs. The label must
        keep enough of the tail to tell two long IDs apart."""
        models = [
            "us.anthropic.claude-opus-5-extended-thinking-preview-v1:0",
            "us.anthropic.claude-opus-5-extended-thinking-preview-v2:0",
        ]

        labels = model_button_labels(models)

        assert len(set(labels)) == 2, labels
        assert labels[0].endswith("v1:0") and labels[1].endswith("v2:0"), labels


class TestVendorGroupingLosesNothing:
    # A listing that mixes known vendors with an ID whose vendor segment Hermes
    # has never heard of, plus one that is not Bedrock-shaped at all.
    MIXED = [
        "us.anthropic.claude-opus-5",
        "meta.llama4-70b",
        "brandnewvendor.super-model-1",
        "us.brandnewvendor.super-model-1",
        "plain-model-no-dot",
    ]

    def test_groups_partition_the_list_so_no_model_is_unreachable(self):
        """``indices`` are the only path from a vendor button to a model, so the
        groups must cover every position exactly once. An ID left out of every
        group cannot be selected at all once the drill-down is inserted — a new
        Bedrock vendor would silently disappear from the picker."""
        groups = group_models_by_vendor(self.MIXED)

        covered = [i for g in groups for i in g["indices"]]
        assert Counter(covered) == Counter(range(len(self.MIXED))), groups
        # Vendor buttons are labelled from ``label``: those must stay unique too.
        assert len({g["label"] for g in groups}) == len(groups)

    def test_unknown_vendors_are_reachable_and_keep_distinct_labels(self):
        """The catch-all group is only useful if what it scopes is selectable:
        every label inside it must be non-empty and distinct."""
        groups = group_models_by_vendor(self.MIXED)

        catch_all = [g for g in groups if g["vendor"] not in {"anthropic", "meta"}]
        assert catch_all, groups
        scoped = [self.MIXED[i] for g in catch_all for i in g["indices"]]
        assert "plain-model-no-dot" in scoped and "brandnewvendor.super-model-1" in scoped

        for group in groups:
            labels = model_button_labels([self.MIXED[i] for i in group["indices"]])
            assert all(label.strip() for label in labels), (group, labels)
            assert len(set(labels)) == len(labels), (group, labels)

    @pytest.mark.parametrize(
        "models",
        [["gpt-4o-mini", "o3"], ["mistral-large-latest"], ["a", "b", "c"]],
    )
    def test_a_list_without_known_vendors_gains_no_vendor_step(self, models):
        """The drill-down is worth a tap only for a Bedrock-shaped catalog. A
        plain provider list must yield no groups, which is how the adapter knows
        to keep the original two-step flow instead of routing every provider
        through a pointless single ``Other`` button."""
        assert group_models_by_vendor(models) == []


class TestOnlyBedrockGetsTheVendorStep:
    """The drill-down is gated on the provider, not on ID shape alone.

    ``openai-codex`` advertises ``openai.gpt-…`` IDs, so a shape-only gate would
    hand a second provider a vendor step nobody asked for. The gate has to
    recognise Bedrock under its aliases, because the picker passes whatever slug
    the listing carries.
    """

    @pytest.mark.parametrize("slug", ["bedrock", "aws", "aws-bedrock", "amazon-bedrock"])
    def test_bedrock_aliases_are_recognised(self, slug):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        assert TelegramAdapter._is_bedrock_provider(slug) is True

    @pytest.mark.parametrize("slug", ["openai", "openai-codex", "anthropic", "moa", ""])
    def test_other_providers_are_not_bedrock(self, slug):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        assert TelegramAdapter._is_bedrock_provider(slug) is False
