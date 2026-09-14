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

    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_variation_buried_between_two_long_common_runs_still_separates(self, count):
        """The hard shape: the IDs agree on a long head AND on a long tail, and
        differ only in the middle. Neither keeping the head nor keeping the tail
        can separate them once both exceed the button width, so the label needs a
        deterministic fallback. Distinctness is the contract — the exact rendering
        is not asserted."""
        models = [f"anthropic.claude-{'a' * 45}{c}{'z' * 60}" for c in "XYZWV"[:count]]

        labels = model_button_labels(models)

        assert len(labels) == count
        assert len(set(labels)) == count, labels
        assert all(label.strip() for label in labels), labels
        assert all(len(label) <= 38 for label in labels), labels

    def test_labels_do_not_depend_on_the_order_models_arrive_in(self):
        """A fallback keyed on list position would relabel every button as soon as
        the catalog is reordered or paged differently. Same set in, same labels
        out."""
        models = [f"anthropic.claude-{'a' * 45}{c}{'z' * 60}" for c in "XYZ"]

        forward = model_button_labels(models)
        backward = model_button_labels(list(reversed(models)))

        assert set(forward) == set(backward)
        assert dict(zip(models, forward)) == dict(zip(reversed(models), backward))

    def test_fallback_cannot_collide_with_an_existing_literal_label(self):
        models = [f"anthropic.claude-{'a' * 45}{c}{'z' * 60}" for c in "XYZ"]
        # Reserve a string that looks exactly like the formatter's own fallback.
        literal = model_button_labels(models)[0]
        models += [literal, "moonshot.same", "moonshotai.same"]
        labels = model_button_labels(models)
        assert len(set(labels)) == len(models), labels
        assert all(0 < len(label) <= 38 for label in labels)


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


class TestNonBedrockProvidersKeepVerbatimLabels:
    """Label shaping is Bedrock-specific and must not reach other providers.

    ``_is_bedrock_provider`` gates the vendor drill-down, but the labels and the
    routing legend are computed for whatever list the picker renders. Stripping a
    ``openai.`` segment off an ``openai-codex`` ID, or explaining routing scopes
    that provider does not have, is a change to a provider this fix never claimed
    to touch.
    """

    def test_a_vendor_like_prefix_is_not_stripped_for_another_provider(self):
        from plugins.platforms.telegram.model_picker_display import model_button_labels as labels_for

        models = ["openai.gpt-6-astra", "openai.gpt-5.6-terra"]

        assert labels_for(models, bedrock=False) == models
        # Bedrock is where the prefix IS redundant, because the vendor step named it.
        assert labels_for(models, bedrock=True) == ["gpt-6-astra", "gpt-5.6-terra"]

    def test_no_routing_legend_for_a_non_bedrock_list(self):
        from plugins.platforms.telegram.model_picker_display import routing_legend

        assert routing_legend(["gpt-4o-mini", "o3"], "us") == ""

    def test_width_clamp_still_applies_and_stays_distinct(self):
        """The clamp is a Telegram limit, not a Bedrock nicety: it applies to every
        provider, and must not merge two distinct IDs into one button there either."""
        from plugins.platforms.telegram.model_picker_display import model_button_labels as labels_for

        models = [f"vendor/some-really-long-model-name-that-will-not-fit-{c}" for c in "AB"]

        labels = labels_for(models, bedrock=False)

        assert all(len(label) <= 38 for label in labels), labels
        assert len(set(labels)) == 2, labels

    @pytest.mark.asyncio
    async def test_real_route_leaves_a_non_bedrock_provider_verbatim(self):
        """Walked through the real callback dispatcher, not the helper alone.

        ``openai-codex`` ships ``openai.``-prefixed IDs, so this is the provider a
        shape-only gate would have re-labelled. It must land straight on the model
        list, with the IDs shown as advertised and no routing legend.
        """
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from gateway.config import PlatformConfig
        from plugins.platforms.telegram import adapter as telegram_adapter
        from plugins.platforms.telegram.adapter import TelegramAdapter

        class _Button:
            def __init__(self, text, callback_data=None):
                self.text, self.callback_data = text, callback_data

        class _Markup:
            def __init__(self, inline_keyboard):
                self.inline_keyboard = inline_keyboard

        original = (telegram_adapter.InlineKeyboardButton, telegram_adapter.InlineKeyboardMarkup)
        telegram_adapter.InlineKeyboardButton, telegram_adapter.InlineKeyboardMarkup = _Button, _Markup
        try:
            adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
            adapter._bot, adapter._app = AsyncMock(), MagicMock()
            models = ["openai.gpt-6-astra", "openai.gpt-5.6-terra"]
            adapter._model_picker_state["12345"] = {
                "providers": [{"slug": "openai-codex", "name": "Codex", "models": models,
                               "total_models": len(models)}],
                "current_model": models[0], "current_provider": "openai-codex",
                "session_key": "s", "on_model_selected": AsyncMock(return_value="ok"), "msg_id": 42}

            query = AsyncMock()
            query.data = "mp:openai-codex"
            query.message = MagicMock()
            query.message.chat_id = 12345
            query.from_user = MagicMock()
            query.edit_message_text = AsyncMock()
            await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())

            assert query.edit_message_text.await_count == 1, "tap reached no handler"
            kwargs = query.edit_message_text.call_args[1]
            buttons = [b for row in kwargs["reply_markup"].inline_keyboard for b in row]
            picks = [(b.text, b.callback_data) for b in buttons
                     if str(b.callback_data).startswith("mm:")]
            # Straight to the models (no vendor step) and IDs untouched.
            assert picks == [("openai.gpt-6-astra", "mm:0"), ("openai.gpt-5.6-terra", "mm:1")]
            assert not [b for b in buttons if str(b.callback_data).startswith("mvd:")]
            assert "in-region" not in kwargs["text"] and "= global" not in kwargs["text"]
        finally:
            telegram_adapter.InlineKeyboardButton, telegram_adapter.InlineKeyboardMarkup = original


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
