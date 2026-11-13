"""Unit tests for the Slack json-render card renderer (pure function, no adapter).

Golden fixture: the blocks output is byte-identical (modulo JSON key order) to
what ``@json-render/chat`` + ``@chat-adapter/slack`` produce for the same spec
(verified against ``cardToSlackBlocks`` in a Node PoC).
"""

import json

import pytest

from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.card_blocks import (
    card_blocks_payload,
    card_fallback_text,
    extract_card,
    render_card_blocks,
)

# The standard sample spec, as the standardComponentDefinitions describe it.
HOME_SPEC = {
    "root": "card",
    "elements": {
        "card": {
            "type": "Card",
            "props": {"title": "Home status", "subtitle": "from Jeeves, just now"},
            "children": ["intro", "kpis", "lights", "actions"],
        },
        "intro": {
            "type": "Text",
            "props": {"content": "Everything looks normal. The kitchen light is still on from dinner."},
            "children": [],
        },
        "kpis": {"type": "Fields", "props": {}, "children": ["f1", "f2", "f3"]},
        "f1": {"type": "Field", "props": {"label": "Lights on", "value": 3}, "children": []},
        "f2": {"type": "Field", "props": {"label": "Doors locked", "value": "2/3"}, "children": []},
        "f3": {"type": "Field", "props": {"label": "Motion (1h)", "value": "front door"}, "children": []},
        "lights": {
            "type": "Table",
            "props": {
                "headers": ["Light", "Room", "State"],
                "rows": [
                    ["Kitchen", "Kitchen", "on, 91%"],
                    ["Porch", "Front", "off"],
                    ["Living", "Living Room", "on, 40%"],
                ],
                "align": ["left", "left", "right"],
                "caption": "Hubitat + Hue devices",
            },
            "children": [],
        },
        "actions": {"type": "Actions", "props": {}, "children": ["off", "snapshot"]},
        "off": {
            "type": "Button",
            "props": {"id": "kitchen-off", "label": "Turn kitchen light off", "style": "primary"},
            "children": [],
        },
        "snapshot": {"type": "LinkButton", "props": {"label": "Front camera", "url": "http://grace:8123"}, "children": []},
    },
}

# Golden: cardToSlackBlocks() output for HOME_SPEC, captured from
# @chat-adapter/slack 4.43 (see the PR description for the capture script).
GOLDEN_BLOCKS = [
    {"type": "header", "text": {"type": "plain_text", "text": "Home status", "emoji": True}},
    {"type": "context", "elements": [{"type": "mrkdwn", "text": "from Jeeves, just now"}]},
    {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "Everything looks normal. The kitchen light is still on from dinner."},
    },
    {
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": "*Lights on*\n3"},
            {"type": "mrkdwn", "text": "*Doors locked*\n2/3"},
            {"type": "mrkdwn", "text": "*Motion (1h)*\nfront door"},
        ],
    },
    {
        "type": "data_table",
        "caption": "Hubitat + Hue devices",
        "rows": [
            [{"type": "raw_text", "text": "Light"}, {"type": "raw_text", "text": "Room"}, {"type": "raw_text", "text": "State"}],
            [{"type": "raw_text", "text": "Kitchen"}, {"type": "raw_text", "text": "Kitchen"}, {"type": "raw_text", "text": "on, 91%"}],
            [{"type": "raw_text", "text": "Porch"}, {"type": "raw_text", "text": "Front"}, {"type": "raw_text", "text": "off"}],
            [{"type": "raw_text", "text": "Living"}, {"type": "raw_text", "text": "Living Room"}, {"type": "raw_text", "text": "on, 40%"}],
        ],
    },
    {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "action_id": "kitchen-off",
                "style": "primary",
                "text": {"type": "plain_text", "text": "Turn kitchen light off", "emoji": True},
            },
            {
                "type": "button",
                "action_id": "link-http://grace:8123",
                "text": {"type": "plain_text", "text": "Front camera", "emoji": True},
                "url": "http://grace:8123",
            },
        ],
    },
]


def _canonical(blocks):
    return json.dumps(blocks, sort_keys=True)


def _card_message(spec=None, before="", after=""):
    body = json.dumps(spec if spec is not None else HOME_SPEC)
    return f"{before}```card\n{body}\n```{after}"


class TestExtractCard:
    def test_extracts_spec_and_strips_the_fence(self):
        spec, stripped = extract_card(_card_message(before="Look:\n", after="\ndone"))
        assert spec == HOME_SPEC
        assert stripped == "Look:\n\ndone"

    def test_no_fence_returns_content_unchanged(self):
        content = "plain markdown with ```json\n{}\n``` inside"
        spec, stripped = extract_card(content)
        assert spec is None
        assert stripped == content

    def test_invalid_json_is_not_a_card(self):
        spec, stripped = extract_card("```card\n{not json}\n```")
        assert spec is None
        assert "```card" in stripped  # falls back to markdown, fence intact

    def test_non_object_spec_is_not_a_card(self):
        assert extract_card("```card\n[1,2]\n```")[0] is None

    def test_unclosed_fence_is_not_a_card(self):
        spec, stripped = extract_card("```card\n{\"root\": \"card\"}")
        assert spec is None
        assert stripped.startswith("```card")

    def test_quoted_example_inside_an_outer_fence_is_not_the_card(self):
        example = json.dumps({"root": "c", "elements": {"c": {"type": "Card", "props": {"title": "Example"}, "children": []}}})
        real = json.dumps({"root": "c", "elements": {"c": {"type": "Card", "props": {"title": "Real"}, "children": []}}})
        content = f"Format:\n````markdown\n```card\n{example}\n```\n````\nAnswer:\n```card\n{real}\n```"
        spec, stripped = extract_card(content)
        assert spec["elements"]["c"]["props"]["title"] == "Real"
        assert "Example" in stripped and "Real" not in stripped

    def test_last_well_formed_fence_wins(self):
        a = json.dumps({"root": "c", "elements": {"c": {"type": "Card", "props": {"title": "A"}, "children": []}}})
        b = json.dumps({"root": "c", "elements": {"c": {"type": "Card", "props": {"title": "B"}, "children": []}}})
        spec, _ = extract_card(f"```card\n{a}\n```\nthen\n```card\n{b}\n```")
        assert spec["elements"]["c"]["props"]["title"] == "B"


class TestRenderCardBlocks:
    def test_matches_the_js_adapter_golden(self):
        assert _canonical(render_card_blocks(HOME_SPEC)) == _canonical(GOLDEN_BLOCKS)

    def test_empty_spec_returns_none(self):
        assert render_card_blocks({}) is None
        assert render_card_blocks({"root": "missing", "elements": {}}) is None

    def test_unknown_component_rejects_the_card(self):
        spec = {"root": "c", "elements": {"c": {"type": "Chart", "props": {"title": "t", "chart": {
            "type": "pie", "segments": [{"label": "a", "value": 1}]}}, "children": []}}}
        assert render_card_blocks(spec) is None

    def test_bad_props_reject_the_card(self):
        spec = {"root": "c", "elements": {"c": {"type": "Card", "props": {"title": 42}, "children": []}}}
        assert render_card_blocks(spec) is None
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["x"]},
            "x": {"type": "Image", "props": {"url": "http://insecure.example/x", "alt": "a"}, "children": []}}}
        assert render_card_blocks(spec) is None  # Image requires https

    def test_duplicate_action_ids_reject_the_card(self):
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["a"]},
            "a": {"type": "Actions", "props": {}, "children": ["b1", "b2"]},
            "b1": {"type": "Button", "props": {"id": "same", "label": "one"}, "children": []},
            "b2": {"type": "Button", "props": {"id": "same", "label": "two"}, "children": []}}}
        assert render_card_blocks(spec) is None

    def test_more_than_five_actions_reject_the_card(self):
        children = [f"b{i}" for i in range(6)]
        elements = {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["a"]},
            "a": {"type": "Actions", "props": {}, "children": children},
        }
        for i in range(6):
            elements[f"b{i}"] = {"type": "Button", "props": {"id": f"id{i}", "label": f"b{i}"}, "children": []}
        assert render_card_blocks({"root": "c", "elements": elements}) is None

    def test_oversized_table_rejects_the_card(self):
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["t"]},
            "t": {"type": "Table", "props": {"headers": ["a", "b"], "rows": [["1", "2"]] * 101}, "children": []}}}
        assert render_card_blocks(spec) is None

    def test_fields_split_across_the_ten_field_cap(self):
        pairs = {f"f{i}": {"type": "Field", "props": {"label": f"L{i}", "value": i}, "children": []} for i in range(12)}
        elements = {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["fs"]},
            "fs": {"type": "Fields", "props": {}, "children": list(pairs)},
            **pairs,
        }
        blocks = render_card_blocks({"root": "c", "elements": elements})
        assert blocks is not None
        sections = [b for b in blocks if b["type"] == "section" and "fields" in b]
        assert [len(s["fields"]) for s in sections] == [10, 2]

    def test_never_raises_on_hostile_input(self):
        hostile = {"root": "c", "elements": {"c": {"type": "Card", "props": {}, "children": ["c"]}}}
        assert render_card_blocks(hostile) is None  # self-referential children
        assert render_card_blocks({"root": [], "elements": []}) is None

    def test_mrkdwn_props_cannot_ping_or_spoof_links(self):
        # Data in props is a prompt-injection surface (a Jira title the agent
        # summarises). ``<!channel>`` must not broadcast, ``<url|label>`` must
        # not become a link, ``<@U…>`` must not mention.
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "T", "subtitle": "<!here> now"}, "children": ["fs", "x"]},
            "fs": {"type": "Fields", "props": {}, "children": ["f"]},
            "f": {"type": "Field", "props": {"label": "Who", "value": "<@U0123> & <https://evil.example|click>"}, "children": []},
            "x": {"type": "Text", "props": {"content": "<!channel> deploy"}, "children": []}}}
        blocks = render_card_blocks(spec)
        mrkdwn = json.dumps([b for b in blocks if b["type"] in ("context", "section")])
        assert "<!" not in mrkdwn and "<@" not in mrkdwn and "<https" not in mrkdwn
        assert "&lt;!channel&gt; deploy" in mrkdwn
        assert "&lt;@U0123&gt; &amp; &lt;https://evil.example|click&gt;" in mrkdwn

    def test_plain_text_surfaces_are_not_backslash_escaped(self):
        # header / button labels are plain_text: nothing is parsed there, and
        # Slack would show a backslash literally.
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "Q3 *Results* & more"}, "children": ["a"]},
            "a": {"type": "Actions", "props": {}, "children": ["b"]},
            "b": {"type": "Button", "props": {"id": "go", "label": "snake_case_label"}, "children": []}}}
        blocks = render_card_blocks(spec)
        assert blocks[0]["text"]["text"] == "Q3 *Results* & more"
        assert blocks[1]["elements"][0]["text"]["text"] == "snake_case_label"
        assert "\\" not in json.dumps(blocks)

    def test_table_caption_defaults_like_the_js_adapter(self):
        # ``caption`` is REQUIRED on data_table; a caption-less spec must not
        # produce invalid_blocks (which would drop the whole card).
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["t"]},
            "t": {"type": "Table", "props": {"headers": ["a"], "rows": [["1"]]}, "children": []}}}
        table = render_card_blocks(spec)[1]
        assert table["type"] == "data_table" and table["caption"] == "Table"

    def test_empty_table_is_a_plain_table_block(self):
        # data_table needs header + 1 row; the JS adapter emits ``table`` for
        # an empty result set.
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["t"]},
            "t": {"type": "Table", "props": {"headers": ["a", "b"], "rows": []}, "children": []}}}
        table = render_card_blocks(spec)[1]
        assert table["type"] == "table" and "caption" not in table and len(table["rows"]) == 1

    def test_table_over_the_aggregate_char_budget_rejects_the_card(self):
        # 100 rows x 20 cols x 50 chars passes the row/col caps but Slack caps a
        # data_table at 20k characters across cells.
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["t"]},
            "t": {"type": "Table", "props": {"headers": [f"h{i}" for i in range(20)],
                                              "rows": [["x" * 50] * 20] * 100}, "children": []}}}
        assert render_card_blocks(spec) is None

    def test_table_page_size_passes_through_clamped(self):
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["t"]},
            "t": {"type": "Table", "props": {"headers": ["a"], "rows": [["1"]], "pageSize": 500}, "children": []}}}
        assert render_card_blocks(spec)[1]["page_size"] == 100

    def test_card_image_url_requires_https_like_image(self):
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t", "imageUrl": "http://x/y.png"}, "children": []}}}
        assert render_card_blocks(spec) is None

    def test_unhandled_buttons_are_dropped_not_shipped(self):
        # A Button nobody handles shows "This app is not responding" on tap.
        # With a policy, unhandled buttons go, LinkButtons stay, and an Actions
        # row left empty disappears while the rest of the card still renders.
        blocks = render_card_blocks(HOME_SPEC, button_allowed=lambda action_id: False)
        actions = [b for b in blocks if b["type"] == "actions"]
        assert len(actions) == 1
        assert [e["action_id"] for e in actions[0]["elements"]] == ["link-http://grace:8123"]
        only_button = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "t"}, "children": ["a"]},
            "a": {"type": "Actions", "props": {}, "children": ["b"]},
            "b": {"type": "Button", "props": {"id": "nope", "label": "x"}, "children": []}}}
        blocks = render_card_blocks(only_button, button_allowed=lambda action_id: False)
        assert [b["type"] for b in blocks] == ["header"]
        # Registered ids survive.
        blocks = render_card_blocks(HOME_SPEC, button_allowed=lambda action_id: action_id == "kitchen-off")
        assert _canonical(blocks) == _canonical(GOLDEN_BLOCKS)


class TestFallbackText:
    def test_summary_covers_title_fields_table_and_buttons(self):
        text = card_fallback_text(HOME_SPEC)
        assert "*Home status*" in text
        assert "from Jeeves, just now" in text
        assert "Lights on: 3" in text
        assert "Kitchen" in text and "on, 91%" in text
        assert "[Turn kitchen light off]" in text
        assert "[Front camera](http://grace:8123)" in text

    def test_summary_escapes_slack_control_characters_only(self):
        # Slack has no backslash escape: ``*``/``_`` stay as-is (a ``\*`` would
        # render literally), while ``& < >`` MUST become entities.
        spec = {"root": "c", "elements": {
            "c": {"type": "Card", "props": {"title": "a * b _c_ <!channel> & co"}, "children": []}}}
        text = card_fallback_text(spec)
        assert "a * b _c_ &lt;!channel&gt; &amp; co" in text
        assert "\\" not in text


class TestCardBlocksPayload:
    def test_returns_stripped_text_fallback_and_blocks(self):
        payload = card_blocks_payload(_card_message(before="Look:\n"))
        assert payload is not None
        stripped, fallback, blocks = payload
        assert stripped == "Look:"
        assert "Home status" in fallback
        assert _canonical(blocks) == _canonical(GOLDEN_BLOCKS)

    def test_content_without_a_card_is_none(self):
        assert card_blocks_payload("just markdown") is None


# ---------------------------------------------------------------------------
# Adapter wiring
# ---------------------------------------------------------------------------


class TestAdapterCardPair:
    def _adapter(self, extra, handled=("kitchen-off",)):
        from gateway.config import PlatformConfig

        adapter = SlackAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
        # What ``_register_plugin_action_handlers`` records for plugin-registered ids.
        adapter._card_action_patterns = list(handled)
        return adapter

    def test_buttons_without_a_plugin_handler_are_dropped(self):
        adapter = self._adapter({"card_blocks": True}, handled=())
        pair = adapter._card_pair(_card_message())
        assert pair is not None
        actions = [b for b in pair[2] if b["type"] == "actions"]
        assert [e["action_id"] for e in actions[0]["elements"]] == ["link-http://grace:8123"]

    def test_regex_plugin_handlers_admit_matching_buttons(self):
        import re

        adapter = self._adapter({"card_blocks": True}, handled=(re.compile(r"^kitchen-"),))
        assert _canonical(adapter._card_pair(_card_message())[2]) == _canonical(GOLDEN_BLOCKS)

    def test_hermes_internal_action_ids_are_never_reachable_from_a_card(self):
        adapter = self._adapter({"card_blocks": True}, handled=())
        for internal in ("hermes_feedback", "hermes_clarify_choice_1", "hermes_clarify_other"):
            assert adapter._card_button_allowed(internal) is False

    def test_message_text_carries_prose_and_the_card_summary(self):
        # ``text`` is what notifications, screen readers and the blocks-rejected
        # retry show: dropping the card summary when prose exists loses the data.
        adapter = self._adapter({"card_blocks": True})
        stripped, fallback, _ = adapter._card_pair(_card_message(before="Status:\n"))
        text = adapter._card_text(stripped, fallback)
        assert text.startswith("Status:")
        assert "Home status" in text and "Lights on: 3" in text
        assert adapter._card_text("", fallback) == fallback
        assert adapter._card_text("", "") == " "

    def test_flag_off_leaves_content_alone(self):
        adapter = self._adapter({})
        assert adapter._card_pair(_card_message()) is None

    def test_flag_on_renders_and_strips(self):
        adapter = self._adapter({"card_blocks": True})
        pair = adapter._card_pair(_card_message(before="Status:\n"))
        assert pair is not None
        stripped, fallback, blocks = pair
        assert stripped == "Status:"
        assert "Home status" in fallback
        assert _canonical(blocks) == _canonical(GOLDEN_BLOCKS)

    def test_maybe_blocks_prefers_the_card(self):
        adapter = self._adapter({"card_blocks": True, "rich_blocks": True})
        content = _card_message(before="Status:\n")
        pair = adapter._card_pair(content)
        assert pair is not None
        assert adapter._maybe_blocks(content) == pair[2]

    def test_broken_card_falls_through_to_rich_blocks(self):
        content = "```card\n{not json}\n```\n# Header"
        adapter = self._adapter({"card_blocks": True, "rich_blocks": True})
        blocks = adapter._maybe_blocks(content)
        assert blocks is not None  # rich_blocks handled it instead
        # The fence renders verbatim (preformatted), not as a card.
        assert any("{not json}" in json.dumps(b) for b in blocks)
        # The fence stays in the content: the markdown path renders it as-is.
        assert "```card" in content

    def test_feedback_buttons_compose_with_cards(self):
        adapter = self._adapter({"card_blocks": True, "feedback_buttons": True})
        blocks = adapter._maybe_blocks(_card_message())
        assert blocks is not None
        assert blocks[-1]["type"] == "context_actions"
