"""Unit tests for gateway/platforms/cards/builder.py — CardBuilder chainable API."""

import json
import unittest

from gateway.platforms.cards.builder import (
    CardBuilder,
    CARD_COLORS,
    COLORS,
    compact_number,
    error_card,
    format_elapsed,
    thinking_card,
    _escape_md,
    _format_code_block,
)


# ---------------------------------------------------------------------------
# format_elapsed
# ---------------------------------------------------------------------------

class TestFormatElapsed(unittest.TestCase):
    """Tests for format_elapsed() duration formatter."""

    def test_under_60_seconds_shows_decimal_seconds(self):
        self.assertEqual(format_elapsed(3200), "3.2s")

    def test_exactly_1_second(self):
        self.assertEqual(format_elapsed(1000), "1.0s")

    def test_over_60_seconds_shows_minutes_and_seconds(self):
        result = format_elapsed(125_000)  # 2m 5s
        self.assertEqual(result, "2m 5s")

    def test_zero_ms_shows_zero_seconds(self):
        self.assertEqual(format_elapsed(0), "0.0s")


# ---------------------------------------------------------------------------
# compact_number
# ---------------------------------------------------------------------------

class TestCompactNumber(unittest.TestCase):
    """Tests for compact_number() formatter."""

    def test_small_number_returned_as_integer_string(self):
        self.assertEqual(compact_number(42), "42")

    def test_thousands_shown_with_k_suffix(self):
        result = compact_number(1200)
        self.assertIn("k", result)
        self.assertIn("1.2", result)

    def test_millions_shown_with_m_suffix(self):
        result = compact_number(3_500_000)
        self.assertIn("m", result)

    def test_large_k_rounded_without_decimal(self):
        result = compact_number(200_000)
        self.assertEqual(result, "200k")


# ---------------------------------------------------------------------------
# _escape_md
# ---------------------------------------------------------------------------

class TestEscapeMd(unittest.TestCase):
    """Tests for _escape_md() Markdown escaper."""

    def test_escapes_backtick(self):
        self.assertIn("\\`", _escape_md("`code`"))

    def test_escapes_asterisk(self):
        self.assertIn("\\*", _escape_md("*bold*"))

    def test_escapes_underscore(self):
        self.assertIn("\\_", _escape_md("_italic_"))

    def test_plain_text_unchanged(self):
        self.assertEqual(_escape_md("hello world"), "hello world")

    def test_escapes_backslash(self):
        self.assertIn("\\\\", _escape_md("back\\slash"))


# ---------------------------------------------------------------------------
# CardBuilder: header()
# ---------------------------------------------------------------------------

class TestCardBuilderHeader(unittest.TestCase):
    """Tests for the header() method."""

    def test_header_sets_title(self):
        card = CardBuilder().header("My Title").build()
        self.assertEqual(card["header"]["title"]["content"], "My Title")

    def test_header_default_color_is_blue(self):
        card = CardBuilder().header("Title").build()
        self.assertEqual(card["header"]["template"], "blue")

    def test_header_semantic_alias_success_maps_to_green(self):
        card = CardBuilder().header("OK", color="success").build()
        self.assertEqual(card["header"]["template"], "green")

    def test_header_semantic_alias_error_maps_to_red(self):
        card = CardBuilder().header("Fail", color="error").build()
        self.assertEqual(card["header"]["template"], "red")

    def test_header_with_subtitle(self):
        card = CardBuilder().header("Title", subtitle="Sub").build()
        self.assertIn("subtitle", card["header"])
        self.assertEqual(card["header"]["subtitle"]["content"], "Sub")

    def test_header_without_subtitle_has_no_subtitle_key(self):
        card = CardBuilder().header("Title").build()
        self.assertNotIn("subtitle", card["header"])


# ---------------------------------------------------------------------------
# CardBuilder: add_text / add_markdown / add_divider
# ---------------------------------------------------------------------------

class TestCardBuilderElements(unittest.TestCase):
    """Tests for element-adding methods."""

    def test_add_text_creates_div_element(self):
        card = CardBuilder().add_text("hello").build()
        elem = card["elements"][0]
        self.assertEqual(elem["tag"], "div")
        self.assertEqual(elem["text"]["content"], "hello")

    def test_add_markdown_creates_markdown_element(self):
        card = CardBuilder().add_markdown("**bold**").build()
        elem = card["elements"][0]
        self.assertEqual(elem["tag"], "markdown")
        self.assertEqual(elem["content"], "**bold**")

    def test_add_divider_creates_hr_element(self):
        card = CardBuilder().add_divider().build()
        self.assertEqual(card["elements"][0]["tag"], "hr")

    def test_chained_calls_produce_multiple_elements(self):
        card = (
            CardBuilder()
            .add_text("first")
            .add_divider()
            .add_markdown("second")
            .build()
        )
        self.assertEqual(len(card["elements"]), 3)
        self.assertEqual(card["elements"][0]["tag"], "div")
        self.assertEqual(card["elements"][1]["tag"], "hr")
        self.assertEqual(card["elements"][2]["tag"], "markdown")


# ---------------------------------------------------------------------------
# CardBuilder: add_button
# ---------------------------------------------------------------------------

class TestCardBuilderButton(unittest.TestCase):
    """Tests for add_button()."""

    def test_add_button_creates_action_element(self):
        card = CardBuilder().add_button("Click me").build()
        elem = card["elements"][0]
        self.assertEqual(elem["tag"], "action")
        btn = elem["actions"][0]
        self.assertEqual(btn["tag"], "button")
        self.assertEqual(btn["text"]["content"], "Click me")

    def test_add_button_with_action_sets_value(self):
        action = {"type": "url", "url": "https://example.com"}
        card = CardBuilder().add_button("Open", action=action).build()
        btn = card["elements"][0]["actions"][0]
        self.assertEqual(btn["value"]["url"], "https://example.com")

    def test_add_button_default_type_is_default(self):
        card = CardBuilder().add_button("Btn").build()
        btn = card["elements"][0]["actions"][0]
        self.assertEqual(btn["type"], "default")

    def test_add_button_custom_type_is_preserved(self):
        card = CardBuilder().add_button("Danger", button_type="danger").build()
        btn = card["elements"][0]["actions"][0]
        self.assertEqual(btn["type"], "danger")


# ---------------------------------------------------------------------------
# CardBuilder: build() and build_v2()
# ---------------------------------------------------------------------------

class TestCardBuilderBuild(unittest.TestCase):
    """Tests for build() and build_v2() output structure."""

    def test_build_contains_config_and_elements(self):
        card = CardBuilder().build()
        self.assertIn("config", card)
        self.assertIn("elements", card)

    def test_build_config_has_wide_screen_mode(self):
        card = CardBuilder().build()
        self.assertTrue(card["config"]["wide_screen_mode"])

    def test_build_has_no_header_when_not_set(self):
        card = CardBuilder().build()
        self.assertNotIn("header", card)

    def test_build_has_header_when_set(self):
        card = CardBuilder().header("H").build()
        self.assertIn("header", card)

    def test_build_v2_has_schema_2_0(self):
        card = CardBuilder().build_v2()
        self.assertEqual(card["schema"], "2.0")

    def test_build_v2_uses_body_elements_not_top_level(self):
        card = CardBuilder().add_text("hello").build_v2()
        self.assertIn("body", card)
        self.assertIn("elements", card["body"])
        self.assertNotIn("elements", {k: card[k] for k in card if k != "body"})

    def test_build_v2_omits_wide_screen_mode_from_config(self):
        card = CardBuilder().build_v2()
        self.assertNotIn("wide_screen_mode", card.get("config", {}))

    def test_set_config_merges_into_config(self):
        card = CardBuilder().set_config(custom_key="custom_val").build()
        self.assertEqual(card["config"]["custom_key"], "custom_val")

    def test_chained_api_header_text_button_build_json(self):
        """Full chain test: header + add_text + add_button."""
        card = (
            CardBuilder()
            .header("Deployment", color="success")
            .add_text("All green")
            .add_button("View", action={"type": "url", "url": "https://logs.example.com"})
            .build()
        )

        self.assertEqual(card["header"]["template"], "green")
        self.assertEqual(len(card["elements"]), 2)

        # Serializable to JSON
        card_str = json.dumps(card, ensure_ascii=False)
        self.assertIn("Deployment", card_str)
        self.assertIn("All green", card_str)
        self.assertIn("logs.example.com", card_str)


# ---------------------------------------------------------------------------
# Factory functions: error_card / thinking_card
# ---------------------------------------------------------------------------

class TestFactoryFunctions(unittest.TestCase):
    """Tests for module-level card factory functions."""

    def test_thinking_card_returns_dict_with_thinking_text(self):
        card = thinking_card()
        self.assertIsInstance(card, dict)
        card_str = json.dumps(card)
        self.assertIn("Thinking", card_str)

    def test_error_card_has_red_header(self):
        card = error_card("Something went wrong")
        self.assertEqual(card["header"]["template"], "red")

    def test_error_card_contains_message(self):
        card = error_card("disk full")
        card_str = json.dumps(card)
        self.assertIn("disk full", card_str)

    def test_error_card_header_title_is_error(self):
        card = error_card("oops")
        self.assertEqual(card["header"]["title"]["content"], "Error")

    def test_thinking_card_is_json_serializable(self):
        card = thinking_card()
        # Should not raise
        json.dumps(card, ensure_ascii=False)

    def test_error_card_is_json_serializable(self):
        card = error_card("test message")
        json.dumps(card, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
