"""``render_plain_text`` / ``strip_markdown(style=...)``: markdown kept readable on surfaces with no
markdown renderer (iMessage, SMS) via combinable caps / unicode / emoji styles."""
import pytest

from gateway.platforms.helpers import strip_markdown
from gateway.platforms.plain_text_style import PLAIN_TEXT_STYLES, parse_plain_text_style, render_plain_text

ALL = frozenset(PLAIN_TEXT_STYLES)


def render(text: str, *styles: str, keep: bool = False) -> str:
    return render_plain_text(text, frozenset(styles), keep_link_targets=keep)


@pytest.mark.parametrize("raw,expected", [
    (None, None), ("", None), ("strip", None), ("STRIP", None),
    ("plain", frozenset()), ("bogus", frozenset()),
    ("caps", frozenset({"caps"})), (" Unicode ", frozenset({"unicode"})),
    ("caps,emoji", frozenset({"caps", "emoji"})), ("caps+unicode", frozenset({"caps", "unicode"})),
    ("all", ALL), ("rich", ALL),
])
def test_parse_plain_text_style(raw, expected) -> None:
    assert parse_plain_text_style(raw) == expected


def test_unset_style_keeps_legacy_strip() -> None:
    text = "# Hi\n**bold** and ~~gone~~"
    assert strip_markdown(text) == strip_markdown(text, style="strip") == "Hi\nbold and ~~gone~~"


def test_style_routes_strip_markdown_through_the_renderer() -> None:
    assert strip_markdown("**bold**", style="caps") == "BOLD"


class TestInline:
    def test_plain_removes_markers(self) -> None:
        assert render("**b** *i* ***bi*** __u__ _e_ ~~s~~") == "b i bi u e s"

    def test_caps_uppercases_bold_only(self) -> None:
        assert render("**b** *i* ***bi***", "caps") == "B i BI"

    def test_unicode_glyphs(self) -> None:
        assert render("**Hi 42** *ok* ***yo***", "unicode") == "𝗛𝗶 𝟰𝟮 𝘰𝘬 𝙮𝙤"

    def test_unicode_strikethrough_uses_combining_stroke(self) -> None:
        assert render("~~ab c~~", "unicode") == "a̶b̶ c̶"

    def test_unicode_leaves_non_ascii(self) -> None:
        assert render("**café ✅**", "unicode") == "𝗰𝗮𝗳é ✅"

    def test_literal_asterisks_and_snake_case_survive(self) -> None:
        assert render("5 * 3 * 2 and file_name_here", *ALL) == "5 * 3 * 2 and file_name_here"


class TestLiterals:
    @pytest.mark.parametrize("styles", [(), ("caps",), ("unicode",), tuple(ALL)])
    def test_code_and_urls_are_never_restyled(self, styles) -> None:
        text = "**run `ls -la` at https://ex.com/a_b**\n```\necho **x** _y_\n```"
        out = render(text, *styles)
        assert "ls -la" in out and "https://ex.com/a_b" in out and "echo **x** _y_" in out

    def test_link_keeps_target_and_styles_label(self) -> None:
        assert render("[**Open**](https://ex.com/a_b)", "caps", keep=True) == "OPEN\nhttps://ex.com/a_b"

    def test_link_target_dropped_without_keep(self) -> None:
        assert render("[Open](https://ex.com)") == "Open"

    def test_non_http_target_keeps_label_only(self) -> None:
        assert render("[mail](mailto:a@b.c)", keep=True) == "mail"


class TestBlocks:
    def test_headings(self) -> None:
        assert render("## Plan **now**") == "Plan now"
        assert render("## Plan", "caps") == "PLAN"
        assert render("## Plan", "unicode") == "𝗣𝗹𝗮𝗻"
        assert render("## Plan", "caps", "unicode") == "𝗣𝗟𝗔𝗡"
        assert render("## Plan", "emoji") == "📌 Plan"
        assert render("### Small", "emoji") == "Small"

    def test_lists_plain(self) -> None:
        assert render("- a\n  - b\n- [ ] c\n- [x] d\n1. e") == "• a\n  ◦ b\n☐ c\n☑ d\n1. e"

    def test_lists_emoji(self) -> None:
        assert render("- a\n- [ ] c\n- [x] d\n3. e\n12. f", "emoji") == "🔹 a\n⬜ c\n✅ d\n3️⃣ e\n12. f"

    def test_quote_and_rule(self) -> None:
        assert render("> said **this**\n\n---", "caps") == "│ said THIS\n\n──────────"
        assert render("***", "emoji") == "➖➖➖"

    def test_table_becomes_bullet_groups(self) -> None:
        out = render("| Tool | Price |\n|------|-------|\n| Claude | $20 |", "emoji")
        assert out == "Claude\n🔹 Price: $20"

    def test_blank_runs_squeezed(self) -> None:
        assert render("a\n\n\n\nb") == "a\n\nb"
