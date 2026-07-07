from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter, _contains_bidi_mark


def _adapter() -> TelegramAdapter:
    return TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={},
        )
    )


def test_format_message_preserves_bullets_bold_italic_and_inline_code():
    rendered = _adapter().format_message(
        "- **Watchpost** קורא `MASTER_PLAN.md`\n"
        "- *TermFleet* נשאר תחת `devops/termfleet`"
    )

    assert "• *Watchpost* קורא `MASTER_PLAN.md`" in rendered
    assert "• _TermFleet_ נשאר תחת `devops/termfleet`" in rendered
    assert "\\-" not in rendered


def test_format_message_rewrites_tables_to_readable_row_groups():
    rendered = _adapter().format_message(
        "| Project | Status | Note |\n"
        "| --- | --- | --- |\n"
        "| Watchpost | TODO | reads `MASTER_PLAN.md` |\n"
        "| Hermes | DONE | **registered** |"
    )

    assert "\\|" not in rendered
    assert "*Watchpost*" in rendered
    assert "• Status: TODO" in rendered
    assert "• Note: reads `MASTER_PLAN.md`" in rendered
    assert "*Hermes*" in rendered
    assert "• Note: *registered*" in rendered


def test_format_message_keeps_tables_inside_code_blocks_untouched():
    rendered = _adapter().format_message(
        "```\n| raw | table |\n| --- | --- |\n| keep | pipes |\n```"
    )

    assert "```" in rendered
    assert "| raw | table |" in rendered
    assert "\\| raw" not in rendered


def test_format_message_wraps_bare_ltr_terms_on_hebrew_lines_without_bidi_marks():
    rendered = _adapter().format_message(
        "בדקתי את Watchpost מול TermFleet בקובץ MASTER_PLAN.md ובנתיב /tmp/hermes-agent/gateway.py"
    )

    assert "`Watchpost`" in rendered
    assert "`TermFleet`" in rendered
    assert "`MASTER_PLAN.md`" in rendered
    assert "`/tmp/hermes-agent/gateway.py`" in rendered
    assert not _contains_bidi_mark(rendered)


def test_format_message_does_not_wrap_ltr_terms_inside_existing_code_or_links():
    rendered = _adapter().format_message(
        "עברית עם `Watchpost` וקישור [Hermes](https://example.com/Hermes) וגם Codex"
    )

    assert rendered.count("`Watchpost`") == 1
    assert "[Hermes](https://example.com/Hermes)" in rendered
    assert "`Codex`" in rendered


def test_format_message_does_not_turn_plain_urls_into_code_spans():
    rendered = _adapter().format_message(
        "קישור רגיל https://example.com/Hermes נשאר קישור, אבל Watchpost עטוף"
    )

    assert "https://example\\.com/Hermes" in rendered
    assert "`https://" not in rendered
    assert "`Watchpost`" in rendered


def test_format_message_preserves_numbered_lists_and_emphasis_for_hebrew():
    rendered = _adapter().format_message(
        "1. **חשוב** להריץ Watchpost\n2. *בדיקה* מול TermFleet"
    )

    assert "1\\. *חשוב* להריץ `Watchpost`" in rendered
    assert "2\\. _בדיקה_ מול `TermFleet`" in rendered
