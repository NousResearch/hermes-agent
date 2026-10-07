"""Invariant: _rich_message_payload emits InputRichMessage.is_rtl exactly for
RTL-script content, and never for Latin-only content. Regression for the
markdown-only payload gap (#78715): the rich renderer picks paragraph base
direction itself and ignores injected RLM, so the flag must ride the payload."""

from plugins.platforms.telegram.adapter import TelegramAdapter


def _payload(content: str) -> dict:
    adapter = object.__new__(TelegramAdapter)  # tests construct without __init__
    return adapter._rich_message_payload(content)


def test_rtl_content_sets_is_rtl():
    for text in (
        "سلام — این **متن** فارسی است با `code`",
        "- `pool_floor` — کمترین دستی که کارِ روز میخواهد",
        "این متن فقط یک توکن لاتین دارد: BLOCKED",
    ):
        payload = _payload(text)
        assert payload["is_rtl"] is True


def test_latin_only_content_omits_is_rtl():
    for text in (
        "pure English **bold** text",
        "1234 digits only, no RTL characters",
        "| col | header |\n|---|---|\n| offered | hired |",
    ):
        payload = _payload(text)
        assert "is_rtl" not in payload
