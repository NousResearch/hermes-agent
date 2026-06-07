from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter() -> DiscordAdapter:
    return object.__new__(DiscordAdapter)


def test_format_message_strips_leading_invisible_unicode():
    adapter = _adapter()

    assert adapter.format_message("\ufeff\u200b\u200c\u200e\u200fHello") == "Hello"


def test_format_message_preserves_invisible_unicode_after_start():
    adapter = _adapter()

    assert adapter.format_message("A\ufeffB\u200bC") == "A\ufeffB\u200bC"


def test_format_message_leaves_regular_content_unchanged():
    adapter = _adapter()

    assert adapter.format_message("Hello **Discord**") == "Hello **Discord**"
