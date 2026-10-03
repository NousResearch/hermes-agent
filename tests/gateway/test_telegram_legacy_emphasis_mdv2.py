"""Legacy Telegram MarkdownV2 emphasis: nested/combined and multiline bold.

Pin the format_message converter (rich messages off) so valid nested
asterisk emphasis is not swallowed by the bold regex and escaped as
literals. Unmatched markers stay escaped/literal (fail-open).
"""

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def test_legacy_nested_and_multiline_emphasis_mdv2():
    adapter = TelegramAdapter(PlatformConfig(token="test-token"))
    assert adapter.format_message("***bold italic***") == "_*bold italic*_"
    assert adapter.format_message("**bold *italic* text**") == "*bold _italic_ text*"
    assert adapter.format_message("**bold\ntext**") == "*bold\ntext*"


def test_legacy_emphasis_preserves_bullets_code_and_unmatched():
    """Pin skip/fail-open paths so a parser tweak cannot silently regress them."""
    adapter = TelegramAdapter(PlatformConfig(token="test-token"))
    # * bullet lists are not swallowed as italic (italic stays single-line)
    assert adapter.format_message("* Item one\n* Item two\n* Item three") == (
        "\\* Item one\n\\* Item two\n\\* Item three"
    )
    # code spans with * stay placeholders and are not rewritten as emphasis
    assert adapter.format_message("`foo *bar* baz`") == "`foo *bar* baz`"
    # unmatched markers stay escaped literals (fail-open, not stripped)
    assert adapter.format_message("**unclosed bold") == "\\*\\*unclosed bold"
