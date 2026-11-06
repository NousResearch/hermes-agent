"""Matrix rich-text sanitizing keeps ordinary prose while removing unsafe raw attributes (#148258)."""

from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter


def _adapter() -> MatrixAdapter:
    return MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))


def test_plain_text_that_resembles_an_event_attribute_is_preserved():
    assert _adapter()._markdown_to_html("read only = no") == "read only = no"


def test_unsafe_attributes_are_removed_from_raw_html_tags():
    html = _adapter()._markdown_to_html(
        '<a href="javascript:alert(1)" onclick="alert(2)">safe label</a>')

    assert html == "<a>safe label</a>"
