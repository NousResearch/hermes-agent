"""``strip_markdown`` link handling: the default drops a link target (SMS, IRC, Feishu, QQ rely on it);
``keep_link_targets=True`` leaves a bare http(s) URL for platforms whose data detection re-links it."""
from gateway.platforms.helpers import strip_markdown


def test_keep_link_targets_keeps_http_urls_only() -> None:
    text = "see [Open it](https://example.com/i?x=1) or [mail](mailto:a@example.com)"
    assert strip_markdown(text) == "see Open it or mail"
    assert strip_markdown(text, keep_link_targets=True) == "see Open it\nhttps://example.com/i?x=1 or mail"
