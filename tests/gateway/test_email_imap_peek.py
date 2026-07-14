"""
Test that EmailAdapter respects the platforms.email.imap_peek config option.

Verifies the IMAP FETCH uses BODY.PEEK[] by default (so polling does not mark
unread messages as read) and RFC822 when imap_peek is explicitly false, and
that the selector actually reaches the imap.uid("fetch", ...) call.
"""

import os
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import EmailAdapter

_EMAIL_ENV = {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_IMAP_PORT": "993",
    "EMAIL_SMTP_HOST": "smtp.test.com",
    "EMAIL_SMTP_PORT": "587",
    "EMAIL_POLL_INTERVAL": "15",
}

# Minimal valid message body so downstream parsing in _fetch_new_messages
# does not raise; the assertion only cares about the FETCH selector.
_SAMPLE_RAW = (
    b"From: sender@test.com\n"
    b"To: hermes@test.com\n"
    b"Subject: peek test\n"
    b"Message-ID: <peek@test.com>\n"
    b"\n"
    b"hello\n"
)


def _make_adapter(extra: dict) -> EmailAdapter:
    config = PlatformConfig(enabled=True, extra=extra)
    with patch.dict(os.environ, _EMAIL_ENV):
        return EmailAdapter(config)


def _fetch_selector(extra: dict) -> str:
    """Run _fetch_new_messages against a mocked IMAP server and return the
    FETCH selector that was passed to imap.uid("fetch", uid, <selector>)."""
    adapter = _make_adapter(extra)
    mock_imap = MagicMock()

    def _uid(cmd, *args):
        if cmd == "search":
            return ("OK", [b"123"])
        if cmd == "fetch":
            return ("OK", [(b"1 (BODY.PEEK[])", _SAMPLE_RAW)])
        return ("OK", [b""])

    mock_imap.uid.side_effect = _uid

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=mock_imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"):
        adapter._fetch_new_messages()

    fetch_calls = [
        c for c in mock_imap.uid.call_args_list if c.args and c.args[0] == "fetch"
    ]
    assert fetch_calls, "expected an imap.uid('fetch', ...) call"
    return fetch_calls[0].args[2]


def test_imap_peek_defaults_to_true():
    """Without explicit config, imap_peek should default to True (BODY.PEEK[])."""
    assert _make_adapter({})._imap_peek is True


def test_imap_peek_false_restores_rfc822():
    """Setting imap_peek: false should disable PEEK and use RFC822."""
    assert _make_adapter({"imap_peek": False})._imap_peek is False


def test_imap_peek_true_explicit():
    """Explicitly setting imap_peek: true should enable PEEK."""
    assert _make_adapter({"imap_peek": True})._imap_peek is True


def test_imap_peek_string_false():
    """String 'false' should be coerced to bool False."""
    assert _make_adapter({"imap_peek": "false"})._imap_peek is False


def test_imap_peek_string_true():
    """String 'true' should be coerced to bool True."""
    assert _make_adapter({"imap_peek": "true"})._imap_peek is True


def test_fetch_uses_body_peek_by_default():
    """The FETCH call must receive (BODY.PEEK[]) by default."""
    assert _fetch_selector({}) == "(BODY.PEEK[])"


def test_fetch_uses_rfc822_when_peek_disabled():
    """With imap_peek: false, the FETCH call must receive (RFC822)."""
    assert _fetch_selector({"imap_peek": False}) == "(RFC822)"
