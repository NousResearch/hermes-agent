"""
Tests for the Email IMAP peek + restart replay-guard behavior.

Covers:
- ``platforms.email.imap_peek`` config coercion (BODY.PEEK[] vs RFC822).
- The real ``load_gateway_config()`` path: a top-level ``platforms.email.imap_peek``
  key must reach ``PlatformConfig.extra`` through the generic non-typed-key
  promotion, not only a hand-built ``extra`` dict.
- The consumed-UID watermark that keeps BODY.PEEK[] safe against bounded-set
  replay (#60637) and the fail-closed startup UID baseline.
"""

import asyncio
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
# does not raise; the assertions only care about the FETCH selector / UID.
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


def _fetched_uids(adapter: EmailAdapter, *, unseen: bytes = b"5 2501", fetch_status: str = "OK"):
    """Run _fetch_new_messages against a mocked IMAP server; return the list
    of UIDs that actually reached the imap.uid('fetch', ...) call."""
    mock_imap = MagicMock()

    def _uid(cmd, *args):
        if cmd == "search":
            return ("OK", [unseen])
        if cmd == "fetch":
            if fetch_status != "OK":
                return ("NO", [])
            return ("OK", [(b"1 (BODY.PEEK[])", _SAMPLE_RAW)])
        return ("OK", [b""])

    mock_imap.uid.side_effect = _uid

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=mock_imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"):
        adapter._fetch_new_messages()

    return [c.args[1] for c in mock_imap.uid.call_args_list if c.args and c.args[0] == "fetch"]


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


# --- config coercion --------------------------------------------------------

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


# --- FETCH selector ---------------------------------------------------------

def test_fetch_uses_body_peek_by_default():
    """The FETCH call must receive (BODY.PEEK[]) by default."""
    assert _fetch_selector({}) == "(BODY.PEEK[])"


def test_fetch_uses_rfc822_when_peek_disabled():
    """With imap_peek: false, the FETCH call must receive (RFC822)."""
    assert _fetch_selector({"imap_peek": False}) == "(RFC822)"


# --- real config-loading path (config.yaml -> PlatformConfig.extra) ---------

def test_imap_peek_reaches_extra_via_load_gateway_config(tmp_path, monkeypatch):
    """A top-level ``platforms.email.imap_peek`` in config.yaml must reach the
    email PlatformConfig.extra via the generic non-typed-key promotion — not
    only a nested ``extra:`` block."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  email:\n"
        "    imap_peek: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    email_cfg = config.platforms.get(Platform.EMAIL)
    assert email_cfg is not None, "email platform missing from config.platforms"
    assert email_cfg.extra.get("imap_peek") is False


def test_platforms_email_overrides_gateway_platforms_email(tmp_path, monkeypatch):
    """Top-level ``platforms`` has documented precedence over
    ``gateway.platforms`` for email adapter keys."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "gateway:\n"
        "  platforms:\n"
        "    email:\n"
        "      imap_peek: true\n"
        "platforms:\n"
        "  email:\n"
        "    imap_peek: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    email_cfg = config.platforms[Platform.EMAIL]
    assert email_cfg.extra.get("imap_peek") is False


# --- startup UID baseline (fail closed) --------------------------------------

def test_connect_fails_closed_when_uid_baseline_search_fails():
    """PEEK must not start without a baseline or old unread mail can replay."""
    adapter = _make_adapter({})
    mock_imap = MagicMock()
    mock_imap.uid.return_value = ("NO", [])

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=mock_imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"), patch.object(
        adapter, "_connect_smtp", return_value=MagicMock()
    ):
        connected = asyncio.run(adapter.connect())

    assert connected is False
    assert adapter._running is False
    mock_imap.logout.assert_called_once()


# --- consumed UID watermark (#60637) -----------------------------------------

def test_max_uid_picks_numeric_max():
    assert _make_adapter({})._max_uid([b"1", b"10", b"2", b"2500"]) == 2500


def test_max_uid_ignores_non_numeric():
    assert _make_adapter({})._max_uid([b"abc", b"5", None]) == 5


def test_max_uid_empty_is_none():
    assert _make_adapter({})._max_uid([]) is None


def test_seed_seen_uids_sets_watermark_to_max_and_trims():
    """Seeding records the highest UID from the full set before trimming."""
    adapter = _make_adapter({})
    uids = [str(i).encode() for i in range(1, 2501)]  # 2500 UIDs > 2000 cap
    adapter._seed_seen_uids(uids)
    assert adapter._uid_watermark == 2500
    assert len(adapter._seen_uids) <= adapter._seen_uids_max


def test_fetch_skips_preexisting_uids_under_peek():
    """Under BODY.PEEK[], a UID at/below the consumed watermark (here b'5',
    dropped from the trimmed _seen_uids) must NOT be replayed, while a UID
    above it (b'2501', genuinely new) is fetched."""
    adapter = _make_adapter({})  # peek defaults to True
    adapter._seed_seen_uids([str(i).encode() for i in range(1, 2501)])
    assert adapter._uid_watermark == 2500

    fetched = _fetched_uids(adapter)
    assert b"5" not in fetched, "pre-existing UID below watermark was replayed"
    assert b"2501" in fetched, "new UID above watermark was not fetched"


def test_consumed_watermark_advances_past_evicted_post_start_uid():
    """After enough new unread mail to trigger a trim, an evicted post-start
    UID must remain below the advancing watermark and never be fetched again."""
    adapter = _make_adapter({})
    adapter._seed_seen_uids(str(i).encode() for i in range(1, 2501))

    # The seed keeps 1,000 entries. Consuming 1,001 more crosses the 2,000 cap
    # and evicts UID 2501 from the bounded set while advancing the watermark.
    for uid in range(2501, 3502):
        adapter._record_consumed_uid(str(uid).encode())

    assert adapter._uid_watermark == 3501
    assert b"2501" not in adapter._seen_uids

    fetched = _fetched_uids(adapter, unseen=b"2501 3502")
    assert fetched == [b"3502"]
    assert adapter._uid_watermark == 3502


def test_failed_fetch_does_not_advance_consumed_watermark():
    """A transient fetch refusal must leave the UID eligible for retry."""
    adapter = _make_adapter({})
    adapter._seed_seen_uids([b"1", b"2"])

    attempted = _fetched_uids(adapter, unseen=b"3", fetch_status="NO")

    assert attempted == [b"3"]
    assert adapter._uid_watermark == 2
    assert b"3" not in adapter._seen_uids
