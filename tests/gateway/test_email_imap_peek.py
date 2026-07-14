"""
Tests for the Email IMAP peek + restart replay-guard behavior.

Covers:
- ``platforms.email.imap_peek`` config coercion (BODY.PEEK[] vs RFC822).
- The ``_apply_yaml_config`` bridge that routes a top-level
  ``platforms.email.imap_peek`` / ``skip_attachments`` key into
  ``PlatformConfig.extra`` (the generic shared-key loop does not cover these).
- The real ``load_gateway_config()`` path so the user-facing config.yaml key
  is proven to reach the adapter, not just a hand-built ``extra`` dict.
- The startup UID watermark that keeps BODY.PEEK[] default safe against
  restart replay in large (>_seen_uids_max) inboxes (#60637).
"""

import os
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import (
    EmailAdapter,
    _apply_yaml_config,
)

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


def _fetched_uids(adapter: EmailAdapter):
    """Run _fetch_new_messages against a mocked IMAP server; return the list
    of UIDs that actually reached the imap.uid('fetch', ...) call."""
    mock_imap = MagicMock()

    def _uid(cmd, *args):
        if cmd == "search":
            return ("OK", [b"5 2501"])
        if cmd == "fetch":
            return ("OK", [(b"1 (BODY.PEEK[])", _SAMPLE_RAW)])
        return ("OK", [b""])

    mock_imap.uid.side_effect = _uid

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=mock_imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"):
        adapter._fetch_new_messages()

    return [c.args[1] for c in mock_imap.uid.call_args_list if c.args and c.args[0] == "fetch"]


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
    adapter = _make_adapter({})
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
    assert fetch_calls[0].args[2] == "(BODY.PEEK[])"


def test_fetch_uses_rfc822_when_peek_disabled():
    """With imap_peek: false, the FETCH call must receive (RFC822)."""
    adapter = _make_adapter({"imap_peek": False})
    mock_imap = MagicMock()

    def _uid(cmd, *args):
        if cmd == "search":
            return ("OK", [b"123"])
        if cmd == "fetch":
            return ("OK", [(b"1 (RFC822)", _SAMPLE_RAW)])
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
    assert fetch_calls[0].args[2] == "(RFC822)"


# --- _apply_yaml_config bridge (pure function) ------------------------------

def test_apply_yaml_config_bridges_imap_peek_false():
    assert _apply_yaml_config({}, {"imap_peek": False}) == {"imap_peek": False}


def test_apply_yaml_config_bridges_skip_attachments():
    assert _apply_yaml_config({}, {"skip_attachments": True}) == {"skip_attachments": True}


def test_apply_yaml_config_returns_none_when_nothing_to_bridge():
    assert _apply_yaml_config({}, {}) is None


# --- real config-loading path (config.yaml -> PlatformConfig.extra) ---------

def test_imap_peek_reaches_extra_via_load_gateway_config(tmp_path, monkeypatch):
    """A top-level ``platforms.email.imap_peek`` in config.yaml must reach the
    email PlatformConfig.extra via the apply_yaml_config_fn bridge — not only a
    nested ``extra:`` block."""
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


def test_skip_attachments_reaches_extra_via_load_gateway_config(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  email:\n"
        "    skip_attachments: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    email_cfg = config.platforms.get(Platform.EMAIL)
    assert email_cfg is not None
    assert email_cfg.extra.get("skip_attachments") is True


# --- startup UID watermark + replay guard (#60637) --------------------------

def test_max_uid_picks_numeric_max():
    assert _make_adapter({})._max_uid([b"1", b"10", b"2", b"2500"]) == 2500


def test_max_uid_ignores_non_numeric():
    assert _make_adapter({})._max_uid([b"abc", b"5", None]) == 5


def test_max_uid_empty_is_none():
    assert _make_adapter({})._max_uid([]) is None


def test_seed_seen_uids_sets_watermark_to_max_and_trims():
    """Startup seeding must record the highest UID (watermark) from the full
    set before trimming, so trimming can never lower the replay floor."""
    adapter = _make_adapter({})
    uids = [str(i).encode() for i in range(1, 2501)]  # 2500 UIDs > 2000 cap
    adapter._seed_seen_uids(uids)
    assert adapter._startup_uid_watermark == 2500
    assert len(adapter._seen_uids) <= adapter._seen_uids_max


def test_fetch_skips_preexisting_uids_under_peek():
    """Under BODY.PEEK[], a UID at/below the startup watermark (here b'5',
    dropped from the trimmed _seen_uids) must NOT be replayed, while a UID
    above it (b'2501', genuinely new) is fetched."""
    adapter = _make_adapter({})  # peek defaults to True
    adapter._seed_seen_uids([str(i).encode() for i in range(1, 2501)])
    assert adapter._startup_uid_watermark == 2500

    fetched = _fetched_uids(adapter)
    assert b"5" not in fetched, "pre-existing UID below watermark was replayed"
    assert b"2501" in fetched, "new UID above watermark was not fetched"
