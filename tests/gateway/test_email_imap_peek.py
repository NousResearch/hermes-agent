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

# The fetch loop performs a bounded header preflight (_PREAUTH_FETCH) before the
# body fetch, so tests must answer header fetches with headers only.
_SAMPLE_HEADERS = _SAMPLE_RAW.split(b"\n\n")[0] + b"\n\n"
_BODY_SELECTORS = ("(BODY.PEEK[])", "(RFC822)")

_ACCEPT_ALL = lambda candidate: True  # noqa: E731


def _make_adapter(extra: dict) -> EmailAdapter:
    config = PlatformConfig(enabled=True, extra=extra)
    with patch.dict(os.environ, _EMAIL_ENV):
        return EmailAdapter(config)


def _fetched_uids(
    adapter: EmailAdapter,
    *,
    unseen: bytes = b"5 2501",
    uidvalidity: int | None = None,
    all_uids: bytes = b"",
    fetch_status: str = "OK",
    fetch_fail_uids: set[bytes] | None = None,
):
    """Run _fetch_new_messages against a mocked IMAP server; return the list
    of UIDs that actually reached a body fetch (header preflights excluded)."""
    mock_imap = MagicMock()
    mock_imap.response.return_value = (
        ("UIDVALIDITY", [str(uidvalidity).encode()])
        if uidvalidity is not None
        else None
    )
    body_fetches: list[bytes] = []

    def _uid(cmd, *args):
        if cmd == "search":
            return ("OK", [all_uids if args[-1] == "ALL" else unseen])
        if cmd == "fetch":
            uid, selector = args[0], args[1]
            if selector.startswith("(BODY.PEEK[HEADER.FIELDS"):
                return ("OK", [(b"1 (BODY.PEEK[HEADER.FIELDS ...])", _SAMPLE_HEADERS)])
            body_fetches.append(uid)
            if fetch_status != "OK" or (fetch_fail_uids and uid in fetch_fail_uids):
                return ("NO", [])
            return ("OK", [(b"1 (BODY.PEEK[])", _SAMPLE_RAW)])
        return ("OK", [b""])

    mock_imap.uid.side_effect = _uid

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=mock_imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"):
        adapter._fetch_new_messages(_ACCEPT_ALL)

    return body_fetches


def _fetch_selector(extra: dict) -> str:
    """Run _fetch_new_messages against a mocked IMAP server and return the
    BODY FETCH selector that was passed to imap.uid("fetch", uid, <selector>)
    (the header preflight always uses BODY.PEEK on headers and is ignored)."""
    adapter = _make_adapter(extra)
    mock_imap = MagicMock()

    def _uid(cmd, *args):
        if cmd == "search":
            return ("OK", [b"123"])
        if cmd == "fetch":
            selector = args[1]
            if selector.startswith("(BODY.PEEK[HEADER.FIELDS"):
                return ("OK", [(b"1 (BODY.PEEK[HEADER.FIELDS ...])", _SAMPLE_HEADERS)])
            return ("OK", [(b"1 (BODY.PEEK[])", _SAMPLE_RAW)])
        return ("OK", [b""])

    mock_imap.uid.side_effect = _uid

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=mock_imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"):
        adapter._fetch_new_messages(_ACCEPT_ALL)

    body_calls = [
        c for c in mock_imap.uid.call_args_list
        if c.args and c.args[0] == "fetch" and c.args[2] in _BODY_SELECTORS
    ]
    assert body_calls, "expected a body imap.uid('fetch', ...) call"
    return body_calls[0].args[2]


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


# --- UIDVALIDITY epoch tracking ----------------------------------------------

def test_uidvalidity_change_resets_and_reseeds_uid_state():
    """A new UIDVALIDITY epoch may restart UIDs below the old watermark; the
    adapter must reseed from the new epoch and process only later arrivals."""
    adapter = _make_adapter({})
    adapter._uidvalidity = 10
    adapter._seed_seen_uids(str(i).encode() for i in range(1, 2501))
    assert adapter._uid_watermark == 2500

    fetched = _fetched_uids(
        adapter,
        unseen=b"1 2 3",
        uidvalidity=11,
        all_uids=b"1 2",
    )

    assert adapter._uidvalidity == 11
    assert fetched == [b"3"]
    assert adapter._uid_watermark == 3


def test_uidvalidity_unknown_to_known_reseeds_existing_replay_state():
    """A newly visible epoch cannot inherit a watermark from an unknown epoch."""
    adapter = _make_adapter({})
    adapter._seed_seen_uids([b"100", b"101"])
    assert adapter._uidvalidity is None

    fetched = _fetched_uids(
        adapter,
        unseen=b"1 2 3",
        uidvalidity=11,
        all_uids=b"1 2",
    )

    assert adapter._uidvalidity == 11
    assert fetched == [b"3"]
    assert adapter._uid_watermark == 3


def test_uidvalidity_reseed_snapshot_survives_empty_inbox_early_return():
    adapter = _make_adapter({})
    adapter._uidvalidity = 10
    adapter._seed_seen_uids([b"100", b"101"])
    adapter._save_uid_snapshot()

    fetched = _fetched_uids(
        adapter,
        unseen=b"",
        uidvalidity=11,
        all_uids=b"1 2",
    )

    assert fetched == []
    snapshot = adapter._seen_uids_snapshot[adapter._address]
    assert snapshot["uidvalidity"] == 11
    assert snapshot["uid_watermark"] == 2
    assert snapshot["seen_uids"] == {b"1", b"2"}


def test_persistent_fetch_failure_does_not_starve_later_uids():
    """A failed UID remains retryable without blocking newer messages."""
    adapter = _make_adapter({})
    adapter._seed_seen_uids([b"1", b"2"])

    first_attempt = _fetched_uids(
        adapter,
        unseen=b"3 4 5",
        fetch_fail_uids={b"4"},
    )

    assert first_attempt == [b"3", b"4", b"5"]
    assert adapter._uid_watermark == 5
    assert adapter._pending_fetch_uids == {b"4"}
    assert adapter._last_fetch_failed is True

    # A persistent refusal retries only the gap; already-consumed UID 5 does
    # not replay even though BODY.PEEK[] leaves it UNSEEN.
    second_attempt = _fetched_uids(
        adapter,
        unseen=b"4 5",
        fetch_fail_uids={b"4"},
    )
    assert second_attempt == [b"4"]
    assert adapter._pending_fetch_uids == {b"4"}

    # Once the server accepts the UID, the exception is cleared while the
    # watermark remains at the highest consumed UID.
    recovered = _fetched_uids(adapter, unseen=b"4 5")
    assert recovered == [b"4"]
    assert adapter._pending_fetch_uids == set()
    assert adapter._uid_watermark == 5


# --- rejected-mail STORE gating under PEEK -----------------------------------

def _rejected_fetch_run(extra: dict) -> MagicMock:
    """Fetch one unseen message with an always-reject gate; return the mock IMAP."""
    adapter = _make_adapter(extra)
    mock_imap = MagicMock()

    def _uid(cmd, *args):
        if cmd == "search":
            return ("OK", [b"7"])
        if cmd == "fetch":
            return ("OK", [(b"1 (BODY.PEEK[HEADER.FIELDS ...])", _SAMPLE_HEADERS)])
        return ("OK", [b""])

    mock_imap.uid.side_effect = _uid

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=mock_imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"):
        adapter._fetch_new_messages(lambda candidate: False)

    return mock_imap


def test_rejected_mail_not_marked_seen_under_peek():
    """Under the PEEK default, rejecting a sender must not STORE \\Seen on the
    server — app-side replay state alone suppresses the message."""
    mock_imap = _rejected_fetch_run({})
    store_calls = [c for c in mock_imap.uid.call_args_list if c.args and c.args[0] == "store"]
    assert store_calls == []


def test_rejected_mail_marked_seen_in_legacy_mode():
    """With imap_peek: false, rejected mail keeps the legacy \\Seen STORE so the
    next UNSEEN search excludes it."""
    mock_imap = _rejected_fetch_run({"imap_peek": False})
    store_calls = [c for c in mock_imap.uid.call_args_list if c.args and c.args[0] == "store"]
    assert len(store_calls) == 1
    assert store_calls[0].args[1:4] == (b"7", "+FLAGS.SILENT", r"(\Seen)")


# --- reconnect snapshot (dict payload + epoch verification) -------------------
def test_reconnect_snapshot_preserves_full_uid_state():
    first = _make_adapter({})
    first._uidvalidity = 10
    first._seed_seen_uids([b"1", b"2"])
    first._record_consumed_uid(b"3")
    first._pending_fetch_uids.add(b"4")
    first._save_uid_snapshot()

    second = _make_adapter({})
    snapshot = second._seen_uids_snapshot[second._address]

    assert second._restore_uid_snapshot(snapshot, current_uidvalidity=10) is True
    assert second._seen_uids == {b"1", b"2", b"3"}
    assert second._uid_watermark == 3
    assert second._uidvalidity == 10
    assert second._pending_fetch_uids == {b"4"}


def test_reconnect_snapshot_rejects_unknown_current_uidvalidity():
    first = _make_adapter({})
    first._uidvalidity = 10
    first._seed_seen_uids([b"100", b"101"])
    first._save_uid_snapshot()

    second = _make_adapter({})
    snapshot = second._seen_uids_snapshot[second._address]

    assert second._restore_uid_snapshot(snapshot, current_uidvalidity=None) is False
    assert second._seen_uids == set()
    assert second._uid_watermark is None


def test_reconnect_snapshot_rejects_changed_uidvalidity():
    first = _make_adapter({})
    first._uidvalidity = 10
    first._seed_seen_uids([b"100", b"101"])
    first._save_uid_snapshot()

    second = _make_adapter({})
    snapshot = second._seen_uids_snapshot[second._address]

    assert second._restore_uid_snapshot(snapshot, current_uidvalidity=11) is False
    assert second._seen_uids == set()
    assert second._uid_watermark is None


def test_reconnect_fails_closed_when_current_uidvalidity_is_unavailable():
    """Do not restore a stale watermark when the mailbox epoch is unknown."""
    EmailAdapter._seen_uids_snapshot.clear()
    first = _make_adapter({})
    first._uidvalidity = 10
    first._seed_seen_uids([b"100", b"101"])
    first._save_uid_snapshot()

    second = _make_adapter({})
    mock_imap = MagicMock()
    mock_imap.response.return_value = None
    mock_imap.uid.return_value = ("OK", [b"1 2"])

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=mock_imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"), patch.object(
        second, "_connect_smtp", return_value=MagicMock()
    ):
        connected = asyncio.run(second.connect(is_reconnect=True))

    assert connected is False
    assert second.fatal_error_code == "email_imap_connect_error"
    assert second._seen_uids == set()
    assert second._uid_watermark is None
    mock_imap.uid.assert_not_called()
    mock_imap.logout.assert_called_once()
    EmailAdapter._seen_uids_snapshot.clear()
