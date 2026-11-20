"""IMAP retry transitions must not lose earlier UIDs or replay consumed mail.

Regression coverage for the header/body preflight integration in PR #48224.
The adapter and reconnect probe are real; only the IMAP transport is mocked.
"""

import asyncio
from contextlib import contextmanager
import imaplib
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import (
    EmailAdapter,
    _MAX_PREAUTH_HEADER_BYTES,
    _PREAUTH_FETCH,
)

_HEADERS = b"From: sender@test.com\r\nSubject: retry test\r\n\r\n"
_BODY = _HEADERS + b"hello\r\n"


@pytest.fixture(autouse=True)
def isolate_snapshots():
    EmailAdapter._seen_uids_snapshot.clear()
    yield
    EmailAdapter._seen_uids_snapshot.clear()


def _make_adapter(peek):
    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "mock-password",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        adapter = EmailAdapter(PlatformConfig(enabled=True, extra={"imap_peek": peek}))
    adapter._uidvalidity = 10
    adapter._seed_seen_uids([b"1", b"2"])
    return adapter


@contextmanager
def _inbox(imap):
    yield imap


def _poll(adapter, *, failed_stage=None, status="NO", terminal=None):
    imap = MagicMock()
    imap.response.return_value = ("UIDVALIDITY", [b"10"])

    def uid_handler(command, *args):
        if command == "search":
            return "OK", [b"3 4"]
        if command == "fetch":
            uid, selector = args
            stage = "header" if selector == _PREAUTH_FETCH else "body"
            if uid == b"3" and stage == failed_stage:
                return status, []
            if stage == "header":
                if uid == b"3" and terminal == "unusable_headers":
                    return "OK", [None]
                headers = b"x" * (_MAX_PREAUTH_HEADER_BYTES + 1) if (
                    uid == b"3" and terminal == "oversized_headers"
                ) else _HEADERS
                return "OK", [(uid, headers)]
            if uid == b"3" and terminal == "malformed_body":
                return "OK", [None]
            return "OK", [(uid, _BODY)]
        return "OK", [b""]

    def authorize(candidate):
        if candidate["uid"] == b"3":
            if terminal == "authorization_error":
                raise ValueError("authorization unavailable")
            return terminal != "rejected"
        return True

    imap.uid.side_effect = uid_handler
    with patch.object(adapter, "_inbox", lambda: _inbox(imap)):
        messages = adapter._fetch_new_messages(authorize)
    return messages, imap


def _fetches(imap, stage):
    return [
        call.args[1] for call in imap.uid.call_args_list
        if call.args[0] == "fetch"
        and (call.args[2] == _PREAUTH_FETCH) == (stage == "header")
    ]


def _reconnect(peek):
    adapter = _make_adapter(peek)
    imap = MagicMock()
    imap.response.return_value = ("UIDVALIDITY", [b"10"])
    with patch.object(adapter, "_inbox", lambda: _inbox(imap)):
        assert adapter._probe_imap(is_reconnect=True)
    # A reconnect must restore the snapshot, not baseline away the retry gap.
    imap.uid.assert_not_called()
    return adapter


@pytest.mark.parametrize("peek", [True, False])
@pytest.mark.parametrize("stage", ["header", "body"])
@pytest.mark.parametrize("status", ["NO", "BAD"])
@pytest.mark.parametrize("reconnect", [False, True])
def test_fetch_gap_retries_after_later_success(peek, stage, status, reconnect):
    adapter = _make_adapter(peek)
    first, first_imap = _poll(adapter, failed_stage=stage, status=status)
    assert [message["uid"] for message in first] == [b"4"]
    assert adapter._uid_watermark == 4
    assert adapter._last_fetch_failed is True
    assert status in adapter._last_fetch_error
    assert adapter._pending_fetch_uids == {b"3"}
    assert b"3" not in adapter._seen_uids
    assert _fetches(first_imap, "body") == ([b"4"] if stage == "header" else [b"3", b"4"])
    snapshot = adapter._seen_uids_snapshot[adapter._address]
    assert snapshot["pending_fetch_uids"] == {b"3"}

    if reconnect:
        adapter = _reconnect(peek)
        assert adapter._uid_watermark == 4
        assert adapter._pending_fetch_uids == {b"3"}

    # Repeated failure must still not block a later UID or forget the gap.
    failed_again, _ = _poll(adapter, failed_stage=stage, status=status)
    assert failed_again == []
    assert adapter._pending_fetch_uids == {b"3"}
    recovered, recovered_imap = _poll(adapter)
    assert [message["uid"] for message in recovered] == [b"3"]
    assert _fetches(recovered_imap, "body") == [b"3"]
    assert adapter._uid_watermark == 4
    assert adapter._pending_fetch_uids == set()
    assert adapter._last_fetch_failed is False
    assert adapter._last_fetch_error == ""
    assert adapter._seen_uids_snapshot[adapter._address]["pending_fetch_uids"] == set()


@pytest.mark.parametrize("peek", [True, False])
@pytest.mark.parametrize("stage", ["header", "body"])
@pytest.mark.parametrize("terminal", [
    "rejected", "authorization_error", "unusable_headers", "oversized_headers", "malformed_body",
])
def test_retry_then_terminal_consumption_stops_fetching(peek, stage, terminal):
    adapter = _make_adapter(peek)
    first, _ = _poll(adapter, failed_stage=stage)
    assert [message["uid"] for message in first] == [b"4"]
    assert adapter._pending_fetch_uids == {b"3"}

    consumed, consumed_imap = _poll(adapter, terminal=terminal)
    assert consumed == []
    assert b"3" in adapter._seen_uids
    assert adapter._pending_fetch_uids == set()
    assert adapter._uid_watermark == 4
    assert adapter._seen_uids_snapshot[adapter._address]["pending_fetch_uids"] == set()
    stores = [call for call in consumed_imap.uid.call_args_list if call.args[0] == "store"]
    assert len(stores) == (0 if peek or terminal == "malformed_body" else 1)
    assert _fetches(consumed_imap, "body") == ([b"3"] if terminal == "malformed_body" else [])

    # Saved consumption must also suppress retries in a fresh adapter.
    fresh = _reconnect(peek)
    repeated, repeated_imap = _poll(fresh, terminal=terminal)
    assert repeated == []
    assert _fetches(repeated_imap, "header") == []
    assert _fetches(repeated_imap, "body") == []


@pytest.mark.parametrize("stage", ["header", "body"])
def test_partial_results_dispatched_before_reconnect(stage):
    adapter = _make_adapter(True)
    imap = MagicMock()
    imap.response.return_value = ("UIDVALIDITY", [b"10"])

    def uid_handler(command, *args):
        if command == "search":
            return "OK", [b"3 4"]
        uid, selector = args
        fetch_stage = "header" if selector == _PREAUTH_FETCH else "body"
        if uid == b"3" and fetch_stage == stage:
            return "NO", []
        return "OK", [(uid, _HEADERS if fetch_stage == "header" else _BODY)]

    imap.uid.side_effect = uid_handler
    events = []

    async def dispatch(message):
        events.append(("dispatch", message["uid"]))

    async def reconnect():
        events.append(("reconnect", None))

    with patch.object(adapter, "_inbox", lambda: _inbox(imap)), patch.object(
        adapter, "_sender_accepted", return_value=True,
    ), patch.object(adapter, "_dispatch_message", AsyncMock(side_effect=dispatch)), patch.object(
        adapter, "_notify_fatal_error", AsyncMock(side_effect=reconnect),
    ):
        asyncio.run(adapter._check_inbox())
    assert events == [("dispatch", b"4"), ("reconnect", None)]
    assert adapter.fatal_error_code == "email_imap_fetch_failed"
    assert adapter._seen_uids_snapshot[adapter._address]["pending_fetch_uids"] == {b"3"}


@pytest.mark.parametrize("known_epoch", [True, False])
@pytest.mark.parametrize("error_type", [imaplib.IMAP4.error, OSError, RuntimeError])
def test_uidvalidity_read_error_is_a_failed_poll(known_epoch, error_type):
    adapter = _make_adapter(True)
    if not known_epoch:
        adapter._uidvalidity = None
    imap = MagicMock()
    imap.response.side_effect = error_type("UIDVALIDITY response failed")
    imap.uid.return_value = ("OK", [b""])
    with patch.object(adapter, "_inbox", lambda: _inbox(imap)):
        assert adapter._fetch_new_messages(lambda _: True) == []
    assert adapter._last_fetch_failed is True
    assert "UIDVALIDITY response failed" in adapter._last_fetch_error
    assert adapter._uid_watermark == 2
    assert adapter._seen_uids == {b"1", b"2"}
    imap.uid.assert_not_called()
