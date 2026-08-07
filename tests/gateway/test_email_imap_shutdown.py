"""Regression tests for interruptible Email IMAP polling shutdown."""

import asyncio
import os
import threading
from unittest.mock import patch

from gateway.config import PlatformConfig
from plugins.platforms.email.adapter import EmailAdapter


_EMAIL_ENV = {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
}

_SAMPLE_RAW = (
    b"From: sender@test.com\n"
    b"To: hermes@test.com\n"
    b"Subject: shutdown test\n"
    b"Message-ID: <shutdown@test.com>\n"
    b"\n"
    b"hello\n"
)


def _make_adapter() -> EmailAdapter:
    with patch.dict(os.environ, _EMAIL_ENV):
        return EmailAdapter(PlatformConfig(enabled=True))


async def _wait_for_event(event: threading.Event, timeout: float = 1.0) -> None:
    async def _wait() -> None:
        while not event.is_set():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_wait(), timeout)


class _BlockingIMAP:
    def __init__(self):
        self.sock = self
        self.search_started = threading.Event()
        self.released = threading.Event()
        self.worker_finished = threading.Event()
        self.shutdown_called = threading.Event()

    def login(self, *_args):
        return "OK", []

    def select(self, *_args):
        return "OK", []

    def response(self, *_args):
        return None

    def uid(self, command, *_args):
        assert command == "search"
        self.search_started.set()
        self.released.wait(timeout=5)
        return "OK", [b""]

    def shutdown(self, _how):
        self.shutdown_called.set()
        self.released.set()

    def close(self):
        pass

    def logout(self):
        self.worker_finished.set()


def test_disconnect_interrupts_blocked_executor_poll():
    adapter = _make_adapter()
    imap = _BlockingIMAP()

    async def _scenario() -> None:
        adapter._running = True
        adapter._poll_task = asyncio.create_task(adapter._check_inbox())
        await _wait_for_event(imap.search_started)
        await asyncio.wait_for(adapter.disconnect(), timeout=1)
        await _wait_for_event(imap.worker_finished)

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"):
        asyncio.run(_scenario())

    assert imap.shutdown_called.is_set()
    assert adapter._active_poll_imap is None
    assert adapter._poll_task is None


class _BatchIMAP:
    def __init__(self, adapter: EmailAdapter):
        self.adapter = adapter
        self.fetched = []

    def login(self, *_args):
        return "OK", []

    def select(self, *_args):
        return "OK", []

    def response(self, *_args):
        return None

    def uid(self, command, *args):
        if command == "search":
            return "OK", [b"1 2 3"]
        if command == "fetch":
            self.fetched.append(args[0])
            self.adapter._imap_poll_stop.set()
            return "OK", [(b"1 (RFC822)", _SAMPLE_RAW)]
        raise AssertionError(f"unexpected IMAP command: {command}")

    def logout(self):
        return "BYE", []


def test_poll_batch_stops_after_shutdown_begins():
    adapter = _make_adapter()
    adapter._running = True
    imap = _BatchIMAP(adapter)

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=imap,
    ), patch("plugins.platforms.email.adapter._send_imap_id"):
        adapter._fetch_new_messages()

    assert imap.fetched == [b"1"]
    assert adapter._active_poll_imap is None


class _FailingIMAP:
    def login(self, *_args):
        raise RuntimeError("login failed")

    def logout(self):
        return "BYE", []


def test_poll_connection_clears_after_exception():
    adapter = _make_adapter()
    adapter._running = True
    imap = _FailingIMAP()

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
        return_value=imap,
    ):
        assert adapter._fetch_new_messages() == []

    assert adapter._active_poll_imap is None


def test_stopped_poll_does_not_open_new_connection():
    adapter = _make_adapter()
    adapter._imap_poll_stop.set()

    with patch(
        "plugins.platforms.email.adapter.imaplib.IMAP4_SSL",
    ) as imap_constructor:
        assert adapter._fetch_new_messages() == []

    imap_constructor.assert_not_called()
