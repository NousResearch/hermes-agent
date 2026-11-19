"""The email connect probes must not run on the gateway's event loop.

``EmailAdapter.connect`` used to call its two blocking probes (``_probe_imap``,
``_probe_smtp``) inline. Pointed at an unreachable host the loop stayed parked for the whole
socket timeout — ``SMTP_CONNECT_TIMEOUT`` (30 s) plus a retry over an IPv4-only socket — and
three parked probes in a row is the loop-liveness watchdog's firing condition, so one
blackholed mail host crashed the gateway into a respawn loop instead of reporting an
unreachable mailbox.

These tests pin the fix: the probes run off the loop under a deadline, and the typed SMTP
authentication failure still short-circuits the reconnect queue.
"""

import asyncio
import os
import smtplib
import time
import unittest
from unittest.mock import patch

from gateway.config import PlatformConfig

_HANG = 0.6  # long enough for many heartbeat turns on a live loop, far short of the deadline


def _make_adapter(**env):
    """An adapter over placeholder credentials — no test here opens a real socket."""
    values = {
        "EMAIL_ADDRESS": "hermes@example.com",
        "EMAIL_PASSWORD": "placeholder",
        "EMAIL_IMAP_HOST": "imap.example.com",
        "EMAIL_SMTP_HOST": "smtp.example.com",
    }
    values.update(env)
    with patch.dict(os.environ, values):
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True))


class _Heartbeat:
    """Counts event-loop turns while the coroutine under test runs."""

    def __init__(self, interval=0.02):
        self.interval = interval
        self.ticks = 0
        self._task = None

    async def _beat(self):
        while True:
            await asyncio.sleep(self.interval)
            self.ticks += 1

    def start(self):
        self._task = asyncio.ensure_future(self._beat())
        return self

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None


class _FakeSMTP:
    """Enough of an ``smtplib.SMTP`` for ``_probe_smtp``'s login/quit path."""

    def __init__(self, login_error=None):
        self._login_error = login_error

    def login(self, *_args, **_kwargs):
        if self._login_error is not None:
            raise self._login_error
        return 235, b"ok"

    def quit(self):
        return 221, b"bye"


def _blackholed(delay):
    """``_connect_smtp`` stand-in for a host that swallows the connection, then times out."""

    def connect(*_args, **_kwargs):
        time.sleep(delay)
        raise TimeoutError("timed out")

    return connect


class TestConnectProbesOffLoop(unittest.TestCase):
    def test_connect_keeps_the_loop_alive_across_a_hung_probe(self):
        adapter = _make_adapter()
        adapter._connect_probe_timeout = 30
        adapter._probe_imap = lambda _is_reconnect: True
        adapter._connect_smtp = _blackholed(_HANG)

        async def scenario():
            beat = _Heartbeat().start()
            try:
                return await adapter.connect(), beat
            finally:
                beat.stop()

        result, beat = asyncio.run(scenario())
        self.assertFalse(result)
        self.assertGreater(beat.ticks, 0, "the event loop was parked by the probe")

    def test_the_pre_fix_inline_shape_does_park_the_loop(self):
        """Negative control: the harness can see a parked loop, so the test above means something."""
        adapter = _make_adapter()
        adapter._connect_smtp = _blackholed(_HANG)

        async def scenario():
            beat = _Heartbeat().start()
            try:
                adapter._probe_smtp()  # exactly what connect() used to do, on the loop
            finally:
                beat.stop()
            return beat

        beat = asyncio.run(scenario())
        self.assertEqual(beat.ticks, 0, "control failed: a parked loop looked live")

    def test_a_blackholed_host_is_bounded_by_the_deadline(self):
        adapter = _make_adapter()
        adapter._connect_probe_timeout = 0.25
        adapter._probe_imap = lambda _is_reconnect: True
        adapter._connect_smtp = _blackholed(3)

        async def scenario():
            beat = _Heartbeat().start()
            try:
                started = time.monotonic()
                ok = await adapter.connect()
                elapsed = time.monotonic() - started
                # Read the state inside the loop: the abandoned worker thread reports its own
                # (equally retryable) socket error once the socket finally times out.
                return ok, elapsed, beat.ticks, adapter._fatal_error_code, adapter._fatal_error_retryable
            finally:
                beat.stop()

        ok, elapsed, ticks, code, retryable = asyncio.run(scenario())
        self.assertFalse(ok)
        self.assertLess(elapsed, 5, "the deadline did not bound the probe phase")
        self.assertEqual(code, "email_connect_probe_timeout")
        self.assertTrue(retryable)
        self.assertGreater(ticks, 0)

    def test_reconnect_flag_reaches_the_imap_probe(self):
        adapter = _make_adapter()
        adapter._require_authenticated_sender = False
        seen = []

        def imap_probe(is_reconnect):
            seen.append(is_reconnect)
            return True

        adapter._probe_imap = imap_probe
        adapter._connect_smtp = lambda *_a, **_k: _FakeSMTP()

        async def scenario():
            ok = await adapter.connect(is_reconnect=True)
            await adapter.disconnect()
            return ok

        self.assertTrue(asyncio.run(scenario()))
        self.assertEqual(seen, [True])

    def test_typed_smtp_auth_failure_stays_fatal_and_non_retryable(self):
        adapter = _make_adapter()
        adapter._connect_smtp = lambda *_a, **_k: _FakeSMTP(
            smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")
        )
        self.assertFalse(adapter._probe_smtp())
        self.assertEqual(adapter._fatal_error_code, "email_auth_error")
        self.assertFalse(adapter._fatal_error_retryable)


if __name__ == "__main__":
    unittest.main()
