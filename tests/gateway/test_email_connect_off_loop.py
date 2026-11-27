"""Regression test: the email adapter's blocking IMAP/SMTP probes must not run on the event loop.

``connect()`` probes IMAP and SMTP before it starts polling. Both probes are
synchronous and use stdlib ``imaplib``/``smtplib`` with a 30s socket timeout, so
running them inline blocks the gateway's event loop for as long as the mail
server takes to answer.

A blocked loop takes every other platform down with it: keepalives, websocket
heartbeats and IMAP reads all time out together, and the loop cannot even
process its own timeout expiry (``agent.deadline`` logs "the event loop has not
processed the expiry after a further 5s"). Observed live as email, Slack and
Telegram failing in the same windows, plus ``asyncio: Exception in callback
ClientWebSocketResponse._send_heartbeat()``.

The poll path already offloads correctly (``_check_inbox`` uses
``run_in_executor``); the connect path must match it. These tests assert the
loop stays responsive while a slow probe is in flight.
"""

import asyncio
import os
import time
import unittest
from unittest.mock import patch

# Long enough that a blocked loop cannot tick, short enough to keep the suite fast.
PROBE_DELAY_SECONDS = 0.3
TICK_INTERVAL_SECONDS = 0.01
# 0.6s of probe work at a 10ms tick is ~60 ticks on a responsive loop, ~0 on a wedged one.
MIN_EXPECTED_TICKS = 20


async def _noop_poll_loop():
    """Stand-in for the real poll loop so the test does not spawn IMAP work."""
    return None


def _make_adapter():
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True))


class TestConnectRunsProbesOffLoop(unittest.TestCase):
    """connect() keeps the event loop ticking while the probes block."""

    def _connect_with_slow_probes(self, adapter):
        """Run connect() with probes that block in a thread; return (result, ticks)."""
        ticks = 0

        def slow_imap(is_reconnect):
            time.sleep(PROBE_DELAY_SECONDS)
            return True

        def slow_smtp():
            time.sleep(PROBE_DELAY_SECONDS)
            return True

        async def scenario():
            nonlocal ticks

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(TICK_INTERVAL_SECONDS)
                    ticks += 1

            ticker_task = asyncio.create_task(ticker())
            with patch.object(adapter, "_probe_imap", slow_imap), \
                 patch.object(adapter, "_probe_smtp", slow_smtp), \
                 patch.object(adapter, "_poll_loop", _noop_poll_loop), \
                 patch.object(adapter, "_wire_plugin_handlers", lambda *a, **k: None):
                result = await adapter.connect()
            ticker_task.cancel()
            return result

        result = asyncio.run(scenario())
        return result, ticks

    def test_loop_ticks_while_imap_and_smtp_probes_block(self):
        """A slow IMAP/SMTP probe must not starve the loop."""
        adapter = _make_adapter()

        result, ticks = self._connect_with_slow_probes(adapter)

        self.assertTrue(result)
        self.assertGreaterEqual(
            ticks,
            MIN_EXPECTED_TICKS,
            f"event loop ticked only {ticks} times across "
            f"{PROBE_DELAY_SECONDS * 2}s of probe work — connect() is blocking the loop",
        )

        adapter._running = False
        if adapter._poll_task:
            adapter._poll_task.cancel()

    def test_connect_still_reports_failure_when_imap_probe_fails(self):
        """Offloading must not change the failure contract: a failed IMAP probe means False."""
        adapter = _make_adapter()
        smtp_called = False

        def failing_imap(is_reconnect):
            return False

        def smtp_probe():
            nonlocal smtp_called
            smtp_called = True
            return True

        async def scenario():
            with patch.object(adapter, "_probe_imap", failing_imap), \
                 patch.object(adapter, "_probe_smtp", smtp_probe), \
                 patch.object(adapter, "_poll_loop", _noop_poll_loop), \
                 patch.object(adapter, "_wire_plugin_handlers", lambda *a, **k: None):
                return await adapter.connect()

        result = asyncio.run(scenario())

        self.assertFalse(result)
        self.assertFalse(adapter._running)
        # Original short-circuit is preserved: no SMTP probe once IMAP has failed.
        self.assertFalse(smtp_called, "SMTP probe ran even though the IMAP probe failed")


if __name__ == "__main__":
    unittest.main()
