"""Regression tests for the Telegram polling health-confirmation log (#90504).

Both reconnect paths end on ``health pending getUpdates progress`` and
``_record_polling_progress`` used to complete silently, so the log stream for
"reconnected and healthy" was byte-identical to "reconnected and hung". The
first confirmed getUpdates round-trip of each generation now emits an INFO
line, turning the pending line into a resolvable pair whose *absence* after a
reconnect is a reliable hung-poll signature.
"""

import asyncio
import logging
from gateway.config import Platform  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _bare_adapter():
    a = TelegramAdapter.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._polling_teardown_started = False
    a._polling_progress_accepting = True
    a._polling_generation = 1
    a._polling_progress_event = asyncio.Event()
    a._polling_network_error_count = 2
    a._polling_conflict_count = 3
    a._polling_conflict_recovery_generation = None
    a._send_path_degraded = True
    return a


class TestPollingHealthConfirmation:




    def test_stale_generation_progress_stays_silent(self, caplog):
        """Progress from an abandoned generation must neither log nor set the
        current event (pre-existing guard, pinned here because the log line
        must inherit the same generation-scoping)."""
        a = _bare_adapter()
        a._polling_generation = 2
        a._polling_progress_event = asyncio.Event()
        with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
            a._record_polling_progress(1)
        assert not [rec for rec in caplog.records if rec.levelno == logging.INFO]
        assert not a._polling_progress_event.is_set()
