"""The wake-word auto-resume watchdog must give up after bounded reopen failures (#131177).

A microphone hot-plug leaves PortAudio's startup device snapshot stale, so every
``resume_listening`` reopen fails until the process restarts. Before the fix the
watchdog swallowed each failure at debug level and retried every ~0.8s forever,
flooding the log (one reporter session: 122 consecutive failures). These tests pin
the bounded give-up contract plus the transient-failure recovery path.
"""

import logging
import queue
import sys
import time
import types

import pytest

from hermes_cli.cli_voice_mixin import CLIVoiceMixin, _WAKE_RESUME_FAILURE_LIMIT


class _WatchdogHost:
    """Just enough of ``HermesCLI`` for ``_start_wake_watchdog`` to run against."""

    def __init__(self) -> None:
        self._wake_word_active = True
        self._wake_suspended = True
        self._should_exit = False
        self._agent_running = False
        self._voice_recording = False
        self._voice_processing = False
        self._pending_input = queue.Queue()
        self._wake_watchdog_started = False


@pytest.fixture
def cli_stub(monkeypatch):
    printed = []
    stub = types.ModuleType("cli")
    stub.logger = logging.getLogger("test-cli")
    stub._DIM = stub._RST = stub._ACCENT = stub._BOLD = ""
    stub._cprint = lambda msg: printed.append(msg)
    monkeypatch.setitem(sys.modules, "cli", stub)
    return printed


def _run_watchdog(monkeypatch, host, resume_side_effect):
    """Start the watchdog with a fake ``resume_listening``; return the resume attempts.

    ``resume_side_effect(attempt)`` runs on the watchdog thread: raise to fail the
    resume, return False to drop the lease, return True to resume. Waits for the
    watchdog to finish via ``_wake_watchdog_started`` (reset in the loop's ``finally``).
    """
    attempts = []

    def fake_resume(*, owner):
        attempts.append(owner)
        return resume_side_effect(len(attempts))

    monkeypatch.setattr("tools.wake_word.resume_listening", fake_resume)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    CLIVoiceMixin._start_wake_watchdog(host)
    deadline = time.monotonic() + 10
    while host._wake_watchdog_started and time.monotonic() < deadline:
        continue  # time.sleep is patched out; spin on the loop's finally flag
    assert host._wake_watchdog_started is False, "watchdog thread did not finish"
    return attempts


class TestWatchdogGivesUpOnPermanentReopenFailure:
    def test_stops_after_bounded_failures(self, monkeypatch, cli_stub):
        host = _WatchdogHost()

        def always_fail(_attempt):
            raise RuntimeError(
                "Error opening InputStream: Internal PortAudio error [PaErrorCode -9986]"
            )

        attempts = _run_watchdog(monkeypatch, host, always_fail)

        assert host._wake_word_active is False
        assert len(attempts) == _WAKE_RESUME_FAILURE_LIMIT, (
            "must not retry past the failure limit"
        )

    def test_announces_the_stop_to_the_user(self, monkeypatch, cli_stub):
        host = _WatchdogHost()

        def always_fail(_attempt):
            raise RuntimeError("boom")

        attempts = _run_watchdog(monkeypatch, host, always_fail)

        assert len(cli_stub) == 1
        assert "Wake word stopped" in cli_stub[0]
        assert "/wake off + /wake on" in cli_stub[0]


class TestWatchdogStillRecoversFromTransientFailures:
    def test_resumes_once_the_microphone_reopens(self, monkeypatch, cli_stub):
        host = _WatchdogHost()

        def fails_twice_then_resumes(attempt):
            if attempt < 3:
                raise RuntimeError("stream contention")
            host._should_exit = True  # success leaves the loop armed; stop the thread
            return True

        attempts = _run_watchdog(monkeypatch, host, fails_twice_then_resumes)

        assert host._wake_word_active is True
        assert host._wake_suspended is False
        assert len(attempts) == 3
        assert cli_stub == [], "no stop message when the resume eventually succeeds"

    def test_dropped_lease_disarms_without_a_message(self, monkeypatch, cli_stub):
        host = _WatchdogHost()
        attempts = _run_watchdog(monkeypatch, host, lambda _a: False)

        assert host._wake_word_active is False
        assert len(attempts) == 1
        assert cli_stub == []
