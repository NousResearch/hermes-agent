"""A long provider wait must survive in the app log, not only on the status line.

The status line is transient (it is rewritten and cleared), so an operator cannot
count long waits or alert on them after the fact. Each silence therefore writes
exactly two records — start and end-with-outcome — and never one per heartbeat.
"""
import logging

from agent.chat_completion_wait_notice import WaitNoticeState

LOGGER_NAME = "agent.chat_completion_wait_notice"


def _waits(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("provider-wait")]


def test_one_silence_logs_exactly_start_and_end_with_outcome(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    state = WaitNoticeState()
    watchdog = ("TTFB", 240.0)
    # 60s..270s of silence: the display is rewritten at most twice, the log once.
    for secs in (60.0, 90.0, 120.0, 150.0, 180.0):
        state.should_emit("first_event", watchdog, model="test-model",
                          provider="test-provider", silence_secs=secs)
    started = _waits(caplog)
    assert len(started) == 1, started
    assert "provider-wait start" in started[0]
    assert "model=test-model" in started[0] and "provider=test-provider" in started[0]
    assert "phase=first_event" in started[0] and "watchdog=TTFB" in started[0]

    state.reset()  # the provider finally answered
    ended = _waits(caplog)
    assert len(ended) == 2, ended
    assert "provider-wait end" in ended[1]
    assert "waited=180s" in ended[1] and "outcome=resumed" in ended[1]


def test_watchdog_kill_is_recorded_as_its_own_outcome_and_next_silence_logs_again(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    state = WaitNoticeState()
    state.should_emit("first_chunk", ("stream stale", 120.0), model="m",
                      provider="p", silence_secs=60.0)
    state.reset(outcome="stale_kill")
    assert "outcome=stale_kill" in _waits(caplog)[-1]

    # A resolved wait must not swallow the next one, and a reset with nothing
    # open must not invent an end record.
    state.reset()
    assert len(_waits(caplog)) == 2
    state.should_emit("first_chunk", None, model="m", provider="p", silence_secs=75.0)
    assert len(_waits(caplog)) == 3 and "provider-wait start" in _waits(caplog)[-1]


def test_identity_is_optional_so_existing_callers_keep_working(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    state = WaitNoticeState()
    assert state.should_emit("first_event", None) is True
    assert state.should_emit("first_event", None) is False
    record = _waits(caplog)[0]
    assert "model=unknown" in record and "provider=unknown" in record
