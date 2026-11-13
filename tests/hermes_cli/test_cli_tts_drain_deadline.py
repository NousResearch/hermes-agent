"""Streaming-TTS drain: the CLI waits on *inactivity*, not a fixed total budget.

With ``voice.auto_tts`` on, the CLI prints each sentence as the TTS pipeline speaks it, so the
drain wait after a turn decides what the user sees as well as what they hear. A fixed total wait
cut a long reply mid-sentence and hid the rest of the text; a total wait replaced by "as long as
audio keeps flowing" keeps the slow-engine case alive.
"""
import threading

import pytest

from hermes_cli import cli_chat_turn_mixin as mixin


class FakeClock:
    """Monotonic clock that advances a fixed step per read: no sleeps, deterministic deadlines."""

    def __init__(self, step: float = 1.0):
        self.now = 0.0
        self.step = step

    def monotonic(self) -> float:
        self.now += self.step
        return self.now


class FakeWorker(threading.Thread):
    """The TTS worker thread: alive for *polls* checks, then exits (``None`` = never exits)."""

    def __init__(self, polls: int | None = None):
        super().__init__(daemon=True, name="fake-tts-worker")
        self.polls = polls
        self.joins = 0

    def is_alive(self) -> bool:
        if self.polls is None:
            return True
        if self.polls <= 0:
            return False
        self.polls -= 1
        return True

    def join(self, timeout=None) -> None:  # the loop's own slice wait; nothing to wait for
        self.joins += 1


@pytest.fixture
def audio(monkeypatch):
    """Controllable playback state, injected where the drain loop imports it from."""
    from tools import voice_mode

    state = {"playing": False}
    monkeypatch.setattr(voice_mode, "is_audio_output_active", lambda: state["playing"])
    return state


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(mixin, "time", fake)
    return fake


def test_worker_that_exits_is_a_clean_drain(audio, clock):
    audio["playing"] = True
    assert mixin._await_tts_drain(FakeWorker(polls=3)) is True


def test_long_playback_outlives_any_fixed_total_budget(audio, clock):
    """20 minutes of speech: far past the old 120 s join, still a clean drain under the ceiling."""
    audio["playing"] = True
    worker = FakeWorker(polls=1200)
    assert mixin._await_tts_drain(worker) is True
    assert clock.now > 120


def test_silence_after_audio_gives_up_after_the_stall_window(audio, clock, monkeypatch):
    """Audio for a few polls, then nothing: the stall window ends the wait, not the ceiling."""
    from tools import voice_mode

    calls = {"n": 0}

    def plays_then_stops():
        calls["n"] += 1
        return calls["n"] < 5

    monkeypatch.setattr(voice_mode, "is_audio_output_active", plays_then_stops)
    assert mixin._await_tts_drain(FakeWorker(polls=None)) is False
    assert clock.now >= mixin._TTS_STALL_SECONDS
    assert clock.now < mixin._TTS_DRAIN_CEILING_SECONDS


def test_a_silent_engine_is_not_waited_on_forever(audio, clock):
    audio["playing"] = False
    assert mixin._await_tts_drain(FakeWorker(polls=None)) is False
    assert clock.now >= mixin._TTS_NO_SIGNAL_SECONDS


def test_a_trickling_engine_is_bounded_by_the_ceiling(audio, clock):
    """Audio forever, worker never exits: the backstop, not the stall window, ends the wait."""
    audio["playing"] = True
    clock.step = 10.0
    assert mixin._await_tts_drain(FakeWorker(polls=None)) is False
    assert clock.now >= mixin._TTS_DRAIN_CEILING_SECONDS


def test_giving_up_says_why_in_the_log(audio, clock, caplog):
    """The reason is the only clue in a support log; the release path's cut line says nothing."""
    audio["playing"] = False
    with caplog.at_level("INFO", logger=mixin.__name__):
        assert mixin._await_tts_drain(FakeWorker(polls=None)) is False
    assert "Streaming TTS drain gave up" in caplog.text
    assert "playback state never observable" in caplog.text
