"""Deterministic, hermetic audio fixtures for the converse voice suite.

The point of these helpers is HONEST tests: they drive the REAL VAD / endpointer /
capture path with audio whose per-30ms-block RMS envelope mimics real speech (sustained
syllable bursts with inter-syllable dips and word gaps), across a range of volumes — so a
"quiet speech is heard" assertion is proven by the real detector, not by a mock. The
generator was calibrated against the deployed server: ~400 RMS realistic speech trips,
~300 and below (at the room-tone floor) does not, and room noise never false-trips.

No network, no models, no audio devices. STT and the agent turn are the only things a test
mocks; VAD, endpointing, WAV capture and PCM resampling all run for real.
"""
from __future__ import annotations

import queue
from contextlib import contextmanager
from typing import List, Optional

import numpy as np

SAMPLE_RATE = 16000
BLOCK = int(SAMPLE_RATE * 0.03)  # 480 samples / 30 ms

# Reference RMS levels (approx) a real mic produces, for the volume matrix.
RMS_AT_ROOM_FLOOR = 300      # at the noise floor — genuinely indistinguishable, must MISS
RMS_QUIET_SPEECH = 500       # a soft voice — must be HEARD
RMS_NORMAL_SPEECH = 2000     # a normal voice
RMS_LOUD_SPEECH = 5000       # a loud/close voice


def speech_like(target_rms: float, *, seconds: float = 1.6, seed: int = 0,
                rate: int = SAMPLE_RATE) -> np.ndarray:
    """int16 mono PCM whose block-RMS envelope mimics speech at ~*target_rms* overall.

    Broadband noise shaped by a ~4 Hz syllable modulation (sustained bursts, shallow dips)
    plus ~250 ms word gaps every ~1.1 s, scaled so the voiced-portion RMS ≈ *target_rms*.
    Deterministic for a given *seed*. Intelligibility doesn't matter — STT is mocked; only
    the RMS dynamics the VAD sees are modelled.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n)
    syllable = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t / rate)
    gap = np.ones(n)
    period = int(1.1 * rate)
    for start in range(int(0.5 * rate), n, period):
        gap[start:start + int(0.25 * rate)] = 0.05
    env = syllable * gap
    sig = rng.standard_normal(n) * env
    voiced = sig[env > 0.3]
    cur = float(np.sqrt(np.mean(voiced ** 2))) if voiced.size else 1.0
    sig = sig * (float(target_rms) / max(cur, 1e-6))
    return np.clip(sig, -32000, 32000).astype(np.int16)


def room_noise(rms: float, *, seconds: float = 1.8, seed: int = 1,
               rate: int = SAMPLE_RATE) -> np.ndarray:
    """Steady broadband room noise at *rms* — used to prove it does NOT false-trip."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    return np.clip(rng.standard_normal(n) * rms, -32000, 32000).astype(np.int16)


def silence(seconds: float = 1.8, rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.int16)


def calibration(floor_rms: float = 50.0, *, blocks: int = 50, seed: int = 7) -> np.ndarray:
    """Leading quiet-room audio so the detector calibrates a real (low) floor before speech."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.standard_normal(BLOCK * blocks) * floor_rms, -32000, 32000).astype(np.int16)


@contextmanager
def mocked_stt(transcript: str = "heard", *, monkeypatch=None):
    """Patch voice_mode.transcribe_recording to return *transcript* (or a stop phrase, etc.)
    whenever the REAL capture path writes a WAV. Everything up to STT runs for real."""
    from tools import voice_mode as vm
    orig = vm.transcribe_recording
    vm.transcribe_recording = lambda path, model=None: {"success": True, "transcript": transcript}
    try:
        yield
    finally:
        vm.transcribe_recording = orig


def run_utterances(pcm_segments: List[np.ndarray], *, transcript: str = "heard",
                   input_rate: int = SAMPLE_RATE, timeout: float = 6.0,
                   endpoint_silence_blocks: int = 55) -> List[str]:
    """Drive a REAL ConverseSession with the given audio segments and return the transcripts
    it produced. Each segment is a separate utterance; trailing silence after each lets the
    endpointer close it. STT is mocked to *transcript*; the VAD/endpoint/capture are real.

    Returns the list of non-empty transcripts the session emitted (one per detected utterance).
    """
    from tools.voice_converse_loop import ConverseSession

    results: List[str] = []
    with mocked_stt(transcript):
        session = ConverseSession(np, input_rate=input_rate)
        session.start()
        try:
            session.stream.feed(calibration().tobytes())
            for seg in pcm_segments:
                session.stream.feed(seg.tobytes())
                session.stream.feed(silence(seconds=endpoint_silence_blocks * 0.03).tobytes())
            # Collect until quiet: pull transcripts until the queue goes silent for `timeout`.
            deadline_empty = timeout
            while True:
                try:
                    item = session.transcripts.get(timeout=deadline_empty)
                except queue.Empty:
                    break
                if item is None:  # shutdown sentinel
                    break
                if item:
                    results.append(item)
                    deadline_empty = 1.5  # once we've heard one, wait only briefly for more
        finally:
            session.stop()
    return results


def heard(pcm: np.ndarray, **kw) -> bool:
    """True iff the real session produced at least one transcript for *pcm*."""
    return len(run_utterances([pcm], **kw)) >= 1


def collect_session_events(pcm_segments: List[np.ndarray], *, quiet_interval: float,
                           input_rate: int = SAMPLE_RATE, max_wait: float = 8.0) -> list:
    """Drive a real ConverseSession in session mode and return the ordered events it produced —
    str transcripts and QuietTick objects. Stops at the first non-empty transcript (a natural
    end marker) or after *max_wait*. Lets tests assert the received-silence quiet accounting."""
    import queue as _queue
    import time

    from tools.voice_converse_loop import ConverseSession

    events: list = []
    with mocked_stt("heard"):
        session = ConverseSession(np, input_rate=input_rate, quiet_interval=quiet_interval)
        session.start()
        try:
            session.stream.feed(calibration().tobytes())
            for seg in pcm_segments:
                session.stream.feed(seg.tobytes())
            deadline = time.monotonic() + max_wait
            while time.monotonic() < deadline:
                try:
                    item = session.transcripts.get(timeout=0.5)
                except _queue.Empty:
                    continue
                if item is None:
                    break
                events.append(item)
                if isinstance(item, str) and item:
                    break  # a real utterance — natural end of collection
        finally:
            session.stop()
    return events
