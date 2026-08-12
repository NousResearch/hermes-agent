"""Deterministic tests for playback-scoped Discord streaming KWS."""
from __future__ import annotations

import threading
import time
import types

import pytest

from plugins.platforms.discord.streaming_kws import (
    DiscordStreamingKwsManager,
    StreamingKwsConfig,
    _QueueItem,
    _build_engine,
    _normalize,
)


class _FakeEngine:
    def __init__(self, _config, _phrases, *, fire_on=1, block=None):
        self.fire_on = fire_on
        self.block = block
        self.closed = False

    def create_stream(self):
        return {"frames": 0}

    def process(self, stream, _pcm):
        if self.block is not None:
            self.block.wait(timeout=2)
        stream["frames"] += 1
        return 0 if stream["frames"] >= self.fire_on else None

    def close(self):
        self.closed = True


def _assert_running(manager: DiscordStreamingKwsManager) -> None:
    assert manager._ready.wait(timeout=1)
    assert manager.snapshot_stats()["state"] == "RUNNING"


def test_manager_rejects_starting_playback_then_accepts_next_fresh_playback():
    factory_entered = threading.Event()
    release_startup = threading.Event()
    processed = threading.Event()
    detected = threading.Event()
    events = []

    class SignallingEngine(_FakeEngine):
        def process(self, stream, pcm):
            processed.set()
            return super().process(stream, pcm)

    engine = SignallingEngine(None, None)

    def blocking_factory(*_args):
        factory_entered.set()
        assert release_startup.wait(timeout=2)
        return engine

    def callback(event):
        events.append(event)
        detected.set()

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        callback,
        engine_factory=blocking_factory,
    )
    pcm = b"\x00" * 3840
    try:
        assert factory_entered.wait(timeout=1)
        assert manager.begin_playback(1, 10) is False
        assert manager.offer_pcm(1, 10, 42, pcm) is False
        assert manager.end_playback(1, 10) is False
        assert processed.is_set() is False

        release_startup.set()
        _assert_running(manager)

        assert manager.begin_playback(1, 11) is True
        assert manager.offer_pcm(1, 11, 42, pcm) is True
        assert detected.wait(timeout=1)
        assert [event["token"] for event in events] == [11]
    finally:
        release_startup.set()
        manager.close()


def test_manager_startup_failure_is_terminal_drained_and_close_is_idempotent():
    factory_entered = threading.Event()
    release_startup = threading.Event()

    def broken_factory(*_args):
        factory_entered.set()
        assert release_startup.wait(timeout=2)
        raise RuntimeError("model startup failed")

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=broken_factory,
    )
    assert factory_entered.wait(timeout=1)
    assert manager.begin_playback(1, 20) is False
    assert manager.offer_pcm(1, 20, 42, b"pcm") is False
    assert manager.end_playback(1, 20) is False

    # Seed stale internal work to prove the failure transition itself drains
    # both channels rather than relying on STARTING admission rejection alone.
    manager._queue.put_nowait(_QueueItem("stop", 0, 0))
    with manager._forced_end_lock:
        manager._forced_ends.add((1, 20))

    release_startup.set()
    assert manager._ready.wait(timeout=1)
    manager._thread.join(timeout=1)
    assert not manager._thread.is_alive()

    stats = manager.snapshot_stats()
    assert stats["state"] == "FAILED"
    assert stats["startup_failed"] == 1
    assert stats["queue_depth"] == 0
    assert stats["forced_end_depth"] == 0

    for _ in range(3):
        assert manager.begin_playback(1, 21) is False
        assert manager.offer_pcm(1, 21, 42, b"pcm") is False
        assert manager.end_playback(1, 21) is False
        manager.close()
        stats = manager.snapshot_stats()
        assert stats["state"] == "FAILED"
        assert stats["queue_depth"] == 0
        assert stats["forced_end_depth"] == 0


def test_config_is_fail_closed_and_clamped():
    default = StreamingKwsConfig.from_mapping({})
    assert default.enabled is False
    assert default.shadow_only is True
    assert default.provider == "faster_whisper"

    configured = StreamingKwsConfig.from_mapping(
        {
            "enabled": "yes",
            "shadow_only": "false",
            "provider": " Whisper ",
            "model_dir": " ~/models/ko ",
            "hotword_bias": "yes",
            "contrast_wake_names": [" 유나야 ", "미나야", "유나야"],
            "num_threads": 0,
            "queue_frames": 2,
        }
    )
    assert configured.enabled is True
    assert configured.shadow_only is False
    assert configured.provider == "whisper"
    assert configured.model_dir == "~/models/ko"
    assert configured.hotword_bias is True
    assert configured.contrast_wake_names == ("유나야", "미나야")
    assert configured.num_threads == 1
    assert configured.queue_frames == 32

    invalid = StreamingKwsConfig.from_mapping(
        {
            "window_ms": "bad",
            "stride_ms": None,
            "min_audio_ms": object(),
            "num_threads": "bad",
            "queue_frames": "bad",
            "contrast_wake_names": 123,
        }
    )
    assert invalid.window_ms == 1600
    assert invalid.stride_ms == 320
    assert invalid.min_audio_ms == 640
    assert invalid.num_threads == 4
    assert invalid.queue_frames == 256
    assert invalid.contrast_wake_names == ()

    single_name = StreamingKwsConfig.from_mapping(
        {"contrast_wake_names": "유나야"}
    )
    assert single_name.contrast_wake_names == ("유나야",)


def test_normalization_is_exact_except_for_known_korean_contraction():
    assert _normalize(" 하나야, 잠깐! ") == "하나야잠깐"
    assert _normalize("하나야 멈추어") == "하나야멈춰"


def test_unknown_provider_fails_closed_before_loading_model():
    with pytest.raises(ValueError, match="Unsupported"):
        _build_engine(
            StreamingKwsConfig(provider="unknown"),
            ("하나야 잠깐",),
        )


def test_shared_faster_whisper_dependency_initialization_is_serialized(monkeypatch):
    from tools.transcription_tools import ensure_faster_whisper_dependency

    state_lock = threading.Lock()
    active = 0
    max_active = 0
    start = threading.Barrier(3)

    def fake_ensure(*_args, **_kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1

    monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)

    def initialize():
        start.wait(timeout=1)
        ensure_faster_whisper_dependency()

    threads = [threading.Thread(target=initialize) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait(timeout=1)
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert max_active == 1


def test_faster_whisper_engine_downsamples_and_detects_rolling_window(
    monkeypatch,
):
    import sys

    pytest.importorskip("numpy")
    from plugins.platforms.discord.streaming_kws import FasterWhisperRollingEngine

    calls = []
    transcript = ["하나야 잠깐"]

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def transcribe(self, audio, **kwargs):
            calls.append((len(audio), kwargs))
            return iter([types.SimpleNamespace(text=transcript[0])]), object()

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *args, **kwargs: None)
    engine = FasterWhisperRollingEngine(
        StreamingKwsConfig(
            enabled=True,
            provider="faster_whisper",
            model="base",
            window_ms=1600,
            stride_ms=400,
            min_audio_ms=800,
            hotword_bias=True,
            contrast_wake_names=("유나야", "미나야"),
        ),
        ("하나야 잠깐", "하나야 멈춰"),
    )
    stream = engine.create_stream()
    assert engine.process(stream, b"") is None
    assert engine.flush(stream) is None
    # 20 ms of 48 kHz stereo int16 per frame; 40 frames = 800 ms.
    frame = b"\x10\x00\xf0\xff" * 960
    detected = None
    for _ in range(40):
        detected = engine.process(stream, frame)
    assert detected == 0
    assert calls[0][1]["compute_type"] == "int8"
    assert calls[-1][0] == 12800
    assert calls[-1][1]["language"] == "ko"
    assert calls[-1][1]["beam_size"] == 1
    assert "하나야 잠깐" in calls[-1][1]["hotwords"]
    assert "유나야 잠깐" in calls[-1][1]["hotwords"]
    assert "미나야 멈춰" in calls[-1][1]["hotwords"]

    transcript[0] = "유나야 잠깐"
    negative_stream = engine.create_stream()
    negative = None
    for _ in range(40):
        negative = engine.process(negative_stream, frame)
    assert negative is None
    assert engine.flush(negative_stream) is None

    transcript[0] = "정하나야 잠깐"
    prefixed_name_stream = engine.create_stream()
    prefixed_match = None
    for _ in range(40):
        prefixed_match = engine.process(prefixed_name_stream, frame)
    assert prefixed_match is None
    engine.close()

    with pytest.raises(RuntimeError, match="at least one phrase"):
        FasterWhisperRollingEngine(
            StreamingKwsConfig(enabled=True),
            (),
        )


def test_faster_whisper_matcher_requires_lexical_boundary(monkeypatch):
    import sys

    pytest.importorskip("numpy")
    from plugins.platforms.discord.streaming_kws import FasterWhisperRollingEngine

    transcript = ["하나야 잠깐만"]

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, _audio, **_kwargs):
            return iter([types.SimpleNamespace(text=transcript[0])]), object()

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *args, **kwargs: None)
    engine = FasterWhisperRollingEngine(
        StreamingKwsConfig(
            enabled=True,
            window_ms=800,
            stride_ms=160,
            min_audio_ms=400,
        ),
        ("하나야 잠깐",),
    )
    frame = b"\x10\x00\xf0\xff" * 960

    longer_token = engine.create_stream()
    result = None
    for _ in range(20):
        result = engine.process(longer_token, frame)
    assert result is None

    transcript[0] = "하나야 잠깐, 들어줘"
    bounded_phrase = engine.create_stream()
    result = None
    for _ in range(20):
        result = engine.process(bounded_phrase, frame)
    assert result == 0
    engine.close()


def test_manager_fires_once_per_playback_and_resets_for_next_token():
    events = []
    fired = threading.Event()

    def callback(event):
        events.append(event)
        fired.set()

    engine = _FakeEngine(None, None, fire_on=2)
    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        callback,
        engine_factory=lambda *_args: engine,
    )
    pcm = b"\x00" * 3840
    try:
        assert manager.begin_playback(1, 10)
        assert manager.offer_pcm(1, 10, 42, pcm, received_at=time.monotonic())
        assert manager.offer_pcm(1, 10, 42, pcm, received_at=time.monotonic())
        assert fired.wait(timeout=1)
        for _ in range(5):
            manager.offer_pcm(1, 10, 42, pcm)
        time.sleep(0.05)
        assert len(events) == 1
        assert events[0]["token"] == 10
        assert events[0]["user_id"] == 42
        assert events[0]["keyword_index"] == 0
        assert events[0]["audio_ms"] == 40

        fired.clear()
        assert manager.end_playback(1, 10)
        assert manager.begin_playback(1, 11)
        assert manager.offer_pcm(1, 11, 42, pcm)
        assert manager.offer_pcm(1, 11, 42, pcm)
        assert fired.wait(timeout=1)
        assert [event["token"] for event in events] == [10, 11]
    finally:
        manager.close()
    assert engine.closed is True


def test_manager_ignores_unknown_user_and_stale_token():
    events = []
    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        events.append,
        engine_factory=lambda *_args: _FakeEngine(None, None),
    )
    try:
        manager.begin_playback(1, 5)
        assert manager.offer_pcm(1, 5, 0, b"x") is False
        assert manager.offer_pcm(1, 4, 42, b"\x00" * 3840)
        time.sleep(0.05)
        assert events == []
        manager.end_playback(1, 5)
        assert manager.offer_pcm(1, 5, 42, b"\x00" * 3840)
        time.sleep(0.05)
        assert events == []
    finally:
        manager.close()


def test_manager_queue_is_bounded_and_reports_drops():
    release = threading.Event()
    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=lambda *_args: _FakeEngine(None, None, fire_on=999, block=release),
    )
    pcm = b"\x00" * 3840
    try:
        manager.begin_playback(1, 9)
        for _ in range(128):
            manager.offer_pcm(1, 9, 42, pcm)
        assert manager.snapshot_stats()["queue_drops"] > 0
        started = time.monotonic()
        assert manager.end_playback(1, 9) is False
        assert time.monotonic() - started < 0.05
    finally:
        release.set()
        manager.close()


def test_close_discards_saturated_queue_after_blocked_inference():
    entered = threading.Event()
    release = threading.Event()
    close_returned = threading.Event()
    process_calls = []

    class BlockingEngine(_FakeEngine):
        def process(self, stream, _pcm):
            process_calls.append(1)
            entered.set()
            assert release.wait(timeout=5)
            stream["frames"] += 1
            return 0

    engine = BlockingEngine(None, None)
    events = []
    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        events.append,
        engine_factory=lambda *_args: engine,
    )
    pcm = b"\x00" * 3840
    assert manager.begin_playback(1, 41)
    assert manager.offer_pcm(1, 41, 42, pcm)
    assert entered.wait(timeout=1)
    for _ in range(128):
        manager.offer_pcm(1, 41, 42, pcm)
    assert manager.snapshot_stats()["queue_drops"] > 0

    closer = threading.Thread(
        target=lambda: (manager.close(), close_returned.set()),
        daemon=True,
    )
    closer.start()
    try:
        assert close_returned.wait(timeout=0.5), "close waited on blocked inference"
        blocked_stats = manager.snapshot_stats()
        assert blocked_stats["state"] == "CLOSING"
        assert blocked_stats["queue_depth"] == 0
        assert blocked_stats["forced_end_depth"] == 0
        assert manager.offer_pcm(1, 41, 42, pcm) is False
    finally:
        release.set()
        closer.join(timeout=2)

    manager._thread.join(timeout=1)
    assert not manager._thread.is_alive()
    assert len(process_calls) == 1
    assert events == []
    assert engine.closed is True
    final_stats = manager.snapshot_stats()
    assert final_stats["state"] == "CLOSED"
    assert final_stats["queue_depth"] == 0

    repeated_close_returned = threading.Event()
    repeated_closer = threading.Thread(
        target=lambda: (manager.close(), repeated_close_returned.set())
    )
    repeated_closer.start()
    assert repeated_close_returned.wait(timeout=0.5)
    repeated_closer.join(timeout=1)
    assert not repeated_closer.is_alive()


def test_close_serializes_with_forced_end_side_channel_and_drains_it():
    inference_entered = threading.Event()
    release_inference = threading.Event()
    forced_add_entered = threading.Event()
    release_forced_add = threading.Event()
    close_attempted = threading.Event()
    close_returned = threading.Event()

    class BlockingEngine(_FakeEngine):
        def process(self, stream, _pcm):
            inference_entered.set()
            assert release_inference.wait(timeout=5)
            stream["frames"] += 1
            return None

    class PausingForcedEnds(set):
        def add(self, item):
            forced_add_entered.set()
            assert release_forced_add.wait(timeout=2)
            super().add(item)

    class ProbeLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "forced-close":
                close_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            self._lock.release()

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=lambda *_args: BlockingEngine(None, None),
    )
    _assert_running(manager)
    pcm = b"\x00" * 3840
    assert manager.begin_playback(1, 72)
    assert manager.offer_pcm(1, 72, 42, pcm)
    assert inference_entered.wait(timeout=1)
    while manager.offer_pcm(1, 72, 42, pcm):
        pass

    with manager._forced_end_lock:
        manager._forced_ends = PausingForcedEnds()
    manager._lifecycle_lock = ProbeLock()
    end_results = []

    ending = threading.Thread(
        target=lambda: end_results.append(manager.end_playback(1, 72)),
        name="forced-end",
    )
    closer = threading.Thread(
        target=lambda: (manager.close(), close_returned.set()),
        name="forced-close",
    )
    ending.start()
    assert forced_add_entered.wait(timeout=1)
    closer.start()
    assert close_attempted.wait(timeout=1)
    release_forced_add.set()
    ending.join(timeout=1)
    assert not ending.is_alive()
    assert end_results == [False]
    assert close_returned.wait(timeout=1)
    closer.join(timeout=1)
    assert not closer.is_alive()

    blocked_stats = manager.snapshot_stats()
    assert blocked_stats["state"] == "CLOSING"
    assert blocked_stats["queue_depth"] == 0
    assert blocked_stats["forced_end_depth"] == 0

    release_inference.set()
    manager._thread.join(timeout=1)
    assert not manager._thread.is_alive()
    assert manager.snapshot_stats()["state"] == "CLOSED"


def test_detection_queued_before_terminal_transition_is_suppressed():
    inference_entered = threading.Event()
    release_inference = threading.Event()
    inference_returned = threading.Event()
    worker_at_callback_gate = threading.Event()
    release_worker = threading.Event()
    callback_called = threading.Event()
    close_returned = threading.Event()

    class DetectingEngine(_FakeEngine):
        def process(self, stream, _pcm):
            inference_entered.set()
            assert release_inference.wait(timeout=2)
            stream["frames"] += 1
            inference_returned.set()
            return 0

    class CallbackGateLock:
        def __init__(self):
            self._lock = threading.RLock()

        def __enter__(self):
            if threading.current_thread().name == "discord-streaming-kws":
                worker_at_callback_gate.set()
                assert release_worker.wait(timeout=2)
            self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            self._lock.release()

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        lambda _event: callback_called.set(),
        engine_factory=lambda *_args: DetectingEngine(None, None),
    )
    _assert_running(manager)
    assert manager.begin_playback(1, 73)
    assert manager.offer_pcm(1, 73, 42, b"\x00" * 3840)
    assert inference_entered.wait(timeout=1)
    manager._lifecycle_lock = CallbackGateLock()

    release_inference.set()
    try:
        assert inference_returned.wait(timeout=1)
        assert worker_at_callback_gate.wait(timeout=1)
        assert callback_called.is_set() is False

        closer = threading.Thread(
            target=lambda: (manager.close(), close_returned.set()),
            name="callback-close",
        )
        closer.start()
        assert close_returned.wait(timeout=1)
        closer.join(timeout=1)
        assert not closer.is_alive()
        assert manager.snapshot_stats()["state"] == "CLOSING"
    finally:
        release_worker.set()

    manager._thread.join(timeout=1)
    assert not manager._thread.is_alive()
    assert callback_called.is_set() is False
    assert manager.snapshot_stats()["state"] == "CLOSED"


def test_close_is_linearizable_with_paused_pcm_admission_in_both_orders():
    class ProbeLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.offer_attempted = threading.Event()
            self.close_attempted = threading.Event()

        def __enter__(self):
            if threading.current_thread().name.startswith("offer-"):
                self.offer_attempted.set()
            elif threading.current_thread().name.startswith("close-"):
                self.close_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            self._lock.release()

    def make_manager():
        manager = DiscordStreamingKwsManager(
            StreamingKwsConfig(enabled=True, queue_frames=32),
            ("하나야 잠깐",),
            lambda _event: None,
            engine_factory=lambda *_args: _FakeEngine(None, None, fire_on=999),
        )
        _assert_running(manager)
        return manager

    # Admission owns the lifecycle lock first: it may commit, then close must
    # drain it before returning.
    admitted_first = make_manager()
    admitted_lock = ProbeLock()
    admitted_first._lifecycle_lock = admitted_lock
    original_put = admitted_first._queue.put_nowait
    offer_at_enqueue = threading.Event()
    release_offer = threading.Event()
    admitted_results = []

    def blocking_put(item):
        if item.kind == "pcm":
            offer_at_enqueue.set()
            assert release_offer.wait(timeout=2)
        return original_put(item)

    admitted_first._queue.put_nowait = blocking_put
    offer = threading.Thread(
        target=lambda: admitted_results.append(
            admitted_first.offer_pcm(1, 70, 42, b"\x00" * 3840)
        ),
        name="offer-first",
    )
    closer = threading.Thread(target=admitted_first.close, name="close-second")
    offer.start()
    assert offer_at_enqueue.wait(timeout=1)
    closer.start()
    assert admitted_lock.close_attempted.wait(timeout=1)
    release_offer.set()
    offer.join(timeout=1)
    closer.join(timeout=1)
    assert not offer.is_alive()
    assert not closer.is_alive()
    assert admitted_results == [True]
    admitted_stats = admitted_first.snapshot_stats()
    assert admitted_stats["queue_depth"] == 0
    assert admitted_stats["forced_end_depth"] == 0

    # Close owns the lifecycle lock first: a concurrent offer cannot commit
    # after the terminal transition.
    closed_first = make_manager()
    closed_lock = ProbeLock()
    closed_first._lifecycle_lock = closed_lock
    original_discard = closed_first._discard_queue
    close_at_drain = threading.Event()
    release_close = threading.Event()
    closed_results = []

    def blocking_discard():
        close_at_drain.set()
        assert release_close.wait(timeout=2)
        original_discard()

    closed_first._discard_queue = blocking_discard
    closer = threading.Thread(target=closed_first.close, name="close-first")
    offer = threading.Thread(
        target=lambda: closed_results.append(
            closed_first.offer_pcm(1, 71, 42, b"\x00" * 3840)
        ),
        name="offer-second",
    )
    closer.start()
    assert close_at_drain.wait(timeout=1)
    offer.start()
    assert closed_lock.offer_attempted.wait(timeout=1)
    assert closed_results == []
    release_close.set()
    closer.join(timeout=1)
    offer.join(timeout=1)
    assert not closer.is_alive()
    assert not offer.is_alive()
    assert closed_results == [False]
    closed_stats = closed_first.snapshot_stats()
    assert closed_stats["queue_depth"] == 0
    assert closed_stats["forced_end_depth"] == 0


def test_manager_idle_flush_can_detect_final_short_phrase():
    events = []
    fired = threading.Event()

    class FlushEngine(_FakeEngine):
        def process(self, stream, _pcm):
            stream["frames"] += 1
            return None

        def flush(self, stream):
            return 0 if stream["frames"] else None

    def callback(event):
        events.append(event)
        fired.set()

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=32),
        ("하나야 잠깐",),
        callback,
        engine_factory=lambda *_args: FlushEngine(None, None),
    )
    try:
        manager.begin_playback(1, 20)
        manager.offer_pcm(1, 20, 42, b"\x00" * 3840)
        assert fired.wait(timeout=1)
        assert len(events) == 1
        assert events[0]["token"] == 20
    finally:
        manager.close()


def test_manager_default_startup_is_non_blocking():
    release = threading.Event()

    def slow_factory(*_args):
        assert release.wait(timeout=2)
        return _FakeEngine(None, None)

    started = time.monotonic()
    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=slow_factory,
    )
    try:
        assert time.monotonic() - started < 0.2
        assert manager.snapshot_stats()["ready"] == 0
        release.set()
        assert manager._ready.wait(timeout=1)
        assert manager.snapshot_stats()["startup_failed"] == 0
    finally:
        release.set()
        manager.close()


def test_manager_synchronous_startup_check_surfaces_bounded_error(caplog):
    def broken_factory(*_args):
        raise RuntimeError("sensitive model path")

    with pytest.raises(RuntimeError, match="failed to start"):
        DiscordStreamingKwsManager(
            StreamingKwsConfig(enabled=True),
            ("하나야 잠깐",),
            lambda _event: None,
            engine_factory=broken_factory,
            start_timeout=1,
        )
    assert "type=RuntimeError" in caplog.text
    assert "sensitive model path" not in caplog.text


def test_manager_callback_error_is_bounded_and_worker_survives(caplog):
    caplog.set_level("INFO")
    fired = threading.Event()

    def callback(_event):
        fired.set()
        raise RuntimeError("sensitive callback data")

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True),
        ("하나야 잠깐",),
        callback,
        engine_factory=lambda *_args: _FakeEngine(None, None, fire_on=1),
    )
    try:
        manager.begin_playback(1, 30)
        manager.offer_pcm(1, 30, 42, b"\x00" * 3840)
        assert fired.wait(timeout=1)
        deadline = time.monotonic() + 1
        while (
            (
                manager.snapshot_stats()["worker_errors"] < 1
                or "type=RuntimeError" not in caplog.text
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert manager.snapshot_stats()["worker_errors"] == 1
        assert "type=RuntimeError" in caplog.text
        assert "sensitive callback data" not in caplog.text
    finally:
        manager.close()


def test_closed_manager_rejects_new_audio_and_control():
    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=lambda *_args: _FakeEngine(None, None),
    )
    manager.close()
    manager.close()
    assert manager.begin_playback(1, 40) is False
    assert manager.offer_pcm(1, 40, 42, b"pcm") is False


def test_worker_inference_and_idle_flush_errors_are_bounded(caplog):
    caplog.set_level("INFO")

    class BrokenEngine(_FakeEngine):
        def process(self, stream, _pcm):
            stream["frames"] += 1
            raise RuntimeError("sensitive inference data")

        def flush(self, _stream):
            raise RuntimeError("sensitive flush data")

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=lambda *_args: BrokenEngine(None, None),
    )
    try:
        manager.begin_playback(1, 50)
        manager.offer_pcm(1, 50, 42, b"\x00" * 3840)
        deadline = time.monotonic() + 1.5
        while (
            (
                manager.snapshot_stats()["worker_errors"] < 2
                or "frame failed type=RuntimeError" not in caplog.text
                or "idle flush failed type=RuntimeError" not in caplog.text
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert manager.snapshot_stats()["worker_errors"] >= 2
        assert "frame failed type=RuntimeError" in caplog.text
        assert "idle flush failed type=RuntimeError" in caplog.text
        assert "sensitive inference data" not in caplog.text
        assert "sensitive flush data" not in caplog.text
    finally:
        manager.close()


def test_close_during_blocked_startup_eventually_closes_engine_and_state():
    factory_entered = threading.Event()
    release_factory = threading.Event()
    engine_holder = []

    def slow_factory(*_args):
        factory_entered.set()
        assert release_factory.wait(timeout=2)
        engine = _FakeEngine(None, None)
        engine_holder.append(engine)
        return engine

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=slow_factory,
    )
    assert factory_entered.wait(timeout=1)

    started = time.monotonic()
    manager.close()
    assert time.monotonic() - started < 0.15
    assert manager.snapshot_stats()["state"] == "CLOSING"
    assert manager.begin_playback(1, 60) is False
    assert manager.offer_pcm(1, 60, 42, b"pcm") is False

    release_factory.set()
    manager._thread.join(timeout=1)
    assert not manager._thread.is_alive()
    assert engine_holder and engine_holder[0].closed is True
    assert manager.snapshot_stats()["state"] == "CLOSED"
    assert manager.snapshot_stats()["queue_depth"] == 0


def test_close_during_blocked_failing_factory_publishes_closed():
    factory_entered = threading.Event()
    release_factory = threading.Event()

    def failing_factory(*_args):
        factory_entered.set()
        assert release_factory.wait(timeout=2)
        raise RuntimeError("private startup body")

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=failing_factory,
    )
    assert factory_entered.wait(timeout=1)

    manager.close()
    assert manager.snapshot_stats()["state"] == "CLOSING"
    release_factory.set()
    manager._thread.join(timeout=1)

    assert not manager._thread.is_alive()
    stats = manager.snapshot_stats()
    assert stats["state"] == "CLOSED"
    assert stats["queue_depth"] == 0
    assert stats["forced_end_depth"] == 0
    assert manager.begin_playback(1, 61) is False
    assert manager.offer_pcm(1, 61, 42, b"pcm") is False


def test_close_is_prompt_while_already_admitted_callback_is_blocked():
    callback_entered = threading.Event()
    release_callback = threading.Event()

    def blocked_callback(_event):
        callback_entered.set()
        assert release_callback.wait(timeout=2)

    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True),
        ("하나야 잠깐",),
        blocked_callback,
        engine_factory=lambda *_args: _FakeEngine(None, None, fire_on=1),
    )
    manager.begin_playback(1, 70)
    manager.offer_pcm(1, 70, 42, b"\x00" * 3840)
    assert callback_entered.wait(timeout=1)

    started = time.monotonic()
    manager.close()
    assert time.monotonic() - started < 0.15
    assert manager.snapshot_stats()["state"] == "CLOSING"
    assert manager.begin_playback(1, 71) is False

    release_callback.set()
    manager._thread.join(timeout=1)
    assert not manager._thread.is_alive()
    assert manager.snapshot_stats()["state"] == "CLOSED"


def test_snapshot_and_admission_share_one_non_inverting_lock_order():
    manager = DiscordStreamingKwsManager(
        StreamingKwsConfig(enabled=True, queue_frames=128),
        ("하나야 잠깐",),
        lambda _event: None,
        engine_factory=lambda *_args: _FakeEngine(None, None, fire_on=100000),
    )
    manager.begin_playback(1, 80)
    errors = []

    def snapshots():
        try:
            for _ in range(500):
                manager.snapshot_stats()
        except BaseException as exc:
            errors.append(exc)

    def offers():
        try:
            for _ in range(500):
                manager.offer_pcm(1, 80, 42, b"\x00" * 384)
        except BaseException as exc:
            errors.append(exc)

    snapshot_thread = threading.Thread(target=snapshots)
    offer_thread = threading.Thread(target=offers)
    snapshot_thread.start()
    offer_thread.start()
    snapshot_thread.join(timeout=1)
    offer_thread.join(timeout=1)

    assert not snapshot_thread.is_alive()
    assert not offer_thread.is_alive()
    assert errors == []
    manager.close()
