"""Host suspend must not masquerade as a silent streaming provider (#121878)."""

from types import SimpleNamespace

from agent import chat_completion_helpers as h
from agent import chat_completion_stream_monitor as monitor


def test_chat_monitor_ignores_sleep_for_stale_and_wait_notice(monkeypatch):
    wall, awake = [1_000_000.0], [1000.0]
    monkeypatch.setattr(monitor.time, "time", lambda: wall[0])
    monkeypatch.setattr(monitor.time, "monotonic", lambda: awake[0])
    call = h._StreamingCall.__new__(h._StreamingCall)
    notices, kills = [], []
    call.agent = SimpleNamespace(
        base_url="https://example.org", _interrupt_requested=False,
        _emit_wait_notice=notices.append, _touch_activity=lambda _: None,
    )
    call.api_kwargs = {"model": "m"}
    call.last_chunk_time = {"t": awake[0] - 456}
    call._stream_stale_timeout = 600
    call._kill_stale_stream = kills.append

    class Done:
        def is_set(self):
            return awake[0] >= 1145

        def wait(self, timeout):
            # First poll is the immediate host resume (six hours of wall time).
            wall[0] += 23000 if awake[0] == 1000 else 1
            awake[0] += 1

    call._call_done = Done()
    call._monitor_loop()
    assert kills == []
    assert any("stream output" in n and "516s" in n for n in notices)
    assert all("234" not in n for n in notices)


def test_bedrock_monitor_ignores_sleep_but_times_out_after_awake_budget(monkeypatch):
    wall, awake = [1_000_000.0], [1000.0]
    monkeypatch.setattr(h.time, "time", lambda: wall[0])
    monkeypatch.setattr(h.time, "monotonic", lambda: awake[0])
    stream = h._BedrockStream.__new__(h._BedrockStream)
    stream.agent = SimpleNamespace(_interrupt_requested=False)
    stream.last_event = 1000.0
    stream.stale_timeout = 3.0
    stream.result = {"error": None, "response": None}
    seen = []
    stream._on_stale = lambda elapsed: seen.append(elapsed)

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return awake[0] < 1005

        def join(self, timeout):
            wall[0] += 23000 if awake[0] == 1000 else 1
            awake[0] += 1

    monkeypatch.setattr(h.threading, "Thread", Worker)
    stream._poll()
    assert len(seen) == 1
    assert 3 < seen[0] < 5
