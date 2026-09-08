"""Resumed chunks clear only the stream monitor's own silence notice."""
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as h


@pytest.mark.parametrize("local_loading,heartbeat_race", [(False, False), (True, False), (False, True)])
def test_resumed_chunks_clear_wait_without_erasing_local_load(monkeypatch, local_loading, heartbeat_race):
    call = h._StreamingCall.__new__(h._StreamingCall)
    notices, touches = [], []
    now = [1000.0]
    call.agent = SimpleNamespace(
        base_url="http://localhost:1234" if local_loading else "https://example.com",
        _interrupt_requested=False,
        _emit_wait_notice=lambda text: notices.append((now[0], text)),
        _touch_activity=lambda text: touches.append((now[0], text)),
    )
    call.api_kwargs = {"model": "test-model"}
    call.last_chunk_time = {"t": now[0]}
    call._stream_stale_timeout = 180.0
    loading = "Loading local model weights"
    monkeypatch.setattr(h, "_managed_local_load_notice",
                        lambda *args: loading if now[0] >= 1060.6 else None)
    monkeypatch.setattr(h.time, "time", lambda: now[0])

    if heartbeat_race:
        heartbeat = call._heartbeat

        def resume_before_notice(waiting_secs):
            if waiting_secs >= 60:
                now[0] += 0.1
                call.last_chunk_time["t"] = now[0]
            heartbeat(waiting_secs)

        monkeypatch.setattr(call, "_heartbeat", resume_before_notice)

    class Done:
        def is_set(self):
            return now[0] >= 1062.0

        def wait(self, timeout):
            now[0] = round(now[0] + timeout, 1)
            if not heartbeat_race and now[0] >= 1061.5:
                call.last_chunk_time["t"] = now[0]

    call._call_done = Done()
    call._monitor_loop()
    assert all(t >= 1060.0 for t, _ in notices)
    assert touches[0][0] == 1030.0  # Quiet gateway heartbeat is still 30s.
    assert "waiting on" in notices[0][1]
    if local_loading:
        assert notices[-1][1] == loading
        assert not any(text == "" for _, text in notices)
    else:
        assert notices[1:] == [(1060.4 if heartbeat_race else 1061.5, "")]
