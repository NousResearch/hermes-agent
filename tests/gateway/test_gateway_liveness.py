import asyncio
import threading
from types import SimpleNamespace

from gateway.run import GatewayRunner


def _runner_for_liveness() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._running_agents = {}
    runner._liveness_lock = threading.Lock()
    runner._liveness_stop = threading.Event()
    runner._liveness_thread = None
    runner._liveness_interval = 15.0
    runner._liveness_forward_progress_counter = 0
    return runner


def test_liveness_snapshot_does_not_advance_forward_progress_without_activity(monkeypatch):
    runner = _runner_for_liveness()
    agent = SimpleNamespace(
        get_activity_summary=lambda: {
            "api_call_count": 1,
            "provider_chunk_count": 0,
            "tool_transition_count": 0,
            "tool_completion_count": 0,
            "api_completion_count": 0,
            "current_tool": None,
            "last_activity_desc": "waiting",
        }
    )
    runner._running_agents = {"telegram:chat:user": agent}

    monotonic_values = iter([100.0, 110.0])
    iso_values = iter([
        "2026-05-16T18:00:00+00:00",
        "2026-05-16T18:00:10+00:00",
    ])
    writes = []

    monkeypatch.setattr("gateway.run.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(runner, "_liveness_now_iso", lambda: next(iso_values))
    monkeypatch.setattr("gateway.status.write_runtime_status", lambda **kwargs: writes.append(kwargs))

    runner._publish_liveness_snapshot()
    runner._publish_liveness_snapshot()

    assert writes[0]["gateway_state"] == "running"
    assert writes[1]["gateway_state"] == "running"
    assert writes[0]["work_active"] is True
    assert writes[1]["work_active"] is True
    assert writes[0]["last_forward_progress_mono"] == 100.0
    assert writes[1]["last_forward_progress_mono"] == 100.0
    assert writes[0]["forward_progress_counter"] == 0
    assert writes[1]["forward_progress_counter"] == 0


def test_liveness_snapshot_advances_forward_progress_when_activity_changes(monkeypatch):
    runner = _runner_for_liveness()
    summaries = iter([
        {
            "api_call_count": 1,
            "provider_chunk_count": 0,
            "tool_transition_count": 0,
            "tool_completion_count": 0,
            "api_completion_count": 0,
            "current_tool": None,
            "last_activity_desc": "api",
        },
        {
            "api_call_count": 2,
            "provider_chunk_count": 0,
            "tool_transition_count": 0,
            "tool_completion_count": 0,
            "api_completion_count": 0,
            "current_tool": None,
            "last_activity_desc": "api",
        },
    ])
    runner._running_agents = {"telegram:chat:user": SimpleNamespace(get_activity_summary=lambda: next(summaries))}

    monotonic_values = iter([100.0, 110.0])
    iso_values = iter([
        "2026-05-16T18:00:00+00:00",
        "2026-05-16T18:00:10+00:00",
    ])
    writes = []

    monkeypatch.setattr("gateway.run.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(runner, "_liveness_now_iso", lambda: next(iso_values))
    monkeypatch.setattr("gateway.status.write_runtime_status", lambda **kwargs: writes.append(kwargs))

    runner._publish_liveness_snapshot()
    runner._publish_liveness_snapshot()

    assert writes[0]["gateway_state"] == "running"
    assert writes[1]["gateway_state"] == "running"
    assert writes[0]["last_forward_progress_mono"] == 100.0
    assert writes[1]["last_forward_progress_mono"] == 110.0
    assert writes[0]["forward_progress_counter"] == 0
    assert writes[1]["forward_progress_counter"] == 1
