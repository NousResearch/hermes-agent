"""How long the sweeper keeps a run's replayable event transport."""

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_runs import _RUN_EVENTS_RETENTION, _RunStream


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def test_finished_run_stays_replayable_for_the_retention_window():
    adapter = _adapter()
    run_id = "run_finished"
    stream = adapter._run_streams[run_id] = _RunStream()
    adapter._run_streams_created[run_id] = 1000.0
    stream.put_nowait({"event": "run.completed"})
    stream.put_nowait(None)
    stream.closed_at = 2000.0

    adapter._sweep_orphaned_runs_once(2000.0 + _RUN_EVENTS_RETENTION)
    assert run_id in adapter._run_streams

    adapter._sweep_orphaned_runs_once(2000.0 + _RUN_EVENTS_RETENTION + 1)
    assert run_id not in adapter._run_streams
    assert run_id not in adapter._run_streams_created


def test_orphaned_transport_without_a_live_run_still_expires_after_the_ttl():
    adapter = _adapter()
    run_id = "run_orphaned"
    adapter._run_streams[run_id] = _RunStream()
    adapter._run_streams_created[run_id] = 1000.0

    adapter._sweep_orphaned_runs_once(1000.0 + adapter._RUN_STREAM_TTL + 1)

    assert run_id not in adapter._run_streams
