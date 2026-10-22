"""Research trace events reach the run's durable SSE journal only when the run opted in.

The trace contract spans three seams: the tool layer emits bounded events, the run
launch carries the explicit per-run opt-in, and the run-event callback forwards the
``research.`` namespace into the existing durable journal. The tool layer is pinned
in ``tests/tools/test_research_trace.py``; this file pins the gateway side, which is
what a ``/v1/runs`` client actually observes.
"""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_runs import _RunStream


@pytest.mark.asyncio
async def test_research_events_reach_the_run_journal_and_do_not_move_run_status(api):
    """A research event is journalled for the run, and only the journal sees it.

    The run's pollable status is a separate contract: a research event is trace
    telemetry, not a lifecycle boundary, so it must not rewrite ``last_event``.
    """
    loop = asyncio.get_running_loop()
    run_id = "run_research"
    api._run_streams[run_id] = _RunStream()
    api._run_statuses[run_id] = {"status": "running"}
    callback = api._make_run_event_callback(run_id, loop)

    callback("research.query", provider="fixture", limit=5)
    await asyncio.sleep(0)  # the callback hops onto the loop via call_soon_threadsafe

    _, event = api._run_streams[run_id].backlog[-1]
    assert event["event"] == "research.query"
    assert event["run_id"] == run_id
    assert event["provider"] == "fixture"
    assert event["limit"] == 5
    # Trace telemetry is not a lifecycle boundary: the pollable status is untouched.
    assert api._run_statuses[run_id].get("last_event") is None


@pytest.mark.asyncio
async def test_only_an_explicitly_opted_in_run_carries_a_research_trace(api, owner, monkeypatch):
    """The per-run opt-in is what turns the trace on; nothing else does.

    The launch is the seam: a run that did not ask for a trace must carry
    ``research_trace_enabled`` false, so the executor installs no sink and the
    journal stays free of research events.
    """
    from gateway.platforms import api_server_runs

    launches = []

    async def capture(_self, run, **kwargs):
        launches.append(run)

    monkeypatch.setattr(api_server_runs, '_execute_run', capture)
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    async with TestClient(TestServer(app)) as client:
        accepted = await client.post(
            '/v1/runs', json={'input': 'hi', 'model': 'hermes-agent', 'research_trace': True})
        assert accepted.status == 202, await accepted.text()
        plain = await client.post('/v1/runs', json={'input': 'hi', 'model': 'hermes-agent'})
        assert plain.status == 202, await plain.text()
    await asyncio.sleep(0)

    assert len(launches) == 2, launches
    opted_in, not_opted_in = launches
    assert opted_in.research_trace_enabled is True
    assert not_opted_in.research_trace_enabled is False
