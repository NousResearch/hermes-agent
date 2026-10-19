"""Late subscribers for ``GET /v1/runs/{id}/events`` (#118138).

The idle sweep expires the transport of a run whose buffer nobody is reading
while the run itself keeps executing. These cases pin what a subscriber that
(re)connects afterwards must see: a re-armed stream for a live run, nothing for
a dead or finished one, and every later event (approval, terminal) on the
transport the subscriber actually holds.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES
from gateway.platforms.api_server_runs import _drop_run_transport
from tests.gateway.api_server_runs_helpers import (
    _create_runs_app,
    _make_adapter,
    _make_slow_agent,
    _read_sse_frame,
)
from tests.waiting import async_wait_until
from tools import approval as approval_mod


@pytest.fixture
def adapter():
    return _make_adapter()


class TestRunEventsLateSubscribe:
    @pytest.mark.asyncio
    async def test_late_subscribe_reattaches_a_live_run(self, adapter):
        """A subscriber that reconnects mid-run must not get run_not_found (#118138).

        The idle sweep expires the transport of a buffer nobody is reading while the run keeps
        executing, and "transport present" used to be the only liveness test — so a reconnect
        mid-run 404'd for the rest of the run even though ``GET /v1/runs/{id}`` still reported
        it running. The reconnect now gets a fresh transport.
        """
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                assert ready.wait(timeout=5), "run never started"

                # What the idle sweep leaves behind: expired transport, still-live run.
                _drop_run_transport(adapter, run_id)
                assert run_id not in adapter._run_streams

                reconnected = await cli.get(f"/v1/runs/{run_id}/events")
                assert reconnected.status == 200, "live run refused its late subscriber"
                assert run_id in adapter._run_streams
                await reconnected.release()

                await cli.post(f"/v1/runs/{run_id}/stop")
                assert interrupted.wait(timeout=5)

    @pytest.mark.asyncio
    async def test_stale_non_terminal_status_without_a_task_does_not_reattach(self, adapter):
        """Execution authority is the live task, not the in-memory status row (#118138 review).

        A non-terminal status whose executor is gone must 404: reattaching there would hand the
        subscriber a transport no producer ever writes to, so it would hang on keepalives instead of
        learning the stream is gone. This pins the boundary the reattach predicate must keep.
        """
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                for _ in range(200):
                    if adapter._run_statuses.get(run_id, {}).get("status") in TERMINAL_STATUSES:
                        break
                    await asyncio.sleep(0.05)

                # Stale row: the executor is gone, the status still says the run is live.
                adapter._active_run_tasks.pop(run_id, None)
                adapter._run_statuses[run_id]["status"] = "running"
                _drop_run_transport(adapter, run_id)

                stale = await cli.get(f"/v1/runs/{run_id}/events")
                assert stale.status == 404
                assert (await stale.json())["error"]["code"] == "run_not_found"
                assert run_id not in adapter._run_streams, "no transport may be handed to a dead run"

    @pytest.mark.asyncio
    async def test_events_after_terminal_status_still_404s(self, adapter):
        """Re-attaching must not resurrect a finished run (#118138)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                for _ in range(200):
                    if adapter._run_statuses.get(run_id, {}).get("status") in TERMINAL_STATUSES:
                        break
                    await asyncio.sleep(0.05)

                _drop_run_transport(adapter, run_id)
                stale = await cli.get(f"/v1/runs/{run_id}/events")
                assert stale.status == 404
                assert (await stale.json())["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_approval_after_swept_transport_replays_to_a_late_subscriber(self, adapter):
        """A pending approval must survive the sweep and replay to a late subscriber (#118138 review).

        The natural client pattern polls status until ``waiting_for_approval`` and only then
        subscribes. If the idle sweep expired the transport while the run kept executing, the
        approval bridge used to enqueue into the dropped buffer: the subscriber got a live but
        empty stream and never saw the question. The bridge now resolves the run's current
        transport and re-arms one when the sweep left none, so the question stays replayable.
        """
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                assert ready.wait(timeout=5), "run never started"

                # The bridge the run registered; _run_agent_sync registers it before the turn.
                for _ in range(100):
                    if run_id in approval_mod._gateway_notify_cbs:
                        break
                    await asyncio.sleep(0.05)
                notify = approval_mod._gateway_notify_cbs[run_id]

                # What the idle sweep leaves behind: expired transport, still-live run.
                _drop_run_transport(adapter, run_id)
                assert run_id not in adapter._run_streams

                notify({
                    "command": "rm -rf /tmp/example",
                    "pattern_keys": ["rm-rf"],
                    "description": "approval after swept transport",
                    "allow_session": True,
                    "allow_permanent": False,
                })
                await async_wait_until(
                    lambda: run_id in adapter._run_streams,
                    timeout=5,
                    what="the pending approval re-arms the swept transport",
                )

                assert run_id in adapter._run_streams, "the pending question must re-arm the transport"
                late = await cli.get(f"/v1/runs/{run_id}/events")
                assert late.status == 200
                sequence, event = await asyncio.wait_for(_read_sse_frame(late), timeout=5)
                assert event is not None and event.get("event") == "approval.request"
                assert event.get("run_id") == run_id

                await late.release()
                await cli.post(f"/v1/runs/{run_id}/stop")
                assert interrupted.wait(timeout=5)

    @pytest.mark.asyncio
    async def test_approval_event_reaches_a_reattached_subscriber(self, adapter):
        """An approval raised after re-attach must go to the new transport (#118138 review).

        Re-attaching hands the subscriber a fresh queue; the bridge must follow that queue
        instead of the one captured at launch, or the pending question lands in a buffer
        nobody reads.
        """
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                assert ready.wait(timeout=5), "run never started"

                for _ in range(100):
                    if run_id in approval_mod._gateway_notify_cbs:
                        break
                    await asyncio.sleep(0.05)
                notify = approval_mod._gateway_notify_cbs[run_id]

                _drop_run_transport(adapter, run_id)
                subscriber = await cli.get(f"/v1/runs/{run_id}/events")
                assert subscriber.status == 200

                notify({
                    "command": "rm -rf /tmp/example",
                    "pattern_keys": ["rm-rf"],
                    "description": "approval after swept transport",
                    "allow_session": True,
                    "allow_permanent": False,
                })
                sequence, event = await asyncio.wait_for(_read_sse_frame(subscriber), timeout=5)
                assert event is not None and event.get("event") == "approval.request"

                await subscriber.release()
                await cli.post(f"/v1/runs/{run_id}/stop")
                assert interrupted.wait(timeout=5)

    @pytest.mark.asyncio
    async def test_reattached_subscriber_receives_the_terminal_event(self, adapter):
        """The terminal event must follow the run's current transport too (#118138 review).

        ``_RunLaunch.put_event`` wrote through the queue captured at launch; after a re-attach
        the terminal ``run.*`` event (the frame clients use to end the stream) was siloed into
        the dropped buffer, so the subscriber waited on keepalives for a run that had finished.
        """
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                assert ready.wait(timeout=5), "run never started"

                _drop_run_transport(adapter, run_id)
                subscriber = await cli.get(f"/v1/runs/{run_id}/events")
                assert subscriber.status == 200

                await cli.post(f"/v1/runs/{run_id}/stop")
                assert interrupted.wait(timeout=5)

                sequence, event = await asyncio.wait_for(_read_sse_frame(subscriber), timeout=5)
                assert event is not None and str(event.get("event", "")).startswith("run.")
                status = adapter._run_statuses[run_id]["status"]
                assert status in TERMINAL_STATUSES
                assert event.get("event") == f"run.{status}"

                await subscriber.release()
