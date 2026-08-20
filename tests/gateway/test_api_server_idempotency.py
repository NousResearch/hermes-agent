"""
Exactly-once affordances on the /v1/runs API.

A client that submits a run and loses the response must be able to retry
without dispatching a second agent, and must be able to correlate its own
Idempotency-Key back to the run the server already started. Stop must also be
replay-safe: a terminal run keeps its outcome.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _make_adapter


@pytest.fixture
def adapter():
    return _make_adapter()


def _create_runs_app(adapter) -> web.Application:
    """Build an app from the adapter's real route table.

    Using ``_http_route_table()`` rather than a hand-written route list means
    these tests fail if the GET /v1/runs row is ever dropped from the table
    that ``connect()`` actually registers.
    """
    app = web.Application()
    app["api_server_adapter"] = adapter
    wanted = ("/v1/runs", "/v1/capabilities")
    for method, path, handler in adapter._http_route_table():
        if path == "/v1/runs" or path.startswith("/v1/runs/") or path in wanted:
            app.router.add_route(method, path, handler)
    return app


async def _submit(client, key=None, body=None):
    """POST /v1/runs, optionally carrying an Idempotency-Key."""
    headers = {"Idempotency-Key": key} if key else {}
    return await client.post("/v1/runs", json=body or {"input": "hello"}, headers=headers)


async def _settle(adapter):
    """Let each run's background task reach its terminal state."""
    for _ in range(50):
        tasks = list(adapter._active_run_tasks.values())
        if all(t.done() for t in tasks):
            return
        await asyncio.sleep(0.01)


@pytest.fixture
def no_agent(adapter):
    """Make the background run fail immediately.

    The endpoint contract under test is settled synchronously in the handler
    (status record, key mapping, response); the agent itself only needs to
    reach a terminal state without spawning executor threads.
    """
    with patch.object(adapter, "_create_agent", side_effect=RuntimeError("no agent in test")):
        yield


# ---------------------------------------------------------------------------
# POST /v1/runs — replay
# ---------------------------------------------------------------------------


class TestRunSubmissionIdempotency:
    @pytest.mark.asyncio
    async def test_repeat_key_returns_same_run_and_starts_one_run(self, adapter, no_agent):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await _submit(cli, key="key-abc")
            assert first.status == 202
            first_body = await first.json()
            run_id = first_body["run_id"]
            assert first_body["status"] == "queued"

            second = await _submit(cli, key="key-abc")
            assert second.status == 202
            second_body = await second.json()

            # Same run, and no second agent was dispatched.
            assert second_body["run_id"] == run_id
            assert second_body["object"] == "hermes.run"
            assert len(adapter._run_statuses) == 1
            await _settle(adapter)

    @pytest.mark.asyncio
    async def test_distinct_keys_start_distinct_runs(self, adapter, no_agent):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await _submit(cli, key="key-1")
            second = await _submit(cli, key="key-2")
            assert (await first.json())["run_id"] != (await second.json())["run_id"]
            assert len(adapter._run_statuses) == 2
            await _settle(adapter)

    @pytest.mark.asyncio
    async def test_no_key_never_replays(self, adapter, no_agent):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await _submit(cli)
            second = await _submit(cli)
            assert (await first.json())["run_id"] != (await second.json())["run_id"]
            assert adapter._run_idempotency == {}
            await _settle(adapter)

    @pytest.mark.asyncio
    async def test_fresh_submit_returns_full_status_payload(self, adapter, no_agent):
        """Callers validate run identity straight off the submit response."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await _submit(cli, key="key-shape", body={"input": "hi", "session_id": "sess-shape"})
            assert resp.status == 202
            body = await resp.json()

            assert body["object"] == "hermes.run"
            assert body["run_id"].startswith("run_")
            assert body["status"] == "queued"
            assert body["session_id"] == "sess-shape"
            assert body["idempotency_key"] == "key-shape"
            await _settle(adapter)

    @pytest.mark.asyncio
    async def test_replay_is_served_while_the_concurrency_gate_is_closed(self, adapter, no_agent):
        """A saturated server must still answer recovery retries.

        429-ing a replay would let a caller record a terminal rejection for a
        run whose agent is still executing.
        """
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await _submit(cli, key="key-saturated")
            run_id = (await first.json())["run_id"]

            def _at_capacity():
                return web.json_response({"error": "at capacity"}, status=429)

            with patch.object(adapter, "_concurrency_limited_response", side_effect=_at_capacity):
                # A fresh submission is refused ...
                fresh = await _submit(cli)
                assert fresh.status == 429

                # ... but the keyed retry still resolves to its run.
                replay = await _submit(cli, key="key-saturated")
                assert replay.status == 202
                replay_body = await replay.json()
                assert replay_body["run_id"] == run_id
                assert replay_body["idempotency_key"] == "key-saturated"

            assert len(adapter._run_statuses) == 1
            await _settle(adapter)

    @pytest.mark.asyncio
    async def test_status_payload_echoes_idempotency_key(self, adapter, no_agent):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            submitted = await _submit(cli, key="key-echo")
            run_id = (await submitted.json())["run_id"]

            resp = await cli.get(f"/v1/runs/{run_id}")
            assert resp.status == 200
            assert (await resp.json())["idempotency_key"] == "key-echo"

            # The echo survives later status transitions.
            await _settle(adapter)
            resp = await cli.get(f"/v1/runs/{run_id}")
            body = await resp.json()
            assert body["idempotency_key"] == "key-echo"
            assert body["status"] == "failed"

    @pytest.mark.asyncio
    async def test_status_payload_omits_key_when_header_absent(self, adapter, no_agent):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            submitted = await _submit(cli)
            run_id = (await submitted.json())["run_id"]

            resp = await cli.get(f"/v1/runs/{run_id}")
            assert "idempotency_key" not in await resp.json()
            await _settle(adapter)

    @pytest.mark.asyncio
    async def test_invalid_body_is_rejected_before_key_is_recorded(self, adapter, no_agent):
        """A malformed retry still gets its 400 and claims no key."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await _submit(cli, key="key-bad", body={"not_input": "x"})
            assert resp.status == 400
            assert adapter._run_idempotency == {}
            assert adapter._run_statuses == {}


# ---------------------------------------------------------------------------
# GET /v1/runs — correlation lookup
# ---------------------------------------------------------------------------


class TestRunCorrelationLookup:
    @pytest.mark.asyncio
    async def test_lookup_by_header_returns_the_run_status(self, adapter, no_agent):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            run_id = (await (await _submit(cli, key="key-lookup")).json())["run_id"]

            resp = await cli.get("/v1/runs", headers={"Idempotency-Key": "key-lookup"})
            assert resp.status == 200
            body = await resp.json()
            assert body["run_id"] == run_id
            assert body["idempotency_key"] == "key-lookup"

            # Same payload the run_id path serves.
            direct = await (await cli.get(f"/v1/runs/{run_id}")).json()
            assert body == direct
            await _settle(adapter)

    @pytest.mark.asyncio
    async def test_lookup_accepts_query_parameter(self, adapter, no_agent):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            run_id = (await (await _submit(cli, key="key-query")).json())["run_id"]

            resp = await cli.get("/v1/runs", params={"idempotency_key": "key-query"})
            assert resp.status == 200
            assert (await resp.json())["run_id"] == run_id
            await _settle(adapter)

    @pytest.mark.asyncio
    async def test_lookup_unknown_key_is_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs", headers={"Idempotency-Key": "never-seen"})
            assert resp.status == 404
            assert (await resp.json())["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_lookup_missing_key_is_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs")
            assert resp.status == 400
            assert (await resp.json())["error"]["code"] == "missing_idempotency_key"

    @pytest.mark.asyncio
    async def test_lookup_is_404_once_the_status_is_swept(self, adapter, no_agent):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            await _submit(cli, key="key-swept")
            await _settle(adapter)

            # Age the terminal status past its retention window and sweep.
            for status in adapter._run_statuses.values():
                status["updated_at"] = 0.0
            adapter._sweep_orphaned_runs_once()

            assert adapter._run_idempotency == {}
            resp = await cli.get("/v1/runs", headers={"Idempotency-Key": "key-swept"})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_lookup_requires_auth(self):
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs", headers={"Idempotency-Key": "key-x"})
            assert resp.status == 401


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — terminal safety
# ---------------------------------------------------------------------------


class TestStopIdempotency:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
    async def test_stop_on_terminal_run_returns_payload_unchanged(self, adapter, terminal):
        adapter._set_run_status("run_done", terminal, session_id="sess-1", idempotency_key="key-1")
        before = dict(adapter._run_statuses["run_done"])
        # A live agent handle would normally make stop act; the terminal
        # status must win regardless.
        adapter._active_run_agents["run_done"] = MagicMock()

        request = MagicMock()
        request.match_info = {"run_id": "run_done"}
        resp = await adapter._handle_stop_run(request)

        assert resp.status == 200
        assert adapter._run_statuses["run_done"]["status"] == terminal
        assert adapter._run_statuses["run_done"] == before
        assert "run_done" not in adapter._stopping_run_ids

    @pytest.mark.asyncio
    async def test_stop_on_active_run_returns_full_status_payload(self, adapter, monkeypatch):
        from tools.process_registry import process_registry

        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset(), raising=False
        )
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda task_id, baseline, *, source: 0,
            raising=False,
        )

        adapter._set_run_status(
            "run_live", "running", session_id="sess-live", idempotency_key="key-live"
        )
        adapter._active_run_agents["run_live"] = MagicMock()

        request = MagicMock()
        request.match_info = {"run_id": "run_live"}
        resp = await adapter._handle_stop_run(request)

        assert resp.status == 200
        body = json.loads(resp.body)
        # Callers get the session/idempotency echo, not the bare stub.
        assert body["status"] == "stopping"
        assert body["run_id"] == "run_live"
        assert body["session_id"] == "sess-live"
        assert body["idempotency_key"] == "key-live"
        assert adapter._run_statuses["run_live"]["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_on_unknown_run_is_still_404(self, adapter):
        request = MagicMock()
        request.match_info = {"run_id": "run_missing"}
        resp = await adapter._handle_stop_run(request)
        assert resp.status == 404


# ---------------------------------------------------------------------------
# /v1/capabilities
# ---------------------------------------------------------------------------


class TestCapabilitiesAdvertisement:
    @pytest.mark.asyncio
    async def test_capabilities_advertise_exactly_once_affordances(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            data = await (await cli.get("/v1/capabilities")).json()

        assert data["features"]["run_idempotency"] is True
        assert data["features"]["run_correlation_lookup"] is True
        assert data["features"]["run_stop_idempotent"] is True
        assert data["endpoints"]["run_lookup"] == {"method": "GET", "path": "/v1/runs"}
        # Run statuses live in memory only — do not claim durability.
        assert "run_status_durable" not in data["features"]
