"""Bounded admission queue for run-starting endpoints (gateway.api_server.run_queue_*).

The concurrency cap no longer refuses a request that arrives while the cap is taken: over cap
the request is admitted and QUEUES in FIFO arrival order, and only a queue already at
``run_queue_max_depth`` (429 ``run_queue_full``) or a wait that expires (429
``run_queue_timeout``) is refused. Regression for the fan-out that met "Too many concurrent
runs" at the cap and lost the work.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(max_runs: int = 1, max_depth: int = 100, wait_seconds: float = 120.0):
    """Adapter with the queue knobs pinned (the config resolvers are covered separately)."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._max_concurrent_runs = max_runs
    adapter._run_queue_max_depth = max_depth
    adapter._run_queue_wait_seconds = wait_seconds
    adapter._run_slots_in_use = 0
    return adapter


def _app(adapter: APIServerAdapter, *routes) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    return app


def _runs_app(adapter: APIServerAdapter) -> web.Application:
    return _app(
        adapter,
        ("POST", "/v1/runs", adapter._handle_runs),
        ("GET", "/v1/runs/{run_id}", adapter._handle_get_run),
    )


def _chat_app(adapter: APIServerAdapter) -> web.Application:
    return _app(
        adapter,
        ("POST", "/v1/chat/completions", adapter._handle_chat_completions),
        ("POST", "/v1/responses", adapter._handle_responses),
    )


def _make_agent(run_result=None, side_effect=None) -> MagicMock:
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    if side_effect is not None:
        agent.run_conversation.side_effect = side_effect
    else:
        agent.run_conversation.return_value = run_result or {"final_response": "done"}
    return agent


async def _wait_for(predicate, timeout: float = 3.0) -> None:
    """Spin (never block the loop) until ``predicate`` holds; fail loudly on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


async def _queue_waiter(adapter: APIServerAdapter):
    """Occupy a real FIFO slot wait, so the next admission sees a non-empty queue."""
    waiter = asyncio.create_task(adapter._acquire_run_slot())
    await _wait_for(lambda: len(adapter._run_slot_waiters) == 1)
    return waiter


async def _end_hold(adapter: APIServerAdapter, waiter) -> None:
    """Drop a held waiter; the caller releases the slot it was waiting for."""
    waiter.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass
    await _wait_for(lambda: waiter not in adapter._run_slot_waiters)


# ---------------------------------------------------------------------------
# POST /v1/runs — admit over cap, execute at most the cap
# ---------------------------------------------------------------------------


class TestRunAdmissionQueue:

    @pytest.mark.asyncio
    async def test_run_over_cap_is_admitted_queued_and_runs_after_the_cap_frees(self):
        adapter = _make_adapter(max_runs=1)
        app = _runs_app(adapter)
        calls = []
        release = threading.Event()

        def _run_conversation(**_kwargs):
            calls.append(1)
            release.wait(timeout=5)
            return {"final_response": "done"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=_make_agent(side_effect=_run_conversation)):
                first = await cli.post("/v1/runs", json={"input": "one"})
                first_body = await first.json()
                await _wait_for(lambda: len(calls) == 1)

                second = await cli.post("/v1/runs", json={"input": "two"})
                second_body = await second.json()

                assert first.status == 202 and first_body["status"] == "started"
                assert second.status == 202 and second_body["status"] == "queued"
                assert adapter._run_statuses[second_body["run_id"]]["status"] == "queued"
                assert len(calls) == 1  # still nothing executing but the first
                await _wait_for(lambda: adapter._run_queue_depth() == 1)

                release.set()
                await _wait_for(
                    lambda: adapter._run_statuses[second_body["run_id"]]["status"] == "completed")

        assert len(calls) == 2  # the queued run did run, no work lost
        assert adapter._run_slots_in_use == 0
        assert adapter._run_queue_depth() == 0

    @pytest.mark.asyncio
    async def test_never_more_than_the_cap_executes_at_once(self):
        adapter = _make_adapter(max_runs=2)
        app = _runs_app(adapter)
        live = {"now": 0, "peak": 0}
        gate = threading.Event()

        def _run_conversation(**_kwargs):
            with threading.Lock():
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
            gate.wait(timeout=5)
            with threading.Lock():
                live["now"] -= 1
            return {"final_response": "done"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=_make_agent(side_effect=_run_conversation)):
                bodies = [await (await cli.post("/v1/runs", json={"input": str(i)})).json()
                          for i in range(5)]

                assert sum(1 for b in bodies if b["status"] == "started") == 2
                assert sum(1 for b in bodies if b["status"] == "queued") == 3

                gate.set()
                for body in bodies:
                    await _wait_for(
                        lambda run_id=body["run_id"]: adapter._run_statuses[run_id]["status"] == "completed")

        assert live["peak"] <= 2
        assert adapter._run_slots_in_use == 0

    @pytest.mark.asyncio
    async def test_queue_at_max_depth_refuses_a_new_run(self):
        adapter = _make_adapter(max_runs=1, max_depth=1)
        app = _runs_app(adapter)
        adapter._run_slots_in_use = 1  # a live turn holds the only slot
        async with TestClient(TestServer(app)) as cli:
            waiter = await _queue_waiter(adapter)
            before = dict(adapter._run_statuses)

            resp = await cli.post("/v1/runs", json={"input": "flood"})
            body = await resp.json()

            assert resp.status == 429
            assert body["error"]["code"] == "run_queue_full"
            assert resp.headers.get("Retry-After")
            assert "run_id" not in body
            assert dict(adapter._run_statuses) == before  # refused before any run row exists

            await _end_hold(adapter, waiter)

    @pytest.mark.asyncio
    async def test_fan_out_larger_than_the_cap_all_complete(self):
        """The motivating case: more concurrent callers than the cap lose no work."""
        adapter = _make_adapter(max_runs=3)
        app = _chat_app(adapter)
        live = {"now": 0, "peak": 0}

        async def _slow_agent(**_kwargs):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.02)
            live["now"] -= 1
            return {"final_response": "ok"}, {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", AsyncMock(side_effect=_slow_agent)):
                responses = await asyncio.gather(*[
                    cli.post("/v1/responses", json={"model": "m", "input": f"lane {i}"})
                    for i in range(11)
                ])

        assert [r.status for r in responses] == [200] * 11
        assert live["now"] == 0  # every lane finished
        assert 1 <= live["peak"] <= 3  # and never more than the cap ran at once
        assert adapter._run_slots_in_use == 0
        assert adapter._run_queue_depth() == 0

    @pytest.mark.asyncio
    async def test_slot_returns_when_the_run_fails(self):
        adapter = _make_adapter(max_runs=1)
        app = _runs_app(adapter)

        def _boom(**_kwargs):
            raise RuntimeError("provider blew up")

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=_make_agent(side_effect=_boom)):
                body = await (await cli.post("/v1/runs", json={"input": "one"})).json()
                await _wait_for(
                    lambda: adapter._run_statuses[body["run_id"]]["status"] == "failed")

        assert adapter._run_slots_in_use == 0

    @pytest.mark.asyncio
    async def test_eleven_run_fan_out_at_cap_three_needs_no_retry(self):
        """D1 proof: 11 /v1/runs while the cap is saturated -> 11 run_ids, 0 refusals."""
        adapter = _make_adapter(max_runs=3, wait_seconds=60.0)
        app = _runs_app(adapter)
        live = {"now": 0, "peak": 0}
        lock = threading.Lock()
        gate = threading.Event()
        busy = threading.Event()

        def _run_conversation(**_kwargs):
            with lock:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
                if live["now"] >= 3:
                    busy.set()
            gate.wait(timeout=20)
            with lock:
                live["now"] -= 1
            return {"final_response": "done"}

        agent = MagicMock()
        agent.run_conversation.side_effect = _run_conversation
        agent.session_prompt_tokens = agent.session_completion_tokens = 0
        agent.session_total_tokens = 0

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=agent):
                first = [await cli.post("/v1/runs", json={"input": f"warm {i}"}) for i in range(3)]
                await _wait_for(busy.is_set)  # all three slots are executing
                burst = [await cli.post("/v1/runs", json={"input": f"lane {i}"}) for i in range(11)]
                bodies = [await resp.json() for resp in burst]
                assert adapter._run_slots_in_use <= 3

                gate.set()
                ids = [b["run_id"] for b in bodies]
                for body in bodies:
                    await _wait_for(
                        lambda rid=body["run_id"]:
                            adapter._run_statuses[rid]["status"] == "completed", timeout=20)

        assert [resp.status for resp in burst] == [202] * 11
        assert [resp.status for resp in first] == [202] * 3
        assert all(body.get("run_id") for body in bodies)
        assert all(body["status"] == "queued" for body in bodies)
        assert len(set(ids)) == 11
        assert live["peak"] == 3
        assert adapter._run_slots_in_use == 0
        assert adapter._run_queue_depth() == 0

    @pytest.mark.asyncio
    async def test_zero_depth_bound_never_refuses_on_depth(self):
        adapter = _make_adapter(max_runs=1, max_depth=0)
        app = _runs_app(adapter)
        adapter._run_slots_in_use = 1
        async with TestClient(TestServer(app)) as cli:
            waiter = await _queue_waiter(adapter)
            with patch.object(adapter, "_create_agent", return_value=_make_agent()):
                resp = await cli.post("/v1/runs", json={"input": "queued"})
                body = await resp.json()

                assert resp.status == 202
                assert body["status"] == "queued"

                await _end_hold(adapter, waiter)
                adapter._release_run_slot()  # the held turn ends -> the waiting run starts
                await _wait_for(
                    lambda: adapter._run_statuses[body["run_id"]]["status"] == "completed")

        assert adapter._run_slots_in_use == 0

    @pytest.mark.asyncio
    async def test_disabled_cap_never_queues(self):
        adapter = _make_adapter(max_runs=0)
        app = _runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=_make_agent()):
                resp = await cli.post("/v1/runs", json={"input": "one"})
                body = await resp.json()

                assert resp.status == 202
                assert body["status"] == "started"
                await _wait_for(lambda: adapter._run_statuses[body["run_id"]]["status"] == "completed")

        assert adapter._run_slots_in_use == 0


# ---------------------------------------------------------------------------
# FIFO hand-off and the one-slot invariant
# ---------------------------------------------------------------------------


class TestSlotQueue:

    @pytest.mark.asyncio
    async def test_release_hands_the_slot_to_the_head_waiter_only(self):
        adapter = _make_adapter(max_runs=1)
        adapter._run_slots_in_use = 1  # the turn that is about to release
        order = []
        holds = [asyncio.Event() for _ in range(3)]

        async def _take(index):
            granted = await adapter._acquire_run_slot()
            assert granted is None
            order.append(index)
            await holds[index].wait()
            adapter._release_run_slot()

        tasks = [asyncio.create_task(_take(i)) for i in range(3)]
        try:
            await _wait_for(lambda: len(adapter._run_slot_waiters) == 3)

            adapter._release_run_slot()  # FIFO: the first arrival gets this slot
            await _wait_for(lambda: order == [0])
            assert adapter._run_slots_in_use == 1
            assert len(adapter._run_slot_waiters) == 2

            holds[0].set()
            await _wait_for(lambda: order == [0, 1])
            assert adapter._run_slots_in_use == 1

            holds[1].set()
            await _wait_for(lambda: order == [0, 1, 2])
            assert adapter._run_slots_in_use == 1

            holds[2].set()
            await _wait_for(lambda: not adapter._run_slot_waiters)
            await _wait_for(lambda: adapter._run_slots_in_use == 0)
        finally:
            for task in tasks:
                task.cancel()
            for hold in holds:
                hold.set()

    @pytest.mark.asyncio
    async def test_cancelled_waiter_is_never_handed_a_slot(self):
        adapter = _make_adapter(max_runs=1)
        adapter._run_slots_in_use = 1
        waiter = await _queue_waiter(adapter)

        await _end_hold(adapter, waiter)
        assert adapter._run_queue_depth() == 0

        adapter._release_run_slot()  # nothing may be granted to the abandoned waiter
        assert adapter._run_slots_in_use == 0

    @pytest.mark.asyncio
    async def test_waiter_that_timed_out_does_not_consume_the_next_free_slot(self):
        adapter = _make_adapter(max_runs=1, wait_seconds=0.05)
        adapter._run_slots_in_use = 1

        granted = await adapter._acquire_run_slot()
        assert granted is not None and granted.status == 429
        assert adapter._run_queue_depth() == 0

        adapter._release_run_slot()
        assert adapter._run_slots_in_use == 0


# ---------------------------------------------------------------------------
# Sync surfaces — /v1/chat/completions and /v1/responses
# ---------------------------------------------------------------------------


class TestSyncSurfaceQueue:

    @pytest.mark.asyncio
    async def test_chat_completions_waits_for_a_slot_instead_of_being_refused(self):
        adapter = _make_adapter(max_runs=1, wait_seconds=5.0)
        app = _chat_app(adapter)
        adapter._run_slots_in_use = 1  # a run in flight holds the only slot
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_run_agent", AsyncMock(return_value=({"final_response": "ok"}, {})),
            ) as run_agent:
                pending = asyncio.create_task(
                    cli.post("/v1/chat/completions",
                             json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
                await _wait_for(lambda: len(adapter._run_slot_waiters) == 1)
                assert not pending.done()  # queued, not refused
                run_agent.assert_not_awaited()

                adapter._release_run_slot()  # the live turn ends -> the queued request starts
                resp = await pending

                assert resp.status == 200
                assert (await resp.json())["choices"][0]["message"]["content"] == "ok"

        assert adapter._run_slots_in_use == 0

    @pytest.mark.asyncio
    async def test_responses_waits_for_a_slot_instead_of_being_refused(self):
        adapter = _make_adapter(max_runs=1, wait_seconds=5.0)
        app = _chat_app(adapter)
        adapter._run_slots_in_use = 1
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_run_agent", AsyncMock(return_value=({"final_response": "ok"}, {})),
            ):
                pending = asyncio.create_task(
                    cli.post("/v1/responses", json={"model": "m", "input": "hi"}))
                await _wait_for(lambda: len(adapter._run_slot_waiters) == 1)
                assert not pending.done()

                adapter._release_run_slot()
                resp = await pending

                assert resp.status == 200

        assert adapter._run_slots_in_use == 0

    @pytest.mark.asyncio
    async def test_expired_wait_answers_429_run_queue_timeout(self):
        adapter = _make_adapter(max_runs=1, wait_seconds=0.1)
        app = _chat_app(adapter)
        adapter._run_slots_in_use = 1
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            body = await resp.json()

        assert resp.status == 429
        assert body["error"]["code"] == "run_queue_timeout"
        assert resp.headers.get("Retry-After")
        assert adapter._run_slots_in_use == 1  # the held turn's slot, no leak
        assert adapter._run_queue_depth() == 0

    @pytest.mark.asyncio
    async def test_queued_run_that_expires_ends_failed_with_the_queue_code(self):
        adapter = _make_adapter(max_runs=1, wait_seconds=0.1)
        app = _runs_app(adapter)
        adapter._run_slots_in_use = 1
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "queued"})
            body = await resp.json()
            assert resp.status == 202 and body["status"] == "queued"

            await _wait_for(lambda: (
                adapter._run_statuses.get(body["run_id"], {}).get("status") == "failed"))

            status = adapter._run_statuses[body["run_id"]]
            assert status["code"] == "run_queue_timeout"
            assert "run slot" in status["error"]

        assert adapter._run_slots_in_use == 1
        adapter._release_run_slot()
        assert adapter._run_slots_in_use == 0

    @pytest.mark.asyncio
    async def test_slot_is_released_when_the_turn_returns_an_error(self):
        adapter = _make_adapter(max_runs=1)
        app = _chat_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "m"})

        assert resp.status == 400  # rejected request, but its slot came back
        assert adapter._run_slots_in_use == 0

    @pytest.mark.asyncio
    async def test_cancelled_waiting_request_leaves_no_waiter(self):
        adapter = _make_adapter(max_runs=1, wait_seconds=30.0)
        app = _chat_app(adapter)
        adapter._run_slots_in_use = 1
        async with TestClient(TestServer(app)) as cli:
            pending = asyncio.create_task(
                cli.post("/v1/chat/completions",
                         json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
            await _wait_for(lambda: len(adapter._run_slot_waiters) == 1)

            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

        assert adapter._run_queue_depth() == 0
        assert adapter._run_slots_in_use == 1  # the disconnected waiter never held one
        adapter._release_run_slot()
        assert adapter._run_slots_in_use == 0


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


class TestQueueConfig:

    def test_resolvers_read_gateway_api_server_keys(self):
        cfg = {"gateway": {"api_server": {"run_queue_max_depth": 7, "run_queue_wait_seconds": 42}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_run_queue_max_depth() == 7
            assert APIServerAdapter._resolve_run_queue_wait_seconds() == 42.0

    def test_resolvers_fall_back_to_defaults(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert APIServerAdapter._resolve_run_queue_max_depth() == 100
            assert APIServerAdapter._resolve_run_queue_wait_seconds() == 120.0

    def test_wait_seconds_never_resolves_to_zero(self):
        cfg = {"gateway": {"api_server": {"run_queue_wait_seconds": 0}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_run_queue_wait_seconds() == 120.0
