"""``active_agents`` must fall back to the truth when NON-chat work ends.

``GatewayRunner._active_work_count()`` aggregates four sources: chat turns,
cron jobs, API-server runs and deferred agent workers.  Only the CHAT
boundary ever wrote that aggregate into ``gateway_state.json``, so whatever
the other sources were worth at the instant a turn ended got baked into the
file and was never corrected:

    cron job starts
    chat turn ends   -> persists active_agents=1 (0 chat slots + 1 cron job)
    cron job ends    -> nothing re-persists; the file stays 1 forever

Every test here asserts the STATE OF THE FILE after the work ends — the
number a reader (``/api/status``, ``gateway_busy``, the restart drain, the
external idle-restarter) actually consults — not that some method was called.
"""

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from cron import scheduler as cron_scheduler
from gateway import status as gateway_status
from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _release_pending_api_work
from gateway.run import GatewayRunner


def _read_state() -> dict:
    path = gateway_status._get_runtime_status_path()
    return json.loads(path.read_text(encoding="utf-8"))


class _StubApiAdapter:
    """Minimal stand-in for the API-server adapter's work accounting."""

    def __init__(self) -> None:
        self._pending_agent_requests = 0

    def active_agent_work_count(self) -> int:
        return self._pending_agent_requests


@pytest.fixture
def runner(monkeypatch):
    """A bare runner wired to the REAL work-count and persist methods."""
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agent_count = lambda: 0
    for name in (
        "_active_deferred_agent_worker_count",
        "_track_deferred_agent_worker",
    ):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner, GatewayRunner))
    runner._active_cron_job_count = GatewayRunner._active_cron_job_count.__get__(
        runner, GatewayRunner
    )
    runner._active_api_run_count = GatewayRunner._active_api_run_count.__get__(
        runner, GatewayRunner
    )
    runner._active_work_count = GatewayRunner._active_work_count.__get__(
        runner, GatewayRunner
    )
    runner._persist_active_agents = GatewayRunner._persist_active_agents.__get__(
        runner, GatewayRunner
    )
    # The notifier resolves the live runner exactly the way the gateway does.
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    return runner


@pytest.fixture(autouse=True)
def _clean_cron_inflight():
    """No cron state may leak between tests in this module."""
    cron_scheduler._running_job_ids.clear()
    cron_scheduler._running_fire_owners.clear()
    yield
    cron_scheduler._running_job_ids.clear()
    cron_scheduler._running_fire_owners.clear()


def test_cron_job_end_decrements_the_persisted_count(runner):
    """The incident path: a cron job in flight when a chat turn ended."""
    assert cron_scheduler.try_register_running_job("job-a") is True
    # The chat turn's own final persist — the last write of the night.
    runner._persist_active_agents()
    assert _read_state()["active_agents"] == 1

    cron_scheduler.release_running_job("job-a")

    # No further chat turn happens.  The file must already tell the truth.
    assert _read_state()["active_agents"] == 0


def test_cron_job_start_is_visible_without_a_chat_turn(runner):
    """The increment edge is symmetric with the chat claim."""
    runner._persist_active_agents()
    assert _read_state()["active_agents"] == 0

    cron_scheduler.try_register_running_job("job-b")

    assert _read_state()["active_agents"] == 1


def test_cron_fire_owner_release_decrements_the_persisted_count(runner, monkeypatch):
    """``_running_fire_owners`` is the second half of the cron aggregate.

    ``get_running_job_ids()`` unions it with ``_running_job_ids``, so
    ``run_one_job``'s owner registration/removal is its own persist edge.
    """
    seen_during_run = {}

    def _fake_heartbeat(job, body):
        seen_during_run["active_agents"] = _read_state()["active_agents"]
        return True

    monkeypatch.setattr(
        cron_scheduler, "_run_with_fire_claim_heartbeat", _fake_heartbeat
    )

    assert cron_scheduler.run_one_job({"id": "job-c"}) is True

    assert seen_during_run["active_agents"] == 1
    assert _read_state()["active_agents"] == 0


def test_api_run_end_decrements_the_persisted_count(runner):
    """Same class as cron: API work lives outside ``_running_agents``."""
    adapter = _StubApiAdapter()
    runner.adapters[Platform.API_SERVER] = adapter

    adapter._pending_agent_requests = 1
    runner._persist_active_agents()
    assert _read_state()["active_agents"] == 1

    reservation = {"active": True}
    _release_pending_api_work(adapter, reservation)

    assert _read_state()["active_agents"] == 0


@pytest.mark.asyncio
async def test_run_task_completion_decrements_the_persisted_count(runner):
    """``/v1/runs`` counts UN-DONE tasks, so completion is its own edge.

    The run task removes itself from ``_active_run_tasks`` through a callback,
    which is why the dict is not the counter — a run that finishes without any
    chat turn afterwards must still bring the persisted number back down.  The
    request is driven through the real handler so the edge exercised here is
    the one production installs, not one this test wires up itself.
    """
    api = APIServerAdapter(PlatformConfig(enabled=True))
    runner.adapters[Platform.API_SERVER] = api
    app = web.Application()
    app.router.add_post("/v1/runs", api._handle_runs)

    original_create_task = asyncio.create_task
    task_started = asyncio.Event()
    allow_task = asyncio.Event()

    def _gated_create_task(coro):
        """Hold the run task open so the in-flight number is observable."""

        async def _gated():
            task_started.set()
            await allow_task.wait()
            return await coro

        return original_create_task(_gated())

    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    with patch(
        "gateway.platforms.api_server.asyncio.create_task",
        side_effect=_gated_create_task,
    ), patch.object(api, "_create_agent", return_value=agent):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/runs", json={"input": "hello"})
            assert response.status == 202
            await task_started.wait()

            assert _read_state()["active_agents"] == 1

            allow_task.set()
            for _ in range(500):
                if _read_state()["active_agents"] == 0:
                    break
                await asyncio.sleep(0.01)

    assert _read_state()["active_agents"] == 0


@pytest.mark.asyncio
async def test_deferred_worker_completion_decrements_the_persisted_count(runner):
    """A deferred worker outlives its turn, so no turn boundary follows it.

    ``_track_deferred_agent_worker`` exists for executor work that is still
    running after the gateway turn that started it returned — precisely the
    shape that leaves an inflated number behind, because the turn's own final
    persist already happened while the worker was in flight.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    runner._track_deferred_agent_worker(future, object())

    assert _read_state()["active_agents"] == 1

    future.set_result(None)
    await asyncio.sleep(0)

    assert _read_state()["active_agents"] == 0


def test_release_without_a_live_gateway_is_a_no_op(monkeypatch):
    """Cron also runs in bare CLI processes that own no status file."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: None)
    cron_scheduler.try_register_running_job("job-d")
    cron_scheduler.release_running_job("job-d")  # must not raise

    path = gateway_status._get_runtime_status_path()
    assert not path.exists() or "active_agents" not in _read_state()


def test_concurrent_status_writers_do_not_lose_each_others_updates(monkeypatch):
    """``write_runtime_status`` is read-merge-write; it must be serialized.

    Cron jobs and API runs persist from their own threads now, so the merge
    window is genuinely concurrent.  An unserialized merge drops the loser's
    update outright — each writer here contributes one platform entry, and a
    lost merge loses that entry.

    The real merge window is microseconds wide, so it is widened here by
    slowing the read→write span: that models a busy host or a slow status
    read, it does not fake anything the code does not really do.
    """
    real_build_pid_record = gateway_status._build_pid_record

    def _slow_build_pid_record():
        time.sleep(0.02)
        return real_build_pid_record()

    monkeypatch.setattr(gateway_status, "_build_pid_record", _slow_build_pid_record)

    writers = [f"platform-{index}" for index in range(6)]
    barrier = threading.Barrier(len(writers))

    def _write(name: str) -> None:
        barrier.wait()
        gateway_status.write_runtime_status(platform=name, platform_state="connected")

    threads = [threading.Thread(target=_write, args=(name,)) for name in writers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    platforms = _read_state()["platforms"]
    assert sorted(platforms) == sorted(writers)
