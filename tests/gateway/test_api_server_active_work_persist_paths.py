"""#122813 review follow-up: the /v1/runs and in-flight-turn persist call sites
must be covered by tests that fail when the persist call is deleted (the
original PR's suites tested the helpers directly — deleting either the
``_handle_runs`` tail persist or the ``_run_agent`` finally persist left every
suite green).

Reuses the full-adapter fixtures from test_api_server_runs so the real
adapter state (metrics, session db, run registries) is exercised.
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

import gateway.platforms.api_server_runs as api_server_runs
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter


class TestRunsPathPersists:
    @pytest.mark.asyncio
    async def test_handle_runs_persists_on_task_register_and_completion(self):
        """Deleting ``_persist_adapter_active_work(self)`` at task registration
        (api_server_runs._handle_runs tail) must fail this test."""
        adapter = _make_adapter()
        persist_counts = []
        adapter._persist_active_work = lambda: persist_counts.append(
            len(adapter._active_run_tasks))

        async def fake_execute_run(_self, run, *, _api_server):
            pass  # no real agent machinery; registration-side effects only

        with patch.object(adapter, "_create_agent"), \
             patch.object(api_server_runs, "_execute_run", fake_execute_run):
            app = _create_runs_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202, await resp.text()

        # RED if the registration persist is deleted: no persist observed a
        # live task (count >= 1) at registration time.
        assert any(n >= 1 for n in persist_counts), (
            "no persist fired while the /v1/runs task was live")
        # Done-callback persist: wait for the task + its done callbacks.
        await asyncio.sleep(0.05)
        assert any(n == 0 for n in persist_counts), (
            "no persist fired after the run task completed")

    @pytest.mark.asyncio
    async def test_run_agent_persists_on_enter_and_finally(self):
        """The in-flight-turn seam (api_server._run_agent's
        ``_inflight_agent_runs += 1`` → persist, ``finally`` → persist) fires
        exactly twice for one turn. Deleting either persist must fail this."""
        adapter = _make_adapter()
        persist_counts = []
        adapter._persist_active_work = lambda: persist_counts.append(
            adapter._inflight_agent_runs)

        async def fake_submit(loop, fn):
            # Inside the awaited executor submit: the enter persist must
            # already have fired with the count at 1.
            persist_counts.append(-adapter._inflight_agent_runs)
            return ({"final_response": "ok"},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        with patch.object(api_server_runs, "_submit_api_worker", fake_submit):
            result, usage = await adapter._run_agent("hi", [])

        assert persist_counts[0] == 1, (
            f"enter persist missing or wrong count: {persist_counts}")
        assert -1 in persist_counts, "executor ran without the count raised"
        assert persist_counts[-1] == 0, (
            f"finally persist missing or wrong count: {persist_counts}")
        assert persist_counts.count(1) >= 1 and persist_counts.count(0) >= 1
        assert result == {"final_response": "ok"}


# ------------------------------------------------- helper-deletion canaries


class TestPersistHelperDeletionCanaries:
    def test_runs_helper_delegates_to_adapter_persist(self):
        adapter = _make_adapter()
        calls = []
        adapter._persist_active_work = lambda: calls.append(1)
        api_server_runs._persist_adapter_active_work(adapter)
        assert calls == [1]

    def test_runs_helper_swallows_persist_failure(self):
        adapter = _make_adapter()

        def boom():
            raise RuntimeError("persist failed")

        adapter._persist_active_work = boom
        api_server_runs._persist_adapter_active_work(adapter)  # must not raise

    def test_notify_run_task_done_persists(self):
        adapter = _make_adapter()
        calls = []
        adapter._persist_active_work = lambda: calls.append(1)
        api_server_runs._notify_run_task_done(adapter)(None)
        assert calls == [1]
