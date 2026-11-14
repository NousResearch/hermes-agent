"""Gateway and managed admission share the host budget through physical settlement."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from gateway.config import Platform
from gateway import session_managed_worker
from hermes_cli.execution_budget import execution_budget_snapshot, try_acquire_execution_permit


def test_gateway_holds_permit_through_real_post_turn_hook_phase(tmp_path):
    from gateway.execution_admission import run_claimed_message

    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 1\n", encoding="utf-8")

    async def exercise():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def post_hook(**kwargs):
            entered.set()
            await finish.wait()

        runner = SimpleNamespace(
            _resolve_profile_home_for_source=lambda source: tmp_path,
            _hm_post_admission_consume=AsyncMock(return_value=(False, None)),
            _handle_message_with_agent=AsyncMock(return_value="done"),
            _run_post_turn_hooks=post_hook,
            _restore_pending_one_turn_model_override=Mock(),
            _clear_durable_active_turn=AsyncMock(), _release_running_agent_state=Mock(),
            _release_turn_lease=Mock())
        source = SimpleNamespace(platform=Platform.TELEGRAM)
        task = asyncio.create_task(run_claimed_message(runner, SimpleNamespace(), source, "route", 7, False))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            assert len(execution_budget_snapshot(profile_home=tmp_path)[1]) == 1
            permit, refusal = try_acquire_execution_permit(profile_home=tmp_path, execution_id="other", surface="desktop")
            assert permit is None and refusal.reason == "MAX_CONCURRENT_SESSIONS"
        finally:
            finish.set()
            assert await task == "done"
        assert execution_budget_snapshot(profile_home=tmp_path)[1] == []
        runner._release_running_agent_state.assert_called_once_with("route", run_generation=7)
        runner._release_turn_lease.assert_called_once_with("route", 7)

    asyncio.run(exercise())


def test_real_managed_entry_refuses_before_spawning_at_shared_host_capacity(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 1\n", encoding="utf-8")
    held, refusal = try_acquire_execution_permit(profile_home=tmp_path, execution_id="desktop", surface="desktop")
    assert refusal is None
    spawning = Mock(side_effect=AssertionError("managed child must not spawn"))
    monkeypatch.setattr(session_managed_worker, "_mark_spawning", spawning)
    authority = SimpleNamespace(profile_id=str(tmp_path), pending_results={})
    try:
        result = asyncio.run(session_managed_worker.execute_managed(
            authority, SimpleNamespace(session_id="managed"), {"admission_id": "job"}, None))
        assert "(1/1)" in result
        assert authority.pending_results["job"]["result"]["error_code"] == "MAX_CONCURRENT_SESSIONS"
        spawning.assert_not_called()
    finally:
        held.release()
