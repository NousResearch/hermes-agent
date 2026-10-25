from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent import subscription_capacity_runtime as runtime


def _snapshot(provider="openai-codex", *, identity="acct-1", used=20.0):
    now = datetime.now(timezone.utc)
    return AccountUsageSnapshot(
        provider=provider,
        source="usage_api" if provider == "openai-codex" else "oauth_usage_api",
        fetched_at=now,
        identity=identity,
        windows=(AccountUsageWindow("global", used, now + timedelta(hours=2)),),
    )


def _task(provider="openai-codex", model="gpt-6-sol", **kwargs):
    values = dict(
        id="ESN-TEST", provider_override=provider, model_override=model, current_run_id=7,
        max_runtime_seconds=60, skills=[], current_step_key=None,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _reset_runtime(monkeypatch):
    monkeypatch.delenv(runtime.CAPACITY_LEASE_ENV, raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "ESN-TEST")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setattr(runtime, "_runtime", None)
    monkeypatch.setattr(runtime, "_runtime_released", False)


def test_paid_alias_is_not_reinterpreted_or_fetched(tmp_path):
    called = []
    payload = runtime.prepare_kanban_worker_capacity(
        _task("openrouter", "openai/gpt-6"), shared_root=tmp_path, profile_home=None,
        config={}, fetcher=lambda provider: called.append(provider),
    )
    assert payload is None
    assert called == []


def test_anthropic_without_stable_usage_identity_fails_closed(tmp_path):
    with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="identity unavailable"):
        runtime.prepare_kanban_worker_capacity(
            _task("anthropic", "claude-sonnet-4"), shared_root=tmp_path, profile_home=None,
            config={"kanban": {"subscription_capacity": {"claude_floor_percent": 12}}},
            fetcher=lambda provider: _snapshot("anthropic", identity=None),
        )


def test_claude_floor_must_be_explicit_and_positive(tmp_path):
    with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="claude_floor_percent"):
        runtime.prepare_kanban_worker_capacity(
            _task("anthropic", "claude-sonnet-4"), shared_root=tmp_path, profile_home=None,
            config={}, fetcher=lambda provider: _snapshot("anthropic"),
        )


def test_acquire_phase_check_and_normal_release_use_same_durable_lease(tmp_path, monkeypatch):
    snapshot = _snapshot()
    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: snapshot,
    )
    monkeypatch.setenv(runtime.CAPACITY_LEASE_ENV, payload)
    assert runtime.bind_prepared_capacity(payload, os.getpid()) is True
    router, lease = runtime._load_runtime()
    router.fetcher = lambda provider: snapshot

    runtime.check_worker_capacity_phase(SimpleNamespace(provider=lease.provider, model=lease.model))
    assert runtime.release_worker_capacity() is True
    assert runtime.release_worker_capacity() is False
    assert router.check_phase(lease).admitted is False


def test_route_change_cannot_escape_to_paid_fallback(tmp_path, monkeypatch):
    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: _snapshot(),
    )
    monkeypatch.setenv(runtime.CAPACITY_LEASE_ENV, payload)
    assert runtime.bind_prepared_capacity(payload, os.getpid()) is True
    with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="route changed"):
        runtime.check_worker_capacity_phase(SimpleNamespace(provider="openrouter", model="gpt-6-sol"))


def test_phase_denial_is_terminal_before_fallback_logic():
    from agent.turn_api_error import handle_api_error

    agent = SimpleNamespace(
        thinking_callback=None,
        _try_activate_fallback=lambda: (_ for _ in ()).throw(AssertionError("fallback invoked")),
    )
    verdict = handle_api_error(
        agent, api_error=runtime.SubscriptionCapacityUnavailable("floor crossed"), _retry=None,
        thinking_spinner=None, messages=[], api_messages=[], api_kwargs={}, system_message="",
        active_system_prompt="", conversation_history=[], approx_tokens=0, retry_count=0,
        max_retries=3, compression_attempts=0, max_compression_attempts=1, api_call_count=1,
        api_request_id="req", api_start_time=0, effective_task_id="task", turn_id="turn",
    )
    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == "subscription_capacity"


def test_real_api_call_boundary_denies_before_transport(monkeypatch):
    from agent.turn_api_call import perform_api_call

    monkeypatch.setattr(
        runtime, "check_worker_capacity_phase",
        lambda agent: (_ for _ in ()).throw(runtime.SubscriptionCapacityUnavailable("denied")),
    )
    agent = SimpleNamespace()
    with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="denied"):
        perform_api_call(
            agent, api_kwargs={}, _original_api_kwargs={}, _llm_middleware_trace=[],
            _moa_prepared_request=None, _retry=None, thinking_spinner=None, retry_count=0,
            api_call_count=1, api_request_id="req", effective_task_id="task", turn_id="turn",
            interrupted=False,
        )


def test_iteration_summary_boundary_denies_before_direct_transport(monkeypatch):
    from agent.chat_completion_helpers import _capacity_guarded_summary_attempt

    monkeypatch.setattr(
        runtime, "check_worker_capacity_phase",
        lambda agent: (_ for _ in ()).throw(runtime.SubscriptionCapacityUnavailable("summary denied")),
    )
    called = []
    agent = SimpleNamespace(provider="openai-codex", model="gpt-6-sol", api_mode="codex_responses")
    with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="summary denied"):
        _capacity_guarded_summary_attempt(agent, lambda retry: called.append(retry), 0)
    assert called == []


def test_pre_spawn_failure_and_hard_death_recovery_release(tmp_path):
    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: _snapshot(),
    )
    assert runtime.release_prepared_capacity(payload) is True

    payload = runtime.prepare_kanban_worker_capacity(
        _task(current_run_id=8), shared_root=tmp_path, profile_home=None, config={},
        fetcher=lambda provider: _snapshot(),
    )
    assert payload
    assert runtime.release_recovered_kanban_capacity(tmp_path, "ESN-TEST", 8) == 1


def test_repair_role_comes_from_durable_workflow_step(tmp_path):
    payload = runtime.prepare_kanban_worker_capacity(
        _task(current_step_key="repair_findings"), shared_root=tmp_path, profile_home=None,
        config={}, fetcher=lambda provider: _snapshot(),
    )
    import json
    assert json.loads(payload)["lease"]["task"] == "repair"


def test_in_process_delegate_cannot_check_or_release_parent_lease(tmp_path, monkeypatch):
    from agent.delegation_context import delegated_child_context

    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: _snapshot(),
    )
    monkeypatch.setenv(runtime.CAPACITY_LEASE_ENV, payload)
    assert runtime.bind_prepared_capacity(payload, os.getpid()) is True
    router, lease = runtime._load_runtime()
    router.fetcher = lambda provider: _snapshot()
    with delegated_child_context("child"):
        with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="not owned"):
            runtime.check_worker_capacity_phase(SimpleNamespace(provider=lease.provider, model=lease.model))
        assert runtime.release_worker_capacity() is False
    assert runtime.release_worker_capacity() is True


def test_spawned_descendant_gets_no_lease_and_cannot_rebind_process(tmp_path, monkeypatch):
    from agent.delegation_context import delegated_child_subprocess_env

    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: _snapshot(),
    )
    monkeypatch.setenv(runtime.CAPACITY_LEASE_ENV, payload)
    assert runtime.bind_prepared_capacity(payload, os.getpid()) is True
    router, lease = runtime._load_runtime()
    router.fetcher = lambda provider: _snapshot()
    runtime.check_worker_capacity_phase(SimpleNamespace(provider=lease.provider, model=lease.model))
    child_env = delegated_child_subprocess_env(dict(os.environ))
    assert runtime.CAPACITY_LEASE_ENV not in child_env

    owner_pid = os.getpid()
    monkeypatch.setattr(runtime.os, "getpid", lambda: owner_pid + 1)
    with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="not owned"):
        runtime.check_worker_capacity_phase(SimpleNamespace(provider=lease.provider, model=lease.model))


def test_copied_envelope_cannot_self_register_or_release(tmp_path, monkeypatch):
    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: _snapshot(),
    )
    monkeypatch.setenv(runtime.CAPACITY_LEASE_ENV, payload)
    router, lease = runtime._load_runtime()
    router.fetcher = lambda provider: _snapshot()

    # A descendant can copy/import the public envelope parser, but its process identity is not the
    # dispatcher authority persisted at admission and therefore cannot create a binding.
    import agent.subscription_capacity as capacity_store
    dispatcher_pid = os.getpid()
    with monkeypatch.context() as child:
        child.setattr(capacity_store.os, "getpid", lambda: dispatcher_pid + 1000)
        assert runtime.bind_prepared_capacity(payload, dispatcher_pid + 1000) is False
        assert runtime.release_prepared_capacity(payload) is False
        with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="not owned"):
            runtime.check_worker_capacity_phase(SimpleNamespace(provider=lease.provider, model=lease.model))
        assert runtime.release_worker_capacity() is False
    assert router.worker_process_matches(lease, dispatcher_pid) is False
    assert runtime.release_prepared_capacity(payload) is True


def test_dispatcher_binding_accepts_only_exact_spawned_pid(tmp_path, monkeypatch):
    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: _snapshot(),
    )
    exact_pid = os.getpid()
    assert runtime.bind_prepared_capacity(payload, exact_pid) is True
    assert runtime.bind_prepared_capacity(payload, exact_pid + 1) is False
    monkeypatch.setenv(runtime.CAPACITY_LEASE_ENV, payload)
    router, lease = runtime._load_runtime()
    router.fetcher = lambda provider: _snapshot()
    runtime.check_worker_capacity_phase(SimpleNamespace(provider=lease.provider, model=lease.model))


def test_replayed_lease_cannot_be_stolen_or_deleted_by_new_dispatcher(tmp_path, monkeypatch):
    snapshot = _snapshot()
    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: snapshot,
    )
    import agent.subscription_capacity as capacity_store
    dispatcher_pid = os.getpid()
    with monkeypatch.context() as replacement:
        replacement.setattr(capacity_store.os, "getpid", lambda: dispatcher_pid + 1000)
        with pytest.raises(runtime.SubscriptionCapacityUnavailable, match="another dispatcher"):
            runtime.prepare_kanban_worker_capacity(
                _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: snapshot,
            )
        assert runtime.bind_prepared_capacity(payload, dispatcher_pid + 1000) is False
        assert runtime.release_prepared_capacity(payload) is False
    assert runtime.release_prepared_capacity(payload) is True


def test_same_dispatcher_replay_cannot_release_live_first_worker(tmp_path):
    snapshot = _snapshot()
    payload = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: snapshot,
    )
    first_worker_pid = os.getpid() + 100
    second_worker_pid = first_worker_pid + 1
    assert runtime.bind_prepared_capacity(payload, first_worker_pid) is True

    replay = runtime.prepare_kanban_worker_capacity(
        _task(), shared_root=tmp_path, profile_home=None, config={}, fetcher=lambda provider: snapshot,
    )
    assert replay == payload
    assert runtime.bind_prepared_capacity(replay, second_worker_pid) is False
    assert runtime.release_prepared_capacity(replay) is False

    state = json.loads(payload)
    router = runtime.SubscriptionCapacityRouter(
        Path(state["db_path"]), (runtime._policy_from_dict(state["policy"]),),
        coordination_scope=state["coordination_scope"], fetcher=lambda provider: snapshot,
    )
    lease = runtime._lease_from_dict(state["lease"])
    assert router.worker_process_matches(lease, first_worker_pid) is True
