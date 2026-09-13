"""Required scheduled-run policy callbacks fail closed."""

from __future__ import annotations

from tools import mcp_tool_handlers as mcp_handlers

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from hermes_cli import lifecycle, plugins, plugins_authority
from hermes_cli.plugins import PluginContext, PluginManifest


def _manager(monkeypatch, callbacks=None):
    manager = plugins.PluginManager()
    manager._discovered = True
    manager._authoritative_policies["required"] = callbacks or {}
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    return manager


def _lease(manager, run_id="run-1", session_id="cron-session"):
    manager._authoritative_runs[run_id] = {
        "policy_id": "required",
        "root_session_id": session_id,
        "session_id": session_id,
        "decision": {"action": "allow"},
    }
    manager._authoritative_run_by_session[session_id] = run_id
    return run_id


def _agent():
    return SimpleNamespace(
        _session_db=None, _cached_system_prompt=None,
        _build_system_prompt=lambda _message: "prompt",
        session_id="cron-session", model="test", platform="cron",
        cron_job_id="job", cron_job_name="job", cron_max_turns=12,
        runtime_policy="required", runtime_task_id="run-1",
    )


def _mcp_server(monkeypatch, mcp_tool, session):
    class Lock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    server = SimpleNamespace(session=session, _rpc_lock=Lock(),
                             _pending_call_context=None,
                             _mark_session_proven=lambda: None,
                             mark_tool_call=lambda: None)
    monkeypatch.setattr("tools.mcp_tool_discovery._get_connected_server_for_call", lambda _name: server)
    monkeypatch.setattr("tools.mcp_tool_loop._run_on_mcp_loop",
                        lambda factory, timeout: asyncio.run(factory()))
    return server


def test_authoritative_registration_is_unique_and_disposable():
    manager = plugins.PluginManager()
    context = PluginContext(PluginManifest(name="policy", source="user"), manager)
    def callback(**_kwargs):
        return {"action": "allow"}
    handle = context.register_authoritative_hook(
        "required", "on_session_start", callback)

    with pytest.raises(ValueError, match="already registered"):
        context.register_authoritative_hook(
            "required", "on_session_start", callback)
    handle.dispose()
    assert manager._authoritative_policies == {}


def test_required_admission_absent_or_raising_blocks_before_model(monkeypatch):
    monkeypatch.setattr(
        "agent.credits_tracker.seed_credits_at_session_start", lambda _agent: None,
    )
    manager = _manager(monkeypatch)
    with pytest.raises(RuntimeError, match="has no 'on_session_start'"):
        _restore_or_build_system_prompt(_agent(), None, [])

    def broken(**_kwargs):
        raise ValueError("broken admission")

    manager._authoritative_policies["required"]["on_session_start"] = broken
    with pytest.raises(ValueError, match="broken admission"):
        _restore_or_build_system_prompt(_agent(), None, [])
    assert manager._authoritative_runs == {}


def test_admission_binds_the_run_not_the_session(monkeypatch):
    """The lease is keyed by the immutable fire id, and re-entry rebinds it."""
    monkeypatch.setattr(
        "agent.credits_tracker.seed_credits_at_session_start", lambda _agent: None,
    )
    seen = []

    def admit(**kwargs):
        seen.append(kwargs)
        return {"action": "allow"}

    manager = _manager(monkeypatch, {"on_session_start": admit})
    agent = _agent()
    _restore_or_build_system_prompt(agent, None, [])

    assert list(manager._authoritative_runs) == ["run-1"]
    assert seen[0]["run_id"] == "run-1"
    assert seen[0]["session_id"] == "cron-session"
    assert seen[0]["root_session_id"] == "cron-session"

    # A compression continuation re-enters the first-turn prompt path. Admission
    # is decided once per fire, so this rebinds instead of double-admitting.
    agent.session_id = "cron-session-child"
    _restore_or_build_system_prompt(agent, None, [])
    assert len(seen) == 1
    assert list(manager._authoritative_runs) == ["run-1"]
    assert manager._authoritative_runs["run-1"]["session_id"] == "cron-session-child"
    assert manager._authoritative_runs["run-1"]["root_session_id"] == "cron-session"


@pytest.mark.parametrize("mode", ["absent", "raising"])
def test_required_metadata_failure_prevents_mcp_rpc(monkeypatch, mode):
    import tools.mcp_tool as mcp_tool

    called = False

    class Session:
        async def call_tool(self, _name, **_kwargs):
            nonlocal called
            called = True

    def broken(**_kwargs):
        raise ValueError("broken identity")

    callbacks = {} if mode == "absent" else {"mcp_request_metadata": broken}
    manager = _manager(monkeypatch, callbacks)
    _lease(manager)
    _mcp_server(monkeypatch, mcp_tool, Session())

    result = mcp_handlers._make_tool_handler("fleet", "claim", 5)(
        {}, session_id="cron-session", task_id="run-1")

    assert "Trusted MCP metadata failed" in json.loads(result)["error"]
    assert called is False


@pytest.mark.parametrize("mode", [
    "absent", "raising",
    {"action": "stop"},
    {"action": "stop", "status": "success"},
    {"action": "stop", "reason": "max_items"},
    {"action": "stop", "reason": "", "status": "success"},
    {"action": "stop", "reason": "   ", "status": "success"},
    {"action": "stop", "reason": 42, "status": "success"},
    {"action": "stop", "reason": "max_items", "status": None},
    {"action": "stop", "reason": "max_items", "status": ""},
    {"action": "stop", "reason": "max_items", "status": "partial"},
    {"action": "stop", "reason": "max_items", "status": ["success"]},
])
def test_required_result_failure_latches_terminal_failure(monkeypatch, mode):
    import tools.mcp_tool as mcp_tool

    def broken(**_kwargs):
        raise ValueError("broken result policy")

    callbacks = {"mcp_request_metadata": lambda **_kwargs: {"meta": {}}}
    if mode == "raising":
        callbacks["mcp_tool_result"] = broken
    elif isinstance(mode, dict):
        callbacks["mcp_tool_result"] = lambda **_kwargs: mode
    manager = _manager(monkeypatch, callbacks)
    _lease(manager)

    class Session:
        async def call_tool(self, _name, **_kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text="done")],
                                   isError=False, meta={})

    _mcp_server(monkeypatch, mcp_tool, Session())

    result = mcp_handlers._make_tool_handler("fleet", "claim", 5)(
        {}, session_id="cron-session", task_id="run-1")

    assert "Trusted MCP result policy failed" in json.loads(result)["error"]
    assert mcp_handlers.consume_mcp_runtime_stop() == {
        "reason": "policy_error", "status": "failure", "policy": "required",
        "run_id": "run-1",
    }


def test_success_directive_on_errored_mcp_result_cannot_settle_green(monkeypatch):
    """Core rejects a trusted success when the MCP tool result itself failed.

    A buggy policy callback that ignores ``is_error`` must not be able to
    promote an errored RPC to a durable terminal success.
    """
    import tools.mcp_tool as mcp_tool

    manager = _manager(monkeypatch, {
        "mcp_request_metadata": lambda **_kwargs: {"meta": {}},
        "mcp_tool_result": lambda **_kwargs: {
            "action": "stop", "reason": "quota_complete", "status": "success",
        },
    })
    _lease(manager)

    class Session:
        async def call_tool(self, _name, **_kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text="quota op failed")],
                                   isError=True, meta={})

    _mcp_server(monkeypatch, mcp_tool, Session())
    # An errored RPC is a breaker strike; do not perturb the file's cumulative count.
    mcp_tool._server_error_counts.pop("fleet", None)
    try:
        result = mcp_handlers._make_tool_handler("fleet", "claim", 5)(
            {}, session_id="cron-session", task_id="run-1")

        assert mcp_handlers.consume_mcp_runtime_stop() == {
            "reason": "policy_error", "status": "failure", "policy": "required",
            "run_id": "run-1",
        }
        assert "Trusted MCP result policy failed" in json.loads(result)["error"]
    finally:
        mcp_tool._server_error_counts.pop("fleet", None)


def test_authoritative_result_policy_sees_mcp_error_state(monkeypatch):
    """A failed MCP result must be visible to the policy, not just its metadata."""
    import tools.mcp_tool as mcp_tool

    seen = {}

    def result_policy(**kwargs):
        seen.update(kwargs)
        return {
            "action": "stop",
            "reason": "mcp_failed",
            "status": "failure" if kwargs.get("is_error") else "success",
        }

    manager = _manager(monkeypatch, {
        "mcp_request_metadata": lambda **_kwargs: {"meta": {}},
        "mcp_tool_result": result_policy,
    })
    _lease(manager)

    class Session:
        async def call_tool(self, _name, **_kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text="boom")],
                                   isError=True, meta={})

    _mcp_server(monkeypatch, mcp_tool, Session())
    mcp_handlers._make_tool_handler("fleet", "claim", 5)(
        {}, session_id="cron-session", task_id="run-1")

    assert seen["is_error"] is True
    assert mcp_handlers.consume_mcp_runtime_stop() == {
        "reason": "mcp_failed", "status": "failure", "policy": "required",
        "run_id": "run-1",
    }


def test_transport_failure_skips_result_policy_and_latches_terminal_failure(monkeypatch):
    import tools.mcp_tool as mcp_tool
    from agent.runtime_policy import apply_mcp_runtime_stop

    result_calls = []
    manager = _manager(monkeypatch, {
        "mcp_request_metadata": lambda **_kwargs: {"meta": {}},
        "mcp_tool_result": lambda **kwargs: (
            result_calls.append(kwargs)
            or {"action": "stop", "reason": "quota_complete", "status": "success"}
        ),
    })
    _lease(manager)

    class Session:
        async def call_tool(self, _name, **_kwargs):
            raise RuntimeError("transport failed")

    _mcp_server(monkeypatch, mcp_tool, Session())
    result = mcp_handlers._make_tool_handler("fleet", "claim", 5)(
        {}, session_id="cron-session", task_id="run-1")

    assert "MCP call failed" in json.loads(result)["error"]
    assert result_calls == []
    agent = _agent()
    apply_mcp_runtime_stop(agent)
    assert agent._runtime_stop_reason == "mcp_transport_error"
    assert agent._runtime_terminal_outcome["status"] == "failure"


def test_observer_policy_stays_non_authoritative_across_lifecycle(monkeypatch):
    import tools.mcp_tool as mcp_tool
    from agent.runtime_policy import apply_mcp_runtime_stop
    from cron.jobs import _normalize_runtime_policy
    from cron.scheduler_settlement import classify_completion, settle_run
    from tools import delegate_tool

    manager = _manager(monkeypatch)
    agent = _agent()
    agent.runtime_policy = _normalize_runtime_policy("observer")
    seen = []

    def observe(hook, **_kwargs):
        seen.append(hook)
        return []

    monkeypatch.setattr(plugins, "invoke_hook", observe)
    monkeypatch.setattr("agent.credits_tracker.seed_credits_at_session_start", lambda _agent: None)
    _restore_or_build_system_prompt(agent, None, [])
    assert manager._authoritative_runs == {}

    class Session:
        async def call_tool(self, _name, **kwargs):
            assert not kwargs.get("meta")
            return SimpleNamespace(content=[SimpleNamespace(text="done")], isError=False, meta={})

    _mcp_server(monkeypatch, mcp_tool, Session())
    result = mcp_handlers._make_tool_handler("fleet", "claim", 5)(
        {}, session_id=agent.session_id, task_id=agent.runtime_task_id)
    assert json.loads(result) == {"result": "done"}
    mcp_handlers._mcp_runtime_stop.set({"status": "success"})
    apply_mcp_runtime_stop(agent)
    assert getattr(agent, "_runtime_stop_reason", None) is None

    monkeypatch.setattr(delegate_tool, "is_spawn_paused", lambda: True)
    assert "paused" in json.loads(delegate_tool.delegate_task(goal="work", parent_agent=agent))["error"]

    job = {"id": "job", "runtime_policy": agent.runtime_policy}
    settle_run(agent, job, "exec-1", agent.session_id, {"completed": True})
    assert seen == ["on_session_start", "mcp_request_metadata", "mcp_tool_result", "on_session_finalize"]
    assert classify_completion(job, "exec-1", agent.session_id, True, None, "")[0] is False


def test_authority_survives_session_rotation(monkeypatch):
    """A compression child session must not silently take the observer path."""
    import tools.mcp_tool as mcp_tool

    metadata_calls = []
    result_calls = []

    manager = _manager(monkeypatch, {
        "mcp_request_metadata": lambda **kwargs: (
            metadata_calls.append(kwargs) or {"meta": {"run": kwargs["run_id"]}}
        ),
        "mcp_tool_result": lambda **kwargs: (
            result_calls.append(kwargs)
            or {"action": "stop", "reason": "max_items", "status": "success"}
        ),
        "on_session_finalize": lambda **_kwargs: {"status": "finalized"},
    })
    _lease(manager)

    # No MCP authority decision may fall back to the fail-open bus once a lease
    # exists. on_session_finalize still notifies compatibility observers.
    real_invoke_hook = plugins.invoke_hook

    def guarded_invoke_hook(hook_name, *args, **kwargs):
        if hook_name.startswith("mcp_"):
            pytest.fail(f"observer bus used for authoritative {hook_name}")
        return real_invoke_hook(hook_name, *args, **kwargs)

    monkeypatch.setattr(plugins, "invoke_hook", guarded_invoke_hook)

    sent_meta = {}

    class Session:
        async def call_tool(self, _name, **kwargs):
            sent_meta.update(kwargs.get("meta") or {})
            return SimpleNamespace(content=[SimpleNamespace(text="done")],
                                   isError=False, meta={})

    _mcp_server(monkeypatch, mcp_tool, Session())

    # The run rotated onto a compression continuation: same fire, new session id.
    plugins_authority.bind_authoritative_run_session("run-1", "cron-session-child")
    mcp_handlers._make_tool_handler("fleet", "claim", 5)(
        {}, session_id="cron-session-child", task_id="run-1")

    assert metadata_calls[0]["session_id"] == "cron-session-child"
    assert metadata_calls[0]["root_session_id"] == "cron-session"
    assert sent_meta == {"run": "run-1"}
    assert result_calls and result_calls[0]["session_id"] == "cron-session-child"
    assert mcp_handlers.consume_mcp_runtime_stop() == {
        "reason": "max_items", "status": "success", "policy": "required",
        "run_id": "run-1",
    }

    # Finalization is still required, and resolves from the rotated id alone.
    receipts = lifecycle.finalize_session(
        session_id="cron-session-child", platform="cron",
    )
    # The receipt binds the rotated session; the fire's run id stays authoritative.
    assert receipts[0] == {
        "status": "finalized", "policy": "required", "run_id": "run-1",
        "session_id": "cron-session-child", "root_session_id": "cron-session",
    }
    assert manager._authoritative_runs == {}
    assert manager._authoritative_run_by_session == {}


def test_authority_resolves_by_rotated_session_without_run_id(monkeypatch):
    """A caller that only knows the rotated session id still fails closed."""
    import tools.mcp_tool as mcp_tool

    called = False

    class Session:
        async def call_tool(self, _name, **_kwargs):
            nonlocal called
            called = True

    manager = _manager(monkeypatch)
    _lease(manager)
    plugins_authority.bind_authoritative_run_session("run-1", "cron-session-child")
    _mcp_server(monkeypatch, mcp_tool, Session())

    result = mcp_handlers._make_tool_handler("fleet", "claim", 5)(
        {}, session_id="cron-session-child", task_id="")

    assert "Trusted MCP metadata failed" in json.loads(result)["error"]
    assert called is False


def test_plugin_unload_leaves_an_in_flight_run_fail_closed(monkeypatch):
    """Absence of the policy must read as "required and missing", not "none"."""
    import tools.mcp_tool as mcp_tool

    called = False

    class Session:
        async def call_tool(self, _name, **_kwargs):
            nonlocal called
            called = True

    manager = _manager(monkeypatch, {
        "mcp_request_metadata": lambda **_kwargs: {"meta": {}},
    })
    _lease(manager)
    manager._unload_scoped()

    assert manager._authoritative_policies == {}
    assert "run-1" in manager._authoritative_runs

    _mcp_server(monkeypatch, mcp_tool, Session())
    result = mcp_handlers._make_tool_handler("fleet", "claim", 5)(
        {}, session_id="cron-session", task_id="run-1")

    assert "Trusted MCP metadata failed" in json.loads(result)["error"]
    assert called is False


def test_required_finalizer_failure_has_no_receipt_and_stays_active(monkeypatch):
    def broken(**_kwargs):
        raise ValueError("broken settlement")

    manager = _manager(monkeypatch, {"on_session_finalize": broken})
    _lease(manager)

    with pytest.raises(ValueError, match="broken settlement"):
        lifecycle.finalize_session(
            session_id="cron-session", runtime_run_id="run-1", platform="cron",
        )
    assert list(manager._authoritative_runs) == ["run-1"]


def test_required_finalizer_demands_receipt(monkeypatch):
    manager = _manager(monkeypatch, {"on_session_finalize": lambda **_kwargs: None})
    _lease(manager)

    with pytest.raises(RuntimeError, match="returned no receipt"):
        lifecycle.finalize_session(
            session_id="cron-session", runtime_run_id="run-1", platform="cron",
        )
    assert list(manager._authoritative_runs) == ["run-1"]


def test_required_finalizer_cannot_be_absent(monkeypatch):
    manager = _manager(monkeypatch)
    _lease(manager)

    with pytest.raises(RuntimeError, match="has no 'on_session_finalize'"):
        lifecycle.finalize_session(
            session_id="cron-session", runtime_run_id="run-1", platform="cron",
        )
    assert list(manager._authoritative_runs) == ["run-1"]


def test_settlement_failure_still_runs_core_and_observer_teardown(monkeypatch):
    """A failed receipt fails the run; it must not skip lifecycle teardown."""
    order = []

    def broken(**_kwargs):
        order.append("settlement")
        raise ValueError("broken settlement")

    manager = _manager(monkeypatch, {"on_session_finalize": broken})
    _lease(manager)

    monkeypatch.setattr(
        "hermes_cli.observability.observe_lifecycle",
        lambda *_a, **_kw: order.append("observability"),
    )
    monkeypatch.setattr(
        plugins, "invoke_hook",
        lambda *_a, **_kw: (order.append("observers"), [])[1],
    )

    class Coordinator:
        @staticmethod
        def finalize_conversation(**_kwargs):
            order.append("relay")

    monkeypatch.setattr("agent.relay_runtime.SESSION_COORDINATOR", Coordinator)
    monkeypatch.setattr("agent.relay_runtime.current_profile_key", lambda: "p")

    with pytest.raises(ValueError, match="broken settlement"):
        lifecycle.finalize_session(
            session_id="cron-session", runtime_run_id="run-1", platform="cron",
        )

    assert order == ["settlement", "observability", "relay", "observers"]
