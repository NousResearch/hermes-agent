"""Conformance proof for session-stable worker interface adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.subagent_lifecycle import (
    SubagentLifecycleService, bind_subagent_parent, dispatch_worker_nested_tool,
)
from agent.worker_interfaces import (
    CANONICAL_WORKER_TOOL,
    InterfaceSelection,
    QualifiedInterfaceProfile,
    bind_worker_interface,
    canonical_worker_capability,
    dispatch_worker_interface_call,
    frozen_worker_interface_contract,
    is_worker_interface_tool,
    normalize_worker_call,
    project_worker_tool_definitions,
    resolve_worker_interface,
    restore_worker_interface_contract,
)
from agent.worker_store import WorkerStore
from hermes_state import SessionDB
from run_agent import AIAgent


def _schema(name):
    return {
        "type": "function",
        "function": {"name": name, "description": name, "parameters": {"type": "object"}},
    }


def _selection(name, definitions):
    return bind_worker_interface(
        InterfaceSelection(name, "explicit", "experimental_unqualified", "fixture", "model"),
        definitions,
    )


def _advertised(selection, semantic):
    return dict(selection.aliases)[semantic]


def test_resolution_and_collision_binding_preserve_native_tools_and_routes(monkeypatch):
    definitions = [_schema("send_message"), _schema(CANONICAL_WORKER_TOOL)]
    hermes = bind_worker_interface(
        resolve_worker_interface({}, provider="Fixture", model="Model"), definitions
    )
    assert [item["function"]["name"] for item in project_worker_tool_definitions(definitions, hermes)] == [
        "send_message", CANONICAL_WORKER_TOOL,
    ]
    assert not is_worker_interface_tool(hermes, "send_message")
    assert is_worker_interface_tool(hermes, CANONICAL_WORKER_TOOL)

    codex = _selection("codex", definitions)
    assert _advertised(codex, "send_message") == "hermes_worker_send_message"
    projected = project_worker_tool_definitions(definitions, codex)
    names = [item["function"]["name"] for item in projected]
    assert "send_message" in names
    assert "hermes_worker_send_message" in names
    assert CANONICAL_WORKER_TOOL not in names
    assert len(names) == len(set(names))
    assert canonical_worker_capability(codex, "send_message") == "send_message"
    assert canonical_worker_capability(codex, "hermes_worker_send_message") == CANONICAL_WORKER_TOOL

    nested_calls = []
    parent = SimpleNamespace(
        _worker_interface_selection=codex,
        _dispatch_worker_interface=lambda name, args: json.dumps({"worker": name, "args": args}),
    )
    monkeypatch.setattr(
        "model_tools.handle_function_call",
        lambda name, args, **kwargs: nested_calls.append((name, args)) or json.dumps({"native": name}),
    )
    with bind_subagent_parent(parent):
        worker_result = json.loads(dispatch_worker_nested_tool(
            "hermes_worker_send_message", {"target": "worker", "message": "guide"}, task_id="test"
        ))
        native_result = json.loads(dispatch_worker_nested_tool(
            "send_message", {"action": "list"}, task_id="test"
        ))
    assert worker_result["worker"] == "hermes_worker_send_message"
    assert native_result == {"native": "send_message"}
    assert nested_calls == [("send_message", {"action": "list"})]

    contract = frozen_worker_interface_contract(codex)
    from agent.subagent_lifecycle import _profile_policy_snapshot

    _revision, stored_policy = _profile_policy_snapshot(
        {}, None, {}, child=SimpleNamespace(
            _worker_interface_selection=codex,
            _worker_effective_tool_names={CANONICAL_WORKER_TOOL},
        ),
    )
    assert stored_policy["worker_interface"] == contract
    restored = restore_worker_interface_contract(
        contract, provider="changed-provider", model="changed-model", definitions=definitions,
    )
    assert restored.aliases == codex.aliases
    assert (restored.name, restored.provider, restored.model) == (
        "codex", "changed-provider", "changed-model",
    )
    legacy = restore_worker_interface_contract(
        None, provider="fixture", model="model", definitions=definitions,
    )
    assert (legacy.name, legacy.source) == ("hermes", "legacy_session")
    with pytest.raises(ValueError, match="now collide"):
        restore_worker_interface_contract(
            contract,
            provider="fixture",
            model="model",
            definitions=[*definitions, _schema("hermes_worker_send_message")],
        )
    with pytest.raises(ValueError, match="now collides"):
        project_worker_tool_definitions(
            [*definitions, _schema("hermes_worker_send_message")], codex,
        )

    qualified = resolve_worker_interface(
        {"orchestration": {"interface": "auto"}},
        provider="Fixture", model="Model",
        qualified_profiles=(QualifiedInterfaceProfile("fixture", "model", "claude", "live:fixture"),),
    )
    assert (qualified.name, qualified.source, qualified.evidence_ref) == (
        "claude", "qualified_exact_match", "live:fixture",
    )
    fallback = resolve_worker_interface(
        {"orchestration": {"interface": "auto"}}, provider="fixture", model="other",
        qualified_profiles=(QualifiedInterfaceProfile("fixture", "model", "claude", "live:fixture"),),
    )
    assert (fallback.name, fallback.provider, fallback.model) == ("hermes", "fixture", "other")

    spawn = normalize_worker_call(codex, _advertised(codex, "spawn_agent"), {
        "message": "work", "profile": "safe", "provider": "fixture-2",
        "model": "model-2", "reasoning_effort": "high",
    })
    assert spawn.arguments == {
        "goal": "work", "profile": "safe", "provider": "fixture-2",
        "model": "model-2", "reasoning_effort": "high",
    }
    with pytest.raises(ValueError, match="Unsupported spawn_agent arguments"):
        normalize_worker_call(codex, _advertised(codex, "spawn_agent"), {
            "message": "work", "fork_parent_history": True,
        })

    claude = _selection("claude", definitions)
    claude_names = {
        item["function"]["name"] for item in project_worker_tool_definitions(definitions, claude)
    }
    assert "Agent" in claude_names
    assert "Task" not in claude_names


def test_configured_native_allowlist_cannot_grant_a_styled_worker_alias():
    from tools.delegate_tool_toolsets import _apply_exact_tool_policy

    definitions = [_schema(CANONICAL_WORKER_TOOL)]
    child = SimpleNamespace(
        _worker_interface_selection=_selection("codex", definitions),
        _executable_tool_names={CANONICAL_WORKER_TOOL},
        valid_tool_names={"send_message"},
        tools=[_schema("send_message")],
    )
    policy = SimpleNamespace(
        allowed_tools=("send_message",), allowed_mcp_tools=None,
        blocked_tools=(), allowed_toolsets=None,
    )
    _apply_exact_tool_policy(child, policy)
    assert child._worker_effective_tool_names == frozenset()
    assert child.valid_tool_names == set()


def _interface_call(selection, semantic, **values):
    if selection.name == "hermes":
        action = {
            "list": "status", "inspect": "inspect", "wait": "wait", "message": "message",
            "followup": "resume", "interrupt": "interrupt", "cancel_tree": "cancel",
            "completions": "completions",
            "ack": "ack", "reconcile": "reconcile",
        }[semantic]
        return CANONICAL_WORKER_TOOL, {"action": action, **values}
    if selection.name == "codex":
        names = {
            "list": "list_agents", "inspect": "inspect_agent", "wait": "wait_agent",
            "message": "send_message", "followup": "followup_task",
            "interrupt": "interrupt_agent", "cancel_tree": "cancel_agent_tree",
            "completions": "worker_control",
            "ack": "worker_control", "reconcile": "worker_control",
        }
        args = {
            "list": {},
            "inspect": {"target": values.get("worker_id"), "run_id": values.get("run_id")},
            "wait": {"target": values.get("worker_id"), "run_id": values.get("run_id"), "timeout_ms": 0},
            "message": {"target": values.get("worker_id"), "message": values.get("message")},
            "followup": {"target": values.get("worker_id"), "message": values.get("message")},
            "interrupt": {"target": values.get("worker_id"), "run_id": values.get("run_id")},
            "cancel_tree": {"target": values.get("worker_id")},
            "completions": {"action": "completions"},
            "ack": {"action": "ack", "target": values.get("worker_id"), "run_id": values.get("run_id")},
            "reconcile": {
                "action": "reconcile", "target": values.get("worker_id"),
                "run_id": values.get("run_id"), "disposition": values.get("reconciliation_disposition"),
                "note": values.get("message"),
            },
        }
        return _advertised(selection, names[semantic]), {
            key: value for key, value in args[semantic].items() if value is not None
        }
    names = {
        "list": "TaskList", "inspect": "TaskOutput", "wait": "TaskOutput",
        "message": "SendMessage", "followup": "SendMessage", "interrupt": "TaskStop",
        "cancel_tree": "TaskStop",
        "completions": "worker_control", "ack": "worker_control", "reconcile": "worker_control",
    }
    args = {
        "list": {},
        "inspect": {"task_id": values.get("worker_id"), "run_id": values.get("run_id")},
        "wait": {"task_id": values.get("worker_id"), "run_id": values.get("run_id"), "block": True, "timeout_ms": 0},
        "message": {"recipient": values.get("worker_id"), "content": values.get("message")},
        "followup": {"recipient": values.get("worker_id"), "content": values.get("message"), "if_idle": True},
        "interrupt": {"task_id": values.get("worker_id"), "run_id": values.get("run_id"), "scope": "run"},
        "cancel_tree": {"task_id": values.get("worker_id"), "scope": "tree"},
        "completions": {"action": "completions"},
        "ack": {"action": "ack", "target": values.get("worker_id"), "run_id": values.get("run_id")},
        "reconcile": {
            "action": "reconcile", "target": values.get("worker_id"),
            "run_id": values.get("run_id"), "disposition": values.get("reconciliation_disposition"),
            "note": values.get("message"),
        },
    }
    return _advertised(selection, names[semantic]), {
        key: value for key, value in args[semantic].items() if value is not None
    }


@pytest.mark.parametrize("interface", ["hermes", "codex", "claude"])
def test_interfaces_route_real_store_controls_with_ordering_and_authorization(
    interface, tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {})
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: {})
    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: None)
    monkeypatch.setattr(
        SubagentLifecycleService, "_schedule_owner",
        classmethod(lambda cls, *_args, **_kwargs: None),
    )

    db = SessionDB(tmp_path / "state.db")
    store = WorkerStore(db)
    store.ensure_schema()
    policy = {
        "role": "leaf", "route": {}, "profile_contract": None, "effective_tools": [],
        "launch_allowed_toolsets": None, "launch_blocked_tools": [],
    }
    worker = store.create_worker("owner", worker_id="target", policy=policy)
    first = store.enqueue_run(worker["worker_id"], "owner", goal="first")
    active = store.claim_run(first["run_id"], "owner")
    foreign = store.create_worker("other-owner", worker_id="foreign", policy=policy)
    foreign_run = store.enqueue_run(foreign["worker_id"], "other-owner", goal="private")

    complete_worker = store.create_worker("owner", worker_id="complete", policy=policy)
    complete_run = store.enqueue_run(complete_worker["worker_id"], "owner", goal="done")
    complete_active = store.claim_run(complete_run["run_id"], "owner")
    store.finish_run(
        complete_active["run_id"], "owner", complete_active["lease_token"],
        status="SUCCEEDED", result={"summary": "done"}, history=[],
    )

    definitions = [_schema("send_message"), _schema(CANONICAL_WORKER_TOOL)]
    selection = bind_worker_interface(
        InterfaceSelection(
            interface, "explicit", "stable" if interface == "hermes" else "experimental_unqualified",
            "fixture", "model",
        ),
        definitions,
    )
    parent = SimpleNamespace(
        session_id="owner", _session_db=db, _delegate_depth=0,
        provider="fixture", model="model", _worker_interface_selection=selection,
    )

    class ClosedChild:
        def close(self):
            return None

    monkeypatch.setattr(
        SubagentLifecycleService, "_build_revalidated_child",
        lambda self, current_worker, *, goal, role: (
            ClosedChild(), SimpleNamespace_get_creds(), {}, current_worker["policy"],
        ),
    )

    def dispatch(semantic, **values):
        name, args = _interface_call(selection, semantic, **values)
        return json.loads(dispatch_worker_interface_call(
            parent, name, args, lambda canonical: AIAgent._dispatch_delegate_task(parent, canonical)
        ))

    listed = dispatch("list")
    assert listed["success"] is True
    assert listed["orchestration_interface"]["worker_service"].endswith("SubagentLifecycleService")
    assert {item["worker_id"] for item in listed["workers"]} == {"target", "complete"}
    inspected = dispatch("inspect", worker_id="target", run_id=active["run_id"])
    assert inspected["run"]["run_id"] == active["run_id"]
    waited = dispatch("wait", worker_id="target", run_id=active["run_id"])
    assert waited["status"] == "RUNNING"

    guidance = dispatch("message", worker_id="target", message="guidance")
    assert guidance["delivery"] == "NEXT_RUN"
    assert len(store.list_runs("target", "owner")) == 1
    assert len(store.list_messages("target", "owner")) == 1

    followup = dispatch("followup", worker_id="target", message="new assignment")
    queued = store.get_run(followup["run_id"], "owner")
    assert queued["status"] == "PENDING"
    assert queued["previous_run_id"] == active["run_id"]
    assert queued["goal"] == "new assignment"
    assert followup["orchestration_interface"]["effective_action"] == "resume"

    interrupted = dispatch("interrupt", worker_id="target", run_id=queued["run_id"])
    assert interrupted["status"] == "INTERRUPTED"
    continued = dispatch("followup", worker_id="target", message="authorized after interrupt")
    continued_run = store.get_run(continued["run_id"], "owner")
    assert continued_run["previous_run_id"] == queued["run_id"]
    cancelled = dispatch("cancel_tree", worker_id="target")
    assert cancelled["cancel_requested"] is True
    rejected = dispatch("followup", worker_id="target", message="must not wake")
    assert "Cancelled workers stay cancelled" in rejected["error"]

    completions = dispatch("completions")
    assert any(item["run_id"] == complete_active["run_id"] for item in completions["completions"])
    acknowledged = dispatch(
        "ack", worker_id="complete", run_id=complete_active["run_id"]
    )
    assert acknowledged["acknowledged"] is True

    reconcile_worker = store.create_worker("owner", worker_id="reconcile", policy=policy)
    reconcile_run = store.enqueue_run(reconcile_worker["worker_id"], "owner", goal="effect")
    reconcile_active = store.claim_run(reconcile_run["run_id"], "owner")
    store.mark_tool_boundary(
        reconcile_active["run_id"], "owner", reconcile_active["lease_token"],
        tool_call_id="synthetic-effect",
    )
    store.finish_run(
        reconcile_active["run_id"], "owner", reconcile_active["lease_token"],
        status="SUCCEEDED", result={"summary": "uncertain"}, history=[],
    )
    reconciled = dispatch(
        "reconcile", worker_id="reconcile", run_id=reconcile_active["run_id"],
        reconciliation_disposition="accepted_unknown_no_replay",
        message="Synthetic outcome accepted without replay.",
    )
    assert reconciled["reconciled"] is True
    assert store.get_worker("reconcile", "owner")["uncertain_side_effect"] is False

    denied = dispatch("inspect", worker_id="foreign", run_id=foreign_run["run_id"])
    assert "foreign owner" in denied["error"].lower()
    db.close()


def SimpleNamespace_get_creds():
    return {
        "execution_limits": SimpleNamespace(
            timeout_seconds=None, max_tool_calls=None, max_followups=8,
        ),
        "max_iterations": 2,
    }
