import json
from pathlib import Path
from types import SimpleNamespace

from agent.orchestration_approvals import ApprovalInterrupt, approval_required, resume_after_approval
from agent.orchestration_capabilities import Capability, choose_delegate_route
from agent.orchestration_hooks import HookRunner
from agent.orchestration_trace import OrchestrationTrace
from agent.workflow_runner import WorkflowSpec, compile_workflow_to_kanban_tasks


def test_trace_writes_jsonl_events(tmp_path):
    trace = OrchestrationTrace.start("sess-1", root_dir=tmp_path, workflow_name="research")
    trace.record("delegate_spawn", child="web", task="scout")
    trace.record("final", ok=True)

    lines = trace.path.read_text().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["event"] == "run_start"
    assert first["run_id"] == trace.run_id
    assert second["event"] == "delegate_spawn"
    assert second["child"] == "web"


def test_capability_router_skips_small_context_and_picks_eligible():
    parent = Capability(name="parent", provider="openai-codex", model="gpt-5.5", context_length=272000)
    candidates = [
        Capability(name="local-qwen", provider="bolt", model="qwen2.5-coder:32b", context_length=32768),
        Capability(name="sonnet", provider="anthropic", model="claude-sonnet-4.6", context_length=200000),
    ]

    route = choose_delegate_route(parent, candidates, required_context=64000, required_toolsets=["terminal"])

    assert route.selected.name == "sonnet"
    assert route.skipped[0].name == "local-qwen"
    assert "context" in route.skipped[0].reason


def test_workflow_yaml_compiles_to_kanban_task_specs():
    spec = WorkflowSpec.from_yaml(
        """
        name: code-change
        nodes:
          inspect:
            assignee: peewee
            body: Inspect repo
          implement:
            assignee: ghost
            depends_on: [inspect]
          review:
            assignee: patch
            depends_on: [implement]
        """
    )

    tasks = compile_workflow_to_kanban_tasks(spec, parent_task_id="ROOT")

    assert [t.title for t in tasks] == ["code-change: inspect", "code-change: implement", "code-change: review"]
    assert tasks[0].parents == ["ROOT"]
    assert tasks[1].parent_node_names == ["inspect"]
    assert tasks[2].assignee == "patch"


def test_approval_interrupt_can_resume_with_decision():
    interrupt = approval_required(
        action="deploy",
        reason="Production deploy requires human approval",
        payload={"service": "gateway"},
    )

    assert isinstance(interrupt, ApprovalInterrupt)
    assert interrupt.status == "pending_approval"
    resumed = resume_after_approval(interrupt, approved=True, approver="landon")
    assert resumed["approved"] is True
    assert resumed["approver"] == "landon"
    assert resumed["action"] == "deploy"


def test_hook_runner_invokes_registered_callable_and_records_result():
    calls = []

    def hook(event, payload):
        calls.append((event, payload["run_id"]))
        return {"ok": True}

    runner = HookRunner()
    runner.register("after_delegate_result", hook)
    results = runner.run("after_delegate_result", {"run_id": "run-1"})

    assert calls == [("after_delegate_result", "run-1")]
    assert results[0].ok is True
    assert results[0].output == {"ok": True}
