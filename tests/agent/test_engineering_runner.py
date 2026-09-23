"""The project runner uses the picker, parent inference, real worker process, and host checks."""

import json
import sys
from types import SimpleNamespace

import pytest

from agent.engineering_execution import CheckSpec
from agent.engineering_workflow import VerificationReceipt
from agent.engineering_runner import (
    WorkerActionError,
    parse_worker_action,
    run_project_workflow,
)


ROUTES = {
    stage: {"provider": "provider-a", "model": model}
    for stage, model in (
        ("planner", "plan"),
        ("worker", "work"),
        ("reviewer", "review"),
    )
}
CATALOGUE = {
    "providers": [
        {
            "slug": "provider-a",
            "models": ["plan", "work", "review"],
            "authenticated": True,
        }
    ]
}
PLAN = json.dumps({
    "objective": "Repair module",
    "constraints": ["Keep the public function"],
    "steps": ["Inspect and edit the module"],
    "acceptance_criteria": ["The host check succeeds"],
})


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"RUN","status":"READY","argv":["python"],"summary":"x"}',
        '{"status":"PASS","summary":"done"}',
        '{"status":"RUN","argv":"python -V","summary":"x"}',
        '{"status":"READY","summary":"done","api_key":"secret"}',
        '{"status":"BLOCKED","summary":"x"}',
        '{"status":[],"summary":"x"}',
        '{"status":"RUN","argv":["cli","--token=leak"],"summary":"x"}',
        "{" + " " * 20_000 + "}",
    ],
)
def test_worker_action_rejects_ambiguous_commands_and_statuses(payload):
    with pytest.raises(WorkerActionError):
        parse_worker_action(payload)


def test_project_runner_uses_real_native_process_and_bound_host_check(
    tmp_path, monkeypatch
):
    target = tmp_path / "module.py"
    target.write_text("value = 'broken'\n", encoding="utf-8")
    calls = []
    worker_turns = 0

    def fake_call_llm(**kwargs):
        nonlocal worker_turns
        calls.append(kwargs)
        kwargs["route_info"].update(
            provider=kwargs["provider"],
            model=kwargs["model"],
        )
        if kwargs["model"] == "plan":
            text = PLAN
        elif kwargs["model"] == "work":
            worker_turns += 1
            if worker_turns == 1:
                text = json.dumps({
                    "status": "RUN",
                    "argv": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('module.py').write_text("
                        "\"value = 'fixed'\\n\", encoding='utf-8')",
                    ],
                    "summary": "Edit module",
                })
            else:
                text = json.dumps({"status": "READY", "summary": "Module edited"})
        else:
            raise AssertionError("A reviewer should not run after a passing check")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    monkeypatch.setattr("agent.engineering_runner.call_llm", fake_call_llm)
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: object())
    monkeypatch.setattr(
        "hermes_cli.inventory.build_model_options_payload",
        lambda *args, **kwargs: CATALOGUE,
    )
    result = run_project_workflow(
        objective="Repair module",
        assignments=ROUTES,
        workspace=tmp_path,
        backend="native",
        checks=(
            CheckSpec(
                "unit",
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; assert Path('module.py').read_text(encoding='utf-8') "
                    "== \"value = 'fixed'\\n\"",
                ),
                20,
            ),
        ),
    )
    assert result.status == "DONE"
    assert target.read_text(encoding="utf-8") == "value = 'fixed'\n"
    assert [call["model"] for call in calls] == ["plan", "work", "work"]
    assert all(call["strict_route"] is True for call in calls)
    assert all("reasoning_config" not in call for call in calls)
    assert result.stage_calls == 3


def test_project_runner_forwards_each_stage_reasoning_to_parent_inference(tmp_path, monkeypatch):
    assignments = {
        stage: {**route, "reasoning_effort": effort}
        for stage, route, effort in (
            ("planner", ROUTES["planner"], "medium"),
            ("worker", ROUTES["worker"], "medium"),
            ("reviewer", ROUTES["reviewer"], "high"),
        )
    }
    calls = []
    checks_run = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        kwargs["route_info"].update(provider=kwargs["provider"], model=kwargs["model"])
        text = (
            json.dumps({"status": "READY", "summary": "Ready for host check"})
            if kwargs["model"] == "work" else PLAN
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def fake_checks(context, *args, **kwargs):
        checks_run.append(context)
        return [VerificationReceipt(
            run_id=context.run_id,
            workspace_id=context.workspace_id,
            attempt_id=context.attempt_id,
            revision=context.revision,
            snapshot_digest=context.snapshot_digest,
            check_id="unit",
            exit_code=1 if len(checks_run) == 1 else 0,
            complete=True,
            timed_out=False,
        )]

    monkeypatch.setattr("agent.engineering_runner.call_llm", fake_call_llm)
    monkeypatch.setattr("agent.engineering_runner.execute_checks", fake_checks)
    monkeypatch.setattr("agent.engineering_runner.workspace_digest", lambda root: "stable-digest")
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: object())
    monkeypatch.setattr(
        "hermes_cli.inventory.build_model_options_payload",
        lambda *args, **kwargs: CATALOGUE,
    )
    result = run_project_workflow(
        objective="Repair module",
        assignments=assignments,
        workspace=tmp_path,
        backend="native",
        checks=(CheckSpec("unit", (sys.executable, "-V"), 20),),
    )
    assert result.status == "DONE"
    assert [call["model"] for call in calls] == ["plan", "work", "review", "work"]
    assert [call["reasoning_config"] for call in calls] == [
        {"enabled": True, "effort": "medium"},
        {"enabled": True, "effort": "medium"},
        {"enabled": True, "effort": "high"},
        {"enabled": True, "effort": "medium"},
    ]
    assert all(call["strict_route"] is True for call in calls)
    assert len(checks_run) == 2

@pytest.mark.parametrize(
    "worker_text, expected_reason, host_calls",
    [
        ("{malformed", "actor_protocol_error", 0),
        (
            json.dumps({"status": "RUN", "summary": "Run one command", "argv": ["noop"]}),
            "execution_boundary_failed", 1,
        ),
    ],
)
def test_worker_protocol_and_incomplete_execution_have_distinct_safe_codes(
    tmp_path, monkeypatch, worker_text, expected_reason, host_calls
):
    calls = []

    def fake_call_llm(**kwargs):
        kwargs["route_info"].update(provider=kwargs["provider"], model=kwargs["model"])
        text = PLAN if kwargs["model"] == "plan" else worker_text
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def fake_host(*args, **kwargs):
        calls.append("writer")
        return SimpleNamespace(complete=False, timed_out=False)

    monkeypatch.setattr("agent.engineering_runner.call_llm", fake_call_llm)
    monkeypatch.setattr("agent.engineering_runner.run_host_command", fake_host)
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: object())
    monkeypatch.setattr(
        "hermes_cli.inventory.build_model_options_payload",
        lambda *args, **kwargs: CATALOGUE,
    )
    result = run_project_workflow(
        objective="Repair module",
        assignments=ROUTES,
        workspace=tmp_path,
        backend="native",
        checks=(CheckSpec("unit", (sys.executable, "-V"), 20),),
    )
    assert result.status == "BLOCKED"
    assert result.reason == expected_reason
    assert len(calls) == host_calls

def test_worker_action_rejects_parent_credential_in_argv(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_PROVIDER_TOKEN", "synthetic-secret-123")
    with pytest.raises(WorkerActionError):
        parse_worker_action(
            json.dumps({
                "status": "RUN",
                "summary": "run",
                "argv": ["tool", "synthetic-secret-123"],
            })
        )
