"""The project runner uses the picker, parent inference, real worker process, and host checks."""

import json
import sys
from types import SimpleNamespace

import pytest

from agent.engineering_execution import CheckSpec
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
    assert result.stage_calls == 3


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
