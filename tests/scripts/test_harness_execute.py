import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import harness_execute as hx


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture
def harness_project(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Agent Rules\n- keep changes small", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("# Claude Rules\n- run tests", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ARCHITECTURE.md").write_text("# Architecture\nCore loop", encoding="utf-8")
    (docs / "ignore.txt").write_text("not markdown", encoding="utf-8")

    write_json(tmp_path / "phases" / "index.json", {"phases": [{"dir": "0-mvp", "status": "pending"}]})
    write_json(
        tmp_path / "phases" / "0-mvp" / "index.json",
        {
            "project": "Hermes Agent",
            "phase": "mvp",
            "steps": [
                {"step": 0, "name": "setup", "status": "completed", "summary": "scaffolded"},
                {"step": 1, "name": "feature", "status": "pending"},
            ],
        },
    )
    (tmp_path / "phases" / "0-mvp" / "step1.md").write_text(
        "# Step 1\n\nImplement the feature.\n\n## Acceptance Criteria\n- tests pass",
        encoding="utf-8",
    )
    return tmp_path


def test_load_guardrails_prefers_repo_context_and_markdown_docs(harness_project):
    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent")

    guardrails = executor.load_guardrails()

    assert "## AGENTS.md" in guardrails
    assert "keep changes small" in guardrails
    assert "## CLAUDE.md" in guardrails
    assert "run tests" in guardrails
    assert "## docs/ARCHITECTURE.md" in guardrails
    assert "Core loop" in guardrails
    assert "ignore.txt" not in guardrails


def test_build_prompt_includes_previous_step_context_and_step_file(harness_project):
    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent")
    index = executor.read_phase_index()
    step = index["steps"][1]

    prompt = executor.build_prompt(step, index)

    assert "Hermes Agent" in prompt
    assert "Step 0 (setup): scaffolded" in prompt
    assert "Implement the feature" in prompt
    assert "phases/0-mvp/index.json" in prompt
    assert 'status' in prompt and 'completed' in prompt


def test_run_invokes_agent_and_marks_completed_step_with_commit(harness_project):
    calls = []

    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        if cmd[:3] == ["git", "status", "--porcelain"]:
            idx_path = harness_project / "phases" / "0-mvp" / "index.json"
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            if data["steps"][1]["status"] == "completed":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=" M phases/0-mvp/index.json\n?? phases/0-mvp/step1-output.json\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "agent":
            idx_path = harness_project / "phases" / "0-mvp" / "index.json"
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            data["steps"][1]["status"] = "completed"
            data["steps"][1]["summary"] = "feature implemented"
            idx_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout='{"ok": true}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent")
    with patch.object(hx.subprocess, "run", side_effect=fake_run):
        executor.run()

    index = json.loads((harness_project / "phases" / "0-mvp" / "index.json").read_text(encoding="utf-8"))
    top = json.loads((harness_project / "phases" / "index.json").read_text(encoding="utf-8"))
    assert index["steps"][1]["status"] == "completed"
    assert "completed_at" in index["steps"][1]
    assert index["completed_at"]
    assert top["phases"][0]["status"] == "completed"
    assert any(cmd[0] == "agent" and "Acceptance Criteria" in cmd[-1] for cmd in calls)
    assert any(cmd[:2] == ["git", "commit"] for cmd in calls)


def test_run_records_error_after_agent_does_not_complete_step(harness_project):
    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent", max_retries=1)
    with patch.object(hx.subprocess, "run", side_effect=fake_run), pytest.raises(SystemExit) as exc:
        executor.run()

    assert exc.value.code == 1
    index = json.loads((harness_project / "phases" / "0-mvp" / "index.json").read_text(encoding="utf-8"))
    top = json.loads((harness_project / "phases" / "index.json").read_text(encoding="utf-8"))
    assert index["steps"][1]["status"] == "error"
    assert "did not update" in index["steps"][1]["error_message"]
    assert top["phases"][0]["status"] == "error"



def test_rejects_dangerous_agent_command_by_default(harness_project):
    with pytest.raises(SystemExit) as exc:
        hx.HarnessExecutor(harness_project, "0-mvp", agent_command="codex exec --full-auto")

    assert exc.value.code == 2


def test_run_refuses_dirty_worktree_before_checkout(harness_project):
    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M unrelated.py\n", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent")
    with patch.object(hx.subprocess, "run", side_effect=fake_run), pytest.raises(SystemExit) as exc:
        executor.run()

    assert exc.value.code == 2


def test_commit_step_stages_only_allowed_paths(harness_project):
    calls = []

    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent")
    with patch.object(hx.subprocess, "run", side_effect=fake_run):
        executor.commit_step(1, "feature", "mvp")

    assert ["git", "add", "-A"] not in calls
    add_calls = [cmd for cmd in calls if cmd[:2] == ["git", "add"]]
    assert add_calls
    flattened = "\n".join(" ".join(cmd) for cmd in add_calls)
    assert "phases/0-mvp" in flattened
    assert "unrelated.py" not in flattened


def test_rejects_non_positive_status_interval(harness_project):
    with pytest.raises(SystemExit) as exc:
        hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent", status_interval=0)

    assert exc.value.code == 2


def test_push_requires_second_confirmation(harness_project):
    calls = []

    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="feat-mvp\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    idx_path = harness_project / "phases" / "0-mvp" / "index.json"
    data = json.loads(idx_path.read_text(encoding="utf-8"))
    data["steps"][1]["status"] = "completed"
    idx_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent", auto_push=True)
    with patch.object(hx.subprocess, "run", side_effect=fake_run), pytest.raises(SystemExit) as exc:
        executor.run()

    assert exc.value.code == 2
    assert not any(cmd[:2] == ["git", "push"] for cmd in calls)


def test_status_snapshot_updates_on_start_and_completion(harness_project):
    calls = []

    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "agent":
            idx_path = harness_project / "phases" / "0-mvp" / "index.json"
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            data["steps"][1]["status"] = "completed"
            data["steps"][1]["summary"] = "feature implemented"
            idx_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout='secret token=abc123456789', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent", status_interval=0.01)
    with patch.object(hx.subprocess, "run", side_effect=fake_run):
        executor.run()

    snapshot = json.loads((harness_project / "phases" / "0-mvp" / "status.json").read_text(encoding="utf-8"))
    output = json.loads((harness_project / "phases" / "0-mvp" / "step1-output.json").read_text(encoding="utf-8"))
    assert snapshot["status"] == "completed"
    assert snapshot["current_step"] == 1
    assert snapshot["completed_steps"] == 2
    assert snapshot["total_steps"] == 2
    assert "stdout" not in output
    assert "abc123456789" not in json.dumps(output)



def test_run_rejects_allowed_paths_mutation_by_agent(harness_project):
    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "agent":
            idx_path = harness_project / "phases" / "0-mvp" / "index.json"
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            data["allowed_paths"] = ["phases/0-mvp/", "unrelated.py"]
            data["steps"][1]["status"] = "completed"
            data["steps"][1]["summary"] = "mutated allowlist"
            idx_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent")
    with patch.object(hx.subprocess, "run", side_effect=fake_run), pytest.raises(SystemExit) as exc:
        executor.run()

    assert exc.value.code == 1



def test_run_rejects_out_of_scope_agent_changes(harness_project):
    status_calls = 0

    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        nonlocal status_calls
        if cmd[:3] == ["git", "status", "--porcelain"]:
            status_calls += 1
            stdout = "" if status_calls == 1 else " M unrelated.py\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "agent":
            idx_path = harness_project / "phases" / "0-mvp" / "index.json"
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            data["steps"][1]["status"] = "completed"
            data["steps"][1]["summary"] = "changed unrelated file"
            idx_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent")
    with patch.object(hx.subprocess, "run", side_effect=fake_run), pytest.raises(SystemExit) as exc:
        executor.run()

    assert exc.value.code == 1



def test_run_rejects_rename_from_out_of_scope_into_allowed_path(harness_project):
    status_calls = 0

    def fake_run(cmd, cwd, capture_output=True, text=True, timeout=None):
        nonlocal status_calls
        if cmd[:3] == ["git", "status", "--porcelain"]:
            status_calls += 1
            stdout = "" if status_calls == 1 else "R  unrelated.py -> phases/0-mvp/new.py\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "agent":
            idx_path = harness_project / "phases" / "0-mvp" / "index.json"
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            data["steps"][1]["status"] = "completed"
            data["steps"][1]["summary"] = "renamed file"
            idx_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = hx.HarnessExecutor(harness_project, "0-mvp", agent_command="agent")
    with patch.object(hx.subprocess, "run", side_effect=fake_run), pytest.raises(SystemExit) as exc:
        executor.run()

    assert exc.value.code == 1
