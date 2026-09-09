"""Real-process acceptance for the integrated parent-managed team service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "team_process_worker.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _configure(home: Path) -> None:
    home.mkdir()
    (home / "config.yaml").write_text(
        """tools:
  tool_search:
    enabled: "off"
delegation:
  independent_completions: true
  max_concurrent_children: 2
  max_iterations: 4
  profiles:
    team-process:
      provider: openrouter
      model: fixture/model
      tool_policy:
        allowed_toolsets: [file]
      workspace_context:
        mode: none
      execution_limits:
        max_iterations: 4
        max_followups: 12
        max_tool_calls: 4
""",
        encoding="utf-8",
    )


def _env(
    home: Path,
    board: Path,
    *,
    phase: str,
    behavior: str = "final",
    marker: Path | None = None,
    capture: Path | None = None,
    effect: Path | None = None,
    events: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        if name.endswith(("_API_KEY", "_ACCESS_TOKEN", "_REFRESH_TOKEN")):
            env.pop(name, None)
    env.pop("HERMES_KANBAN_TASK", None)
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_DB"] = str(board)
    env["HERMES_KANBAN_BOARD"] = "default"
    env["OPENROUTER_API_KEY"] = "synthetic-team-process-placeholder"
    env["HERMES_TEAM_PHASE"] = phase
    env["HERMES_TEAM_BEHAVIOR"] = behavior
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for name, path in (
        ("HERMES_TEAM_MARKER", marker),
        ("HERMES_TEAM_CAPTURE", capture),
        ("HERMES_TEAM_EFFECT", effect),
        ("HERMES_TEAM_EVENTS", events),
    ):
        if path is not None:
            env[name] = str(path)
    return env


def _argv(home: Path, command: str, *, payload: dict | None = None, task_ref: str = "") -> list[str]:
    argv = [sys.executable, str(FIXTURE), command, "--home", str(home)]
    if payload is not None:
        argv.extend(("--payload", json.dumps(payload)))
    if task_ref:
        argv.extend(("--task-ref", task_ref))
    return argv


def _run(
    home: Path,
    board: Path,
    command: str,
    *,
    phase: str,
    payload: dict | None = None,
    task_ref: str = "",
    wait: bool = False,
    expire_leases: bool = False,
    capture: Path | None = None,
    effect: Path | None = None,
) -> dict:
    argv = _argv(home, command, payload=payload, task_ref=task_ref)
    if wait:
        argv.append("--wait")
    if expire_leases:
        argv.append("--expire-leases")
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=_env(home, board, phase=phase, capture=capture, effect=effect),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _start(
    home: Path,
    board: Path,
    payload: dict,
    *,
    phase: str,
    behavior: str,
    marker: Path,
    capture: Path,
    effect: Path,
    events: Path,
) -> subprocess.Popen:
    return subprocess.Popen(
        _argv(home, "dispatch", payload=payload) + ["--wait"],
        cwd=REPO_ROOT,
        env=_env(
            home, board, phase=phase, behavior=behavior, marker=marker,
            capture=capture, effect=effect, events=events,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_marker(marker: Path, process: subprocess.Popen, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"fixture exited before marker: {stdout}\n{stderr}")
        time.sleep(0.02)
    process.kill()
    process.wait(timeout=5)
    raise AssertionError("timed out waiting for subprocess barrier")


def _kill_after_marker(marker: Path, process: subprocess.Popen) -> None:
    _wait_for_marker(marker, process)
    process.kill()
    process.wait(timeout=5)
    assert process.returncode != 0


def _json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _assert_role_alternation(records: list[dict]) -> None:
    for record in records:
        roles = record["roles"]
        assert all(left != right for left, right in zip(roles, roles[1:]))


def test_attached_held_run_survives_real_process_death_and_executes_once(tmp_path):
    home, board = tmp_path / "profile", tmp_path / "kanban.db"
    capture, effect, events = (tmp_path / name for name in ("requests.jsonl", "effects.jsonl", "events.jsonl"))
    _configure(home)
    created = _run(
        home, board, "dispatch", phase="create",
        payload={
            "action": "create", "title": "implementation-context", "profile": "team-process",
            "idempotency_key": "restart-once",
        },
    )
    start = {"action": "start", "task_ref": created["task_ref"]}
    marker = tmp_path / "attached.ready"
    process = _start(
        home, board, start, phase="held-attachment", behavior="block-before-schedule",
        marker=marker, capture=capture, effect=effect, events=events,
    )
    _kill_after_marker(marker, process)

    held = _json_lines(events)
    assert len(held) == 1 and held[0]["task_ref"] == created["task_ref"]
    before = _run(home, board, "snapshot", phase="snapshot", task_ref=created["task_ref"])
    assert before["ownership_verified"] is True
    assert [(item["run_ref"], item["status"]) for item in before["attachments"]] == [
        (held[0]["run_ref"], "PENDING")
    ]

    resumed = _run(
        home, board, "dispatch", phase="implementation-resume", payload=start,
        wait=True, capture=capture, effect=effect,
    )
    assert resumed["recovered"] is True
    assert resumed["run_ref"] == held[0]["run_ref"]
    assert resumed["worker_ref"] == held[0]["worker_ref"]
    assert resumed["terminal"]["status"] == "SUCCEEDED"
    after = _run(home, board, "snapshot", phase="snapshot", task_ref=created["task_ref"])
    assert [item["status"] for item in after["attachments"]] == ["SUCCEEDED"]
    records = _json_lines(capture)
    assert [record["phase"] for record in records] == ["implementation-resume"]
    assert records[0]["user_markers"] == ["implementation-context"]
    assert _json_lines(effect) == []
    _assert_role_alternation(records)


def test_review_correction_restart_retains_context_and_never_replays_uncertain_effect(tmp_path):
    home, board = tmp_path / "profile", tmp_path / "kanban.db"
    capture, effect, events = (tmp_path / name for name in ("requests.jsonl", "effects.jsonl", "events.jsonl"))
    _configure(home)
    created = _run(
        home, board, "dispatch", phase="create",
        payload={"action": "create", "title": "implementation-context", "profile": "team-process"},
    )
    task_ref = created["task_ref"]
    implementation = _run(
        home, board, "dispatch", phase="implementation", payload={"action": "start", "task_ref": task_ref},
        wait=True, capture=capture, effect=effect,
    )
    assert implementation["terminal"]["status"] == "SUCCEEDED"
    assert _run(
        home, board, "dispatch", phase="submit-one",
        payload={"action": "submit_review", "task_ref": task_ref, "summary": "review-one", "reviewer": "team-process"},
    )["status"] == "review"
    first_review = _run(
        home, board, "dispatch", phase="review-one", payload={"action": "start", "task_ref": task_ref},
        wait=True, capture=capture, effect=effect,
    )
    assert first_review["terminal"]["status"] == "SUCCEEDED"

    marker = tmp_path / "correction-attached.ready"
    correction_process = _start(
        home, board,
        {"action": "request_changes", "task_ref": task_ref, "message": "correction-context"},
        phase="correction-held", behavior="block-before-schedule", marker=marker,
        capture=capture, effect=effect, events=events,
    )
    _kill_after_marker(marker, correction_process)
    correction_ref = _json_lines(events)[0]
    held = _run(home, board, "snapshot", phase="snapshot", task_ref=task_ref)
    assert [item["role"] for item in held["attachments"]] == [
        "implementer", "reviewer", "correction",
    ]
    correction = held["attachments"][-1]
    assert correction["status"] == "PENDING"
    assert correction["worker_ref"] == implementation["worker_ref"]
    assert correction["previous_run_ref"] == implementation["run_ref"]
    assert correction["run_ref"] == correction_ref["run_ref"]

    resumed = _run(
        home, board, "dispatch", phase="correction-resume",
        payload={"action": "start", "task_ref": task_ref}, wait=True,
        capture=capture, effect=effect,
    )
    assert resumed["recovered"] is True
    assert resumed["run_ref"] == correction["run_ref"]
    assert resumed["terminal"]["status"] == "SUCCEEDED"
    assert _run(
        home, board, "dispatch", phase="submit-two",
        payload={"action": "submit_review", "task_ref": task_ref, "summary": "review-two", "reviewer": "team-process"},
    )["status"] == "review"

    crash_marker = tmp_path / "uncertain-effect.ready"
    reviewer = _start(
        home, board, {"action": "start", "task_ref": task_ref},
        phase="review-two", behavior="tool-crash", marker=crash_marker,
        capture=capture, effect=effect, events=events,
    )
    _wait_for_marker(crash_marker, reviewer)
    reviewer.wait(timeout=5)
    assert reviewer.returncode == 91
    assert len(_json_lines(effect)) == 1

    blocked = _run(
        home, board, "dispatch", phase="uncertain-restart",
        payload={"action": "start", "task_ref": task_ref}, expire_leases=True,
        capture=capture, effect=effect,
    )
    assert "Reconcile uncertain worker effects before scheduling" in blocked["error"]
    repeated = _run(
        home, board, "dispatch", phase="uncertain-repeat",
        payload={"action": "start", "task_ref": task_ref}, capture=capture, effect=effect,
    )
    assert "Reconcile uncertain worker effects before scheduling" in repeated["error"]
    assert len(_json_lines(effect)) == 1

    final = _run(home, board, "snapshot", phase="snapshot", task_ref=task_ref)
    assert final["ownership_verified"] is True
    assert [item["role"] for item in final["attachments"]] == [
        "implementer", "reviewer", "correction", "reviewer",
    ]
    assert final["attachments"][-1]["status"] == "INTERRUPTED"
    assert final["attachments"][-1]["uncertain_side_effect"] is True
    records = _json_lines(capture)
    correction_records = [record for record in records if record["phase"] == "correction-resume"]
    assert len(correction_records) == 1
    assert correction_records[0]["user_markers"] == ["implementation-context", "correction-context"]
    assert len({record["pid"] for record in records}) == len(records)
    _assert_role_alternation(records)


def test_saved_workflow_pause_branch_join_and_review_continue_across_process_restarts(tmp_path):
    home, board = tmp_path / "profile", tmp_path / "kanban.db"
    capture, effect, events = (
        tmp_path / name for name in ("requests.jsonl", "effects.jsonl", "events.jsonl")
    )
    _configure(home)
    definition = {
        "name": "Process workflow",
        "steps": [
            {
                "key": "left", "title": "workflow-left", "profile": "team-process",
                "reviewer": "team-process", "max_corrections": 1,
            },
            {
                "key": "right", "title": "workflow-right", "profile": "team-process",
                "reviewer": "team-process", "max_corrections": 1,
            },
            {
                "key": "join", "title": "workflow-join", "profile": "team-process",
                "reviewer": "team-process", "depends_on": ["left", "right"],
                "max_corrections": 1,
            },
        ],
    }
    saved = _run(
        home, board, "dispatch", phase="workflow-save",
        payload={"action": "workflow_save", "definition": definition},
    )
    assert saved["version"] == 1
    invoke = {
        "action": "workflow_invoke",
        "template_ref": saved["template_ref"],
        "admission_key": "process-workflow-one",
        "input": {"fixture": "finite"},
    }

    invoke_marker = tmp_path / "workflow-branch-attached.ready"
    invoking = _start(
        home, board, invoke, phase="workflow-invoke-held", behavior="block-before-schedule",
        marker=invoke_marker, capture=capture, effect=effect, events=events,
    )
    _kill_after_marker(invoke_marker, invoking)
    initial_attachment = _json_lines(events)[0]

    listed = _run(
        home, board, "dispatch", phase="workflow-list",
        payload={"action": "workflow_list"},
    )
    assert [item["template_ref"] for item in listed["templates"]] == [saved["template_ref"]]
    assert len(listed["invocations"]) == 1
    initial = listed["invocations"][0]
    workflow_ref = initial["workflow_ref"]
    task_refs = {item["step_key"]: item["task_ref"] for item in initial["steps"]}
    assert initial_attachment["task_ref"] in {task_refs["left"], task_refs["right"]}
    held_key = next(
        key for key in ("left", "right") if task_refs[key] == initial_attachment["task_ref"]
    )
    other_key = "right" if held_key == "left" else "left"
    held_marker = f"workflow-{held_key}"
    assert initial["template_ref"] == saved["template_ref"]
    assert {item["step_key"]: item["status"] for item in initial["steps"]} == {
        held_key: "running", other_key: "ready", "join": "todo",
    }

    paused = _run(
        home, board, "dispatch", phase="workflow-pause",
        payload={
            "action": "workflow_pause", "workflow_ref": workflow_ref, "expected_version": 1,
        },
    )
    assert paused["control_state"] == "paused"
    assert paused["control_version"] == 2
    repeated = _run(
        home, board, "dispatch", phase="workflow-paused-reinvoke", payload=invoke,
    )
    assert repeated["workflow_ref"] == workflow_ref
    assert repeated["advancement"]["outcomes"] == []
    assert repeated["advancement"]["workflow"]["control_state"] == "paused"
    assert {
        item["step_key"]: item["task_ref"]
        for item in repeated["advancement"]["workflow"]["steps"]
    } == task_refs
    held_branch = _run(
        home, board, "snapshot", phase="workflow-paused-snapshot", task_ref=task_refs[held_key],
    )
    unclaimed_other = _run(
        home, board, "snapshot", phase="workflow-paused-snapshot", task_ref=task_refs[other_key],
    )
    assert [(item["run_ref"], item["status"]) for item in held_branch["attachments"]] == [
        (initial_attachment["run_ref"], "PENDING")
    ]
    assert unclaimed_other["attachments"] == []

    resumed = _run(
        home, board, "dispatch", phase="workflow-resume",
        payload={
            "action": "workflow_resume", "workflow_ref": workflow_ref, "expected_version": 2,
        },
        wait=True, capture=capture, effect=effect,
    )
    assert resumed["workflow"]["control_state"] == "active"
    assert resumed["workflow"]["control_version"] == 3
    branch_outcomes = {item["task_ref"]: item for item in resumed["outcomes"]}
    assert branch_outcomes[task_refs[held_key]]["run_ref"] == initial_attachment["run_ref"]
    assert branch_outcomes[task_refs[held_key]]["recovered"] is True
    assert branch_outcomes[task_refs[held_key]]["terminal"]["status"] == "SUCCEEDED"
    assert branch_outcomes[task_refs[other_key]]["terminal"]["status"] == "SUCCEEDED"

    assert _run(
        home, board, "dispatch", phase="workflow-left-submit-one",
        payload={
            "action": "submit_review", "task_ref": task_refs[held_key],
            "summary": f"{held_marker} review-one", "reviewer": "team-process",
        },
    )["status"] == "review"
    reviewer_marker = tmp_path / "workflow-left-review-attached.ready"
    held_reviewer = _start(
        home, board, {"action": "start", "task_ref": task_refs[held_key]},
        phase="workflow-left-review-held", behavior="block-before-schedule",
        marker=reviewer_marker, capture=capture, effect=effect, events=events,
    )
    _kill_after_marker(reviewer_marker, held_reviewer)
    reviewer_attachment = _json_lines(events)[-1]
    restarted_reviewer = _run(
        home, board, "dispatch", phase="workflow-left-review-resume",
        payload={"action": "start", "task_ref": task_refs[held_key]},
        wait=True, capture=capture, effect=effect,
    )
    assert restarted_reviewer["run_ref"] == reviewer_attachment["run_ref"]
    assert restarted_reviewer["terminal"]["status"] == "SUCCEEDED"

    correction_marker = tmp_path / "workflow-left-correction-attached.ready"
    held_correction = _start(
        home, board,
        {
            "action": "request_changes", "task_ref": task_refs[held_key],
            "message": "workflow-correction",
        },
        phase="workflow-left-correction-held", behavior="block-before-schedule",
        marker=correction_marker, capture=capture, effect=effect, events=events,
    )
    _kill_after_marker(correction_marker, held_correction)
    correction_attachment = _json_lines(events)[-1]
    held_durable = _run(
        home, board, "snapshot", phase="workflow-left-snapshot", task_ref=task_refs[held_key],
    )
    assert [item["role"] for item in held_durable["attachments"]] == [
        "implementer", "reviewer", "correction",
    ]
    assert held_durable["attachments"][-1]["worker_ref"] == initial_attachment["worker_ref"]
    assert held_durable["attachments"][-1]["previous_run_ref"] == initial_attachment["run_ref"]
    assert held_durable["attachments"][-1]["run_ref"] == correction_attachment["run_ref"]
    restarted_correction = _run(
        home, board, "dispatch", phase="workflow-left-correction-resume",
        payload={"action": "start", "task_ref": task_refs[held_key]},
        wait=True, capture=capture, effect=effect,
    )
    assert restarted_correction["run_ref"] == correction_attachment["run_ref"]
    assert restarted_correction["terminal"]["status"] == "SUCCEEDED"

    assert _run(
        home, board, "dispatch", phase="workflow-left-submit-two",
        payload={
            "action": "submit_review", "task_ref": task_refs[held_key],
            "summary": f"{held_marker} correction ready", "reviewer": "team-process",
        },
    )["status"] == "review"
    held_second_review = _run(
        home, board, "dispatch", phase="workflow-left-review-two",
        payload={"action": "start", "task_ref": task_refs[held_key]},
        wait=True, capture=capture, effect=effect,
    )
    assert held_second_review["terminal"]["status"] == "SUCCEEDED"
    assert _run(
        home, board, "dispatch", phase="workflow-left-accept",
        payload={"action": "accept", "task_ref": task_refs[held_key], "summary": "held accepted"},
    )["status"] == "done"

    assert _run(
        home, board, "dispatch", phase="workflow-right-submit",
        payload={
            "action": "submit_review", "task_ref": task_refs[other_key],
            "summary": f"workflow-{other_key} ready", "reviewer": "team-process",
        },
    )["status"] == "review"
    other_review = _run(
        home, board, "dispatch", phase="workflow-right-review",
        payload={"action": "start", "task_ref": task_refs[other_key]},
        wait=True, capture=capture, effect=effect,
    )
    assert other_review["terminal"]["status"] == "SUCCEEDED"
    assert _run(
        home, board, "dispatch", phase="workflow-right-accept",
        payload={"action": "accept", "task_ref": task_refs[other_key], "summary": "other accepted"},
    )["status"] == "done"

    join_started = _run(
        home, board, "dispatch", phase="workflow-join",
        payload={
            "action": "workflow_resume", "workflow_ref": workflow_ref, "expected_version": 3,
        },
        wait=True, capture=capture, effect=effect,
    )
    join = next(item for item in join_started["outcomes"] if item["task_ref"] == task_refs["join"])
    assert join["terminal"]["status"] == "SUCCEEDED"
    assert _run(
        home, board, "dispatch", phase="workflow-join-submit",
        payload={
            "action": "submit_review", "task_ref": task_refs["join"],
            "summary": "workflow-join ready", "reviewer": "team-process",
        },
    )["status"] == "review"
    join_review = _run(
        home, board, "dispatch", phase="workflow-join-review",
        payload={"action": "start", "task_ref": task_refs["join"]},
        wait=True, capture=capture, effect=effect,
    )
    assert join_review["terminal"]["status"] == "SUCCEEDED"
    accepted = _run(
        home, board, "dispatch", phase="workflow-join-accept",
        payload={"action": "accept", "task_ref": task_refs["join"], "summary": "join accepted"},
    )
    assert accepted["workflow_completed"] is True

    final = _run(
        home, board, "dispatch", phase="workflow-final",
        payload={"action": "workflow_inspect", "workflow_ref": workflow_ref},
    )
    assert final["workflow_ref"] == workflow_ref
    assert final["template_ref"] == saved["template_ref"]
    assert {item["step_key"]: item["task_ref"] for item in final["steps"]} == task_refs
    assert final["completed"] is True
    assert {item["status"] for item in final["steps"]} == {"done"}
    held_final = _run(
        home, board, "snapshot", phase="workflow-left-final", task_ref=task_refs[held_key],
    )
    assert [item["role"] for item in held_final["attachments"]] == [
        "implementer", "reviewer", "correction", "reviewer",
    ]
    assert len({item["run_ref"] for item in held_final["attachments"]}) == 4
    assert {item["status"] for item in held_final["attachments"]} == {"SUCCEEDED"}
    records = _json_lines(capture)
    correction_records = [
        item for item in records if item["phase"] == "workflow-left-correction-resume"
    ]
    assert len(correction_records) == 1
    assert correction_records[0]["user_markers"] == [held_marker, "workflow-correction"]
    assert _json_lines(effect) == []
    _assert_role_alternation(records)
