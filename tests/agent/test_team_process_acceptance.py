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
