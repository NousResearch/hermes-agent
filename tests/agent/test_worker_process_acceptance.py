"""Real-process acceptance for durable worker crash and restart boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "worker_process_acceptance.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _configure(home: Path) -> None:
    home.mkdir()
    (home / "config.yaml").write_text(
        # Native tool crash boundaries are exercised here; tool-search bridge
        # dispatch is covered separately by the enforcement acceptance tests.
        """tools:
  tool_search:
    enabled: "off"
delegation:
  independent_completions: true
  max_concurrent_children: 2
  max_iterations: 4
  profiles:
    process-acceptance:
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
    *,
    behavior: str = "final",
    credential: bool = True,
    marker: Path | None = None,
    capture: Path | None = None,
    effect: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        if name.endswith(("_API_KEY", "_ACCESS_TOKEN", "_REFRESH_TOKEN")):
            env.pop(name, None)
    env["HERMES_HOME"] = str(home)
    env["HERMES_ACCEPTANCE_BEHAVIOR"] = behavior
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if credential:
        env["OPENROUTER_API_KEY"] = "synthetic-acceptance-placeholder"
    if marker is not None:
        env["HERMES_ACCEPTANCE_MARKER"] = str(marker)
    if capture is not None:
        env["HERMES_ACCEPTANCE_CAPTURE"] = str(capture)
    if effect is not None:
        env["HERMES_ACCEPTANCE_EFFECT"] = str(effect)
    return env


def _run(
    home: Path,
    command: str,
    *args: str,
    behavior: str = "final",
    credential: bool = True,
    capture: Path | None = None,
    effect: Path | None = None,
) -> dict:
    completed = subprocess.run(
        [sys.executable, str(FIXTURE), command, "--home", str(home), *args],
        cwd=REPO_ROOT,
        env=_env(home, behavior=behavior, credential=credential, capture=capture, effect=effect),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _start(
    home: Path,
    command: str,
    *args: str,
    behavior: str,
    marker: Path,
    capture: Path,
    effect: Path | None = None,
) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(FIXTURE), command, "--home", str(home), *args],
        cwd=REPO_ROOT,
        env=_env(home, behavior=behavior, marker=marker, capture=capture, effect=effect),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for(
    path: Path, process: subprocess.Popen, timeout: float = 10, diagnostics: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            detail = _json_lines(diagnostics) if diagnostics is not None else []
            raise AssertionError(
                f"fixture exited before fault marker: {stdout}\n{stderr}\nrequest diagnostics: {detail}"
            )
        time.sleep(0.02)
    process.kill()
    process.wait(timeout=5)
    raise AssertionError("timed out waiting for subprocess fault marker")


def _kill_after_marker(process: subprocess.Popen, marker: Path) -> None:
    _wait_for(marker, process)
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


def test_enqueue_and_leased_launch_survive_process_restart_fifo_once(tmp_path):
    home, capture = tmp_path / "profile", tmp_path / "requests.jsonl"
    _configure(home)
    seed = _run(home, "seed", capture=capture)
    first = _run(
        home, "enqueue", "--worker-id", seed["worker_id"], "--run-id", seed["run_id"],
        "--message", "enqueue-one",
    )
    second = _run(
        home, "enqueue", "--worker-id", seed["worker_id"], "--run-id", first["run_id"],
        "--message", "enqueue-two",
    )

    marker = tmp_path / "leased.ready"
    process = _start(
        home, "control", "--action", "wait", "--worker-id", seed["worker_id"],
        "--run-id", second["run_id"], "--timeout", "20",
        behavior="block-before-checkpoint", marker=marker, capture=capture,
    )
    _kill_after_marker(process, marker)

    terminal = _run(
        home, "control", "--action", "wait", "--worker-id", seed["worker_id"],
        "--run-id", second["run_id"], "--timeout", "20", "--expire-leases",
        capture=capture,
    )
    assert terminal["status"] == "SUCCEEDED"
    snapshot = _run(home, "snapshot", "--worker-id", seed["worker_id"])
    assert [item["status"] for item in snapshot["runs"]] == [
        "SUCCEEDED", "INTERRUPTED", "SUCCEEDED",
    ]
    records = _json_lines(capture)
    execution_tags = [record["latest_user_markers"] for record in records if record["latest_user_markers"]]
    assert execution_tags == [["enqueue-one"], ["enqueue-two"]]
    assert len({record["pid"] for record in records}) >= 3
    _assert_role_alternation(records)


def test_message_delivery_and_conversation_checkpoint_cross_cold_restart(tmp_path):
    home = tmp_path / "profile"
    capture, effect = tmp_path / "requests.jsonl", tmp_path / "effects.jsonl"
    _configure(home)
    seed = _run(home, "seed", capture=capture)
    queued = _run(
        home, "enqueue", "--worker-id", seed["worker_id"], "--run-id", seed["run_id"],
        "--message", "checkpoint-resume",
    )
    for text in ("message-one", "message-two"):
        delivered = _run(
            home, "control", "--action", "message", "--worker-id", seed["worker_id"],
            "--message", text,
        )
        assert delivered["delivery"] == "NEXT_RUN"

    first_marker = tmp_path / "delivery.ready"
    first_process = _start(
        home, "control", "--action", "wait", "--worker-id", seed["worker_id"],
        "--run-id", queued["run_id"], "--timeout", "20",
        behavior="block-before-checkpoint", marker=first_marker, capture=capture, effect=effect,
    )
    _kill_after_marker(first_process, first_marker)

    checkpoint_marker = tmp_path / "checkpoint.ready"
    checkpoint_process = _start(
        home, "control", "--action", "resume", "--worker-id", seed["worker_id"],
        "--run-id", queued["run_id"], "--message", "checkpoint-resume", "--wait",
        "--timeout", "20", "--expire-leases",
        behavior="checkpoint-block", marker=checkpoint_marker, capture=capture, effect=effect,
    )
    _kill_after_marker(checkpoint_process, checkpoint_marker)

    snapshot = _run(
        home, "snapshot", "--worker-id", seed["worker_id"], "--expire-leases",
    )
    assert snapshot["message_statuses"] == ["DELIVERED", "DELIVERED"], _json_lines(capture)
    assert snapshot["history_roles"][-3:] == ["user", "assistant", "tool"]
    assert snapshot["history_has_provider_session_handle"] is False
    assert len(_json_lines(effect)) == 1

    resumed = _run(
        home, "control", "--action", "resume", "--worker-id", seed["worker_id"],
        "--message", "post-checkpoint", "--wait", "--timeout", "20",
        capture=capture, effect=effect,
    )
    assert resumed["terminal"]["status"] == "SUCCEEDED"
    records = _json_lines(capture)
    post_checkpoint = [record for record in records if "post-checkpoint" in record["latest_user_markers"]]
    assert len(post_checkpoint) == 1
    assert not ({"message-one", "message-two"} & set(post_checkpoint[0]["latest_user_markers"]))
    assert len({record["system_hash"] for record in records}) == 1
    _assert_role_alternation(records)


def test_ambiguous_completed_effect_requires_reconciliation_and_is_not_replayed(tmp_path):
    home = tmp_path / "profile"
    capture, effect = tmp_path / "requests.jsonl", tmp_path / "effects.jsonl"
    _configure(home)
    seed = _run(home, "seed", capture=capture)
    queued = _run(
        home, "enqueue", "--worker-id", seed["worker_id"], "--run-id", seed["run_id"],
        "--message", "ambiguous-effect",
    )
    marker = tmp_path / "effect.ready"
    process = _start(
        home, "control", "--action", "wait", "--worker-id", seed["worker_id"],
        "--run-id", queued["run_id"], "--timeout", "20",
        behavior="tool-crash", marker=marker, capture=capture, effect=effect,
    )
    _wait_for(marker, process, diagnostics=capture)
    process.wait(timeout=5)
    assert process.returncode == 91
    assert len(_json_lines(effect)) == 1

    blocked = _run(
        home, "control", "--action", "resume", "--worker-id", seed["worker_id"],
        "--run-id", queued["run_id"], "--message", "reconciled-resume", "--expire-leases",
        capture=capture, effect=effect,
    )
    assert "uncertain tool side effect" in blocked["error"]
    assert len(_json_lines(effect)) == 1
    snapshot = _run(home, "snapshot", "--worker-id", seed["worker_id"])
    assert snapshot["worker_uncertain_side_effect"] is True

    reconciled = _run(
        home, "control", "--action", "reconcile", "--worker-id", seed["worker_id"],
        "--run-id", queued["run_id"], "--message", "synthetic owner reconciliation",
        "--disposition", "accepted_unknown_no_replay",
    )
    assert reconciled["reconciled"] is True
    resumed = _run(
        home, "control", "--action", "resume", "--worker-id", seed["worker_id"],
        "--message", "reconciled-resume", "--wait", "--timeout", "20",
        capture=capture, effect=effect,
    )
    assert resumed["terminal"]["status"] == "SUCCEEDED"
    assert len(_json_lines(effect)) == 1


def test_credential_disappearance_fails_closed_before_cold_resume(tmp_path):
    home, capture = tmp_path / "profile", tmp_path / "requests.jsonl"
    _configure(home)
    seed = _run(home, "seed", capture=capture)
    blocked = _run(
        home, "control", "--action", "resume", "--worker-id", seed["worker_id"],
        "--run-id", seed["run_id"], "--message", "post-checkpoint",
        credential=False, capture=capture,
    )
    assert not blocked.get("success", False)
    assert "Worker resume admission failed" in blocked["error"]
    assert "No LLM provider configured" in blocked["error"]
    assert len(_json_lines(capture)) == 1  # the original seed; no revoked-route request
    snapshot = _run(home, "snapshot", "--worker-id", seed["worker_id"])
    assert len(snapshot["runs"]) == 1


def test_grouped_background_completion_restores_and_acks_after_cold_restart(tmp_path):
    home, capture = tmp_path / "profile", tmp_path / "requests.jsonl"
    _configure(home)
    dispatched = _run(home, "async-group", "--timeout", "20", capture=capture)
    assert dispatched["dispatch_status"] == "dispatched"
    assert dispatched["durable_state"] == "completed"
    assert dispatched["delivery_state"] == "pending"

    restored = _run(home, "async-restore", "--ack")
    assert restored["restored"] == 1
    assert restored["events"] == [{
        "delegation_id": dispatched["delegation_id"],
        "group": "joined",
        "restored": True,
        "result_indexes": [0, 1],
        "result_statuses": ["completed", "completed"],
        "claimed": True,
        "delivery_state": "delivered",
    }]
    assert _run(home, "async-restore")["restored"] == 0
    records = _json_lines(capture)
    assert sorted(record["latest_user_markers"] for record in records) == [["group-one"], ["group-two"]]
    assert len({record["pid"] for record in records}) == 1
    _assert_role_alternation(records)
