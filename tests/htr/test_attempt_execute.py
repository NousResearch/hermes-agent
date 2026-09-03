"""Isolated tests for the Stage 33-B attempt execute/verify primitive."""

from __future__ import annotations

import inspect
import hashlib
import socket
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from htr.attempt_execute import (
    CANARY_BYTES,
    CANARY_FILE_NAME,
    CANARY_RELATIVE_PATH,
    AttemptExecuteError,
    execute_attempt_canary,
    verify_attempt_canary,
)
from htr.ids import new_attempt_id, new_run_id, new_task_id
from htr.paths import (
    attempt_status_path,
    result_json_path,
    artifacts_dir,
    task_status_path,
)
from htr.state import (
    ATTEMPT_CREATED,
    ATTEMPT_RESULT_SUBMITTED,
    ATTEMPT_RUNNING,
    ATTEMPT_VERIFICATION_FAILED,
    ATTEMPT_VERIFICATION_PASSED,
    TASK_CREATED,
    TASK_RUNNING,
    RunSealBlockedError,
)
from htr import events, io

PROTECTED_RUN = Path("/home/unaliu/.hermes/runs/run_20260825_78cd3f")
PROTECTED_PROJECT = Path(
    "/home/unaliu/.hermes/.htr/project_registry/projects/"
    "prj_20260825_65362a/record.json"
)
PROTECTED_FILES = (
    PROTECTED_PROJECT,
    PROTECTED_RUN / "run_manifest.json",
    PROTECTED_RUN / "task_events.jsonl",
    PROTECTED_RUN / "approvals.jsonl",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protected_baseline():
    snapshot = {}
    for path in PROTECTED_FILES:
        if not path.is_file():
            return None
        snapshot[str(path)] = (_sha256_file(path), path.stat().st_size)
    return snapshot


@pytest.fixture(scope="module")
def protected_snapshot():
    before = _protected_baseline()
    yield before
    after = _protected_baseline()
    if before is None:
        return
    assert after == before


def _bootstrap_running(tmp_path, *, task_running=True, attempt_running=True):
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    if task_running:
        events.apply_task_transition(
            run_id, task_id, TASK_RUNNING, actor="test", base_dir=tmp_path
        )
    events.register_attempt(
        run_id, task_id, attempt_id, actor="test", base_dir=tmp_path
    )
    if attempt_running:
        events.apply_attempt_transition(
            run_id,
            task_id,
            attempt_id,
            ATTEMPT_RUNNING,
            actor="test",
            base_dir=tmp_path,
        )
    return run_id, task_id, attempt_id


def _canary_path(tmp_path, run_id, task_id, attempt_id) -> Path:
    return artifacts_dir(run_id, task_id, attempt_id, tmp_path) / CANARY_FILE_NAME


def test_happy_path_execute_then_verify(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(tmp_path)
    with (
        patch.object(subprocess, "Popen") as popen,
        patch.object(subprocess, "run") as run,
        patch.object(socket.socket, "connect") as connect,
    ):
        executed = execute_attempt_canary(
            run_id, task_id, attempt_id, base_dir=tmp_path
        )
        verified = verify_attempt_canary(
            run_id, task_id, attempt_id, base_dir=tmp_path
        )
    popen.assert_not_called()
    run.assert_not_called()
    connect.assert_not_called()

    expected_digest = hashlib.sha256(CANARY_BYTES).hexdigest()
    canary = _canary_path(tmp_path, run_id, task_id, attempt_id)
    assert canary.read_bytes() == CANARY_BYTES
    assert executed["sha256"] == expected_digest
    assert executed["artifact"]["path"] == CANARY_RELATIVE_PATH
    assert executed["artifact"]["sha256"] == expected_digest
    assert executed["result"]["outputs"]["sha256"] == expected_digest

    attempt_status = io.read_json(
        attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    )
    assert attempt_status["status"] == ATTEMPT_VERIFICATION_PASSED
    assert io.read_json(result_json_path(run_id, task_id, attempt_id, tmp_path))
    assert verified["outcome"] == "passed"
    assert {item["name"] for item in verified["checks"]} == {
        "artifact_exists",
        "artifact_inside_workspace",
        "bytes_match_canary",
        "sha256_matches_execute_record",
        "manifest_matches_file",
    }
    assert all(item["status"] == "passed" for item in verified["checks"])


def test_attempt_not_running_is_rejected(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(
        tmp_path, attempt_running=False
    )
    with pytest.raises(AttemptExecuteError, match="is not running"):
        execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    canary = _canary_path(tmp_path, run_id, task_id, attempt_id)
    assert not canary.exists()
    status = io.read_json(attempt_status_path(run_id, task_id, attempt_id, tmp_path))
    assert status["status"] == ATTEMPT_CREATED
    assert not result_json_path(run_id, task_id, attempt_id, tmp_path).exists()


def test_task_not_running_is_rejected(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(
        tmp_path, task_running=False, attempt_running=True
    )
    with pytest.raises(AttemptExecuteError, match="task .* is not running"):
        execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    assert io.read_json(task_status_path(run_id, task_id, tmp_path))["status"] == TASK_CREATED
    assert not _canary_path(tmp_path, run_id, task_id, attempt_id).exists()
    assert not result_json_path(run_id, task_id, attempt_id, tmp_path).exists()


def test_tampered_artifact_fails_verification(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(tmp_path)
    execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    canary = _canary_path(tmp_path, run_id, task_id, attempt_id)
    canary.write_bytes(b"tampered\n")
    verified = verify_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    assert verified["outcome"] == "failed"
    status = io.read_json(attempt_status_path(run_id, task_id, attempt_id, tmp_path))
    assert status["status"] == ATTEMPT_VERIFICATION_FAILED
    assert canary.read_bytes() == b"tampered\n"


def test_deleted_artifact_fails_verification(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(tmp_path)
    execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    canary = _canary_path(tmp_path, run_id, task_id, attempt_id)
    canary.unlink()
    verified = verify_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    assert verified["outcome"] == "failed"
    status = io.read_json(attempt_status_path(run_id, task_id, attempt_id, tmp_path))
    assert status["status"] == ATTEMPT_VERIFICATION_FAILED
    assert not canary.exists()


def test_artifacts_dir_symlink_escape_is_rejected(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(tmp_path)
    artifacts_path = artifacts_dir(run_id, task_id, attempt_id, tmp_path)
    outside = tmp_path / "outside_escape"
    outside.mkdir()
    sentinel = outside / CANARY_FILE_NAME
    artifacts_path.rename(tmp_path / "artifacts_backup")
    artifacts_path.symlink_to(outside)
    with pytest.raises(AttemptExecuteError, match="symlink escape|unsafe directory"):
        execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    assert not sentinel.exists()
    assert list(outside.iterdir()) == []
    assert not result_json_path(run_id, task_id, attempt_id, tmp_path).exists()
    status = io.read_json(attempt_status_path(run_id, task_id, attempt_id, tmp_path))
    assert status["status"] == ATTEMPT_RUNNING


def test_canary_file_symlink_escape_is_rejected(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(tmp_path)
    outside = tmp_path / "outside_file"
    outside.write_bytes(b"secret\n")
    canary = _canary_path(tmp_path, run_id, task_id, attempt_id)
    canary.symlink_to(outside)
    with pytest.raises(AttemptExecuteError, match="symlink escape|already exists"):
        execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    assert outside.read_bytes() == b"secret\n"
    assert canary.is_symlink()
    assert not result_json_path(run_id, task_id, attempt_id, tmp_path).exists()


def test_duplicate_execute_is_rejected(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(tmp_path)
    first = execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    with pytest.raises(AttemptExecuteError, match="is not running"):
        execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    canary = _canary_path(tmp_path, run_id, task_id, attempt_id)
    assert canary.read_bytes() == CANARY_BYTES
    assert first["sha256"] == hashlib.sha256(CANARY_BYTES).hexdigest()
    status = io.read_json(attempt_status_path(run_id, task_id, attempt_id, tmp_path))
    assert status["status"] == ATTEMPT_RESULT_SUBMITTED


def test_sealed_run_is_rejected(tmp_path, protected_snapshot):
    run_id, task_id, attempt_id = _bootstrap_running(tmp_path)
    closure = tmp_path / run_id / "run_final_closure_record.json"
    closure.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RunSealBlockedError):
        execute_attempt_canary(run_id, task_id, attempt_id, base_dir=tmp_path)
    assert not _canary_path(tmp_path, run_id, task_id, attempt_id).exists()
    assert not result_json_path(run_id, task_id, attempt_id, tmp_path).exists()
    status = io.read_json(attempt_status_path(run_id, task_id, attempt_id, tmp_path))
    assert status["status"] == ATTEMPT_RUNNING


def test_execute_does_not_accept_caller_outcome_or_checksum():
    execute_params = inspect.signature(execute_attempt_canary).parameters
    verify_params = inspect.signature(verify_attempt_canary).parameters
    for name in ("outcome", "sha256", "result", "passed", "failed", "checksum"):
        assert name not in execute_params
        assert name not in verify_params


def test_protected_run_untouched_at_end(protected_snapshot):
    after = _protected_baseline()
    if protected_snapshot is None:
        pytest.skip("protected local canary run is not present on this host")
    assert after == protected_snapshot
