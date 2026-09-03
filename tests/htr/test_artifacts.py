import pytest

from htr import artifacts, events, io, paths
from htr.ids import new_attempt_id, new_run_id, new_task_id
from htr.schemas import validate
from htr.state import ATTEMPT_RUNNING


def _bootstrap_attempt(tmp_path):
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    events.register_attempt(
        run_id, task_id, attempt_id, actor="test", base_dir=tmp_path
    )
    return run_id, task_id, attempt_id


def test_read_artifact_manifest_normalizes_bootstrap_manifest(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    manifest = artifacts.read_artifact_manifest(
        run_id, task_id, attempt_id, base_dir=tmp_path
    )
    assert manifest["attempt_id"] == attempt_id
    assert manifest["artifacts"] == []
    assert manifest["schema_version"] == "1"
    assert manifest["run_id"] == run_id
    assert manifest["task_id"] == task_id


def test_write_artifact_manifest_writes_valid_manifest(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "artifacts": [],
    }
    target = artifacts.write_artifact_manifest(
        run_id, task_id, attempt_id, manifest, base_dir=tmp_path
    )
    assert target.exists()
    validate(io.read_json(target), "artifact_manifest")


def test_add_artifact_creates_entry_in_manifest(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    entry = artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/output.txt",
        kind="file",
        sha256="abc123",
        size_bytes=3,
        metadata={"note": "demo"},
        base_dir=tmp_path,
    )
    assert entry["path"] == "artifacts/output.txt"
    manifest = artifacts.read_artifact_manifest(
        run_id, task_id, attempt_id, base_dir=tmp_path
    )
    assert manifest["artifacts"] == [entry]


def test_add_artifact_idempotent_same_path_kind_and_metadata(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    first = artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/output.txt",
        kind="file",
        sha256="abc123",
        size_bytes=3,
        metadata={"note": "demo"},
        base_dir=tmp_path,
    )
    second = artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/output.txt",
        kind="file",
        sha256="abc123",
        size_bytes=3,
        metadata={"note": "demo"},
        base_dir=tmp_path,
    )
    assert second == first
    assert len(artifacts.list_artifacts(run_id, task_id, attempt_id, tmp_path)) == 1


def test_add_artifact_conflicting_checksum_raises(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/output.txt",
        kind="file",
        sha256="abc123",
        base_dir=tmp_path,
    )
    with pytest.raises(artifacts.ArtifactConflict):
        artifacts.add_artifact(
            run_id,
            task_id,
            attempt_id,
            path="artifacts/output.txt",
            kind="file",
            sha256="different",
            base_dir=tmp_path,
        )


def test_list_artifacts_returns_entries(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    entry = artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/output.txt",
        kind="file",
        base_dir=tmp_path,
    )
    assert artifacts.list_artifacts(run_id, task_id, attempt_id, tmp_path) == [entry]


def test_add_artifact_does_not_update_attempt_status(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    status_path = paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    before = io.read_json(status_path)
    artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/output.txt",
        kind="file",
        base_dir=tmp_path,
    )
    assert io.read_json(status_path) == before


def test_add_artifact_does_not_append_lifecycle_event(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/output.txt",
        kind="file",
        base_dir=tmp_path,
    )
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before


def test_add_artifact_rejects_invalid_metadata_type(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        artifacts.add_artifact(
            run_id,
            task_id,
            attempt_id,
            path="artifacts/output.txt",
            kind="file",
            metadata="bad",
            base_dir=tmp_path,
        )


def test_manifest_preserves_existing_artifacts(tmp_path):
    run_id, task_id, attempt_id = _bootstrap_attempt(tmp_path)
    first = artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/a.txt",
        kind="file",
        base_dir=tmp_path,
    )
    second = artifacts.add_artifact(
        run_id,
        task_id,
        attempt_id,
        path="artifacts/b.txt",
        kind="file",
        base_dir=tmp_path,
    )
    entries = artifacts.list_artifacts(run_id, task_id, attempt_id, tmp_path)
    assert entries == [first, second]
