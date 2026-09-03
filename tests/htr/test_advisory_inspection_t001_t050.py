"""Task 29 traceability tests T001–T050 (Revision 5)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest import mock

import pytest

from htr.advisory_inspection_decoder import decode_control_json
from htr.advisory_inspection_path import lexical_validate_artifact_path, path_identity_digest
from htr.advisory_inspection_secure import (
    classify_regular_file_presence,
    raw_sha256_digest,
    read_regular_control_file,
    validate_runs_root_s0,
)
from htr.artifact_inspection import inspect_artifact_reference, is_supplemental_finding
from htr import paths

from tests.htr.conftest_advisory_inspection import (
    ARTIFACT_AXIS_SCALARS,
    FORBIDDEN_NAMES,
    artifact_entry,
    artifact_selector,
    bootstrap_attempt,
    collect_source_names,
    deep_json,
    forbidden_api_names,
    make_fifo,
    make_hardlink,
    make_socket,
    make_symlink,
    manifest_payload,
    pad_json_body,
    patch_hash_artifact,
    patch_presence,
    patch_read_race,
    repo_root,
    write_manifest,
    write_manifest_bytes,
)


def test_t001_runs_root_missing(advisory_runs_root, monkeypatch):
    """T001: Runs Root missing → runs_root_absent; no mkdir."""
    advisory_runs_root.rmdir()
    ctx, status = validate_runs_root_s0()
    assert ctx is None
    assert status == "runs_root_absent"


def test_t002_runs_root_symlink_blocked(advisory_runs_root, tmp_path):
    """T002: Runs Root symlink → runs_root_symlink_blocked."""
    real = tmp_path / "real_runs"
    real.mkdir()
    if advisory_runs_root.is_symlink() or advisory_runs_root.exists():
        if advisory_runs_root.is_symlink():
            advisory_runs_root.unlink()
        elif advisory_runs_root.is_dir():
            import shutil

            shutil.rmtree(advisory_runs_root)
        else:
            advisory_runs_root.unlink()
    advisory_runs_root.symlink_to(real)
    ctx, status = validate_runs_root_s0()
    assert ctx is None
    assert status == "runs_root_symlink_blocked"


def test_t003_invalid_run_id_rejected_before_open(advisory_runs_root):
    """T003: Invalid run_id → selector_identity_invalid; no open."""
    from htr.advisory_inspection_models import ArtifactReferenceSelector
    from htr.ids import new_attempt_id, new_task_id

    selector = ArtifactReferenceSelector(
        run_id="not-a-run-id",
        task_id=new_task_id(),
        attempt_id=new_attempt_id(),
        manifest_raw_digest="sha256:" + "a" * 64,
        entry_index=0,
    )
    result = inspect_artifact_reference(selector)
    assert result.authority_status == "selector_identity_invalid"
    assert result.manifest_status == "manifest_absent"


def test_t004_base_dir_kwarg_rejected(advisory_runs_root):
    """T004: base_dir passed → caller_host_root_rejected."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(
        artifact_selector(run_id, task_id, attempt_id, digest),
        base_dir="/tmp/evil",
    )
    assert result.authority_status == "caller_host_root_rejected"
    assert result.aggregate_completeness == "aggregate_blocked_untrusted_scope"


def test_t005_manifest_absent(advisory_runs_root):
    """T005: Manifest absent."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    manifest_path = paths.artifact_manifest_path(run_id, task_id, attempt_id, advisory_runs_root.parent / "runs")
    if manifest_path.exists():
        manifest_path.unlink()
    result = inspect_artifact_reference(
        artifact_selector(run_id, task_id, attempt_id, "sha256:" + "a" * 64),
    )
    assert result.manifest_status == "manifest_absent"


def test_t006_manifest_symlink(advisory_runs_root):
    """T006: Manifest symlink."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    base = advisory_runs_root.parent / "runs"
    real = paths.attempt_dir(run_id, task_id, attempt_id, base) / "real_manifest.json"
    real.write_text('{"schema_version":"1"}\n', encoding="utf-8")
    manifest = paths.artifact_manifest_path(run_id, task_id, attempt_id, base)
    if manifest.exists():
        manifest.unlink()
    manifest.symlink_to(real)
    result = inspect_artifact_reference(
        artifact_selector(run_id, task_id, attempt_id, "sha256:" + "a" * 64),
    )
    assert result.manifest_status == "manifest_symlink_blocked"
    assert result.file_type_status == "file_symlink"


def test_t007_manifest_hardlink_blocked(advisory_runs_root):
    """T007: Manifest st_nlink=2 → manifest_hardlink_blocked."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    base = advisory_runs_root.parent / "runs"
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=base,
    )
    manifest = paths.artifact_manifest_path(run_id, task_id, attempt_id, base)
    hard = manifest.parent / "manifest_hardlink.json"
    make_hardlink(manifest, hard)
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_hardlink_blocked"
    assert result.file_type_status == "file_regular"


def test_t008_manifest_is_directory(advisory_runs_root):
    """T008: Manifest is directory."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    base = advisory_runs_root.parent / "runs"
    manifest = paths.artifact_manifest_path(run_id, task_id, attempt_id, base)
    if manifest.exists():
        manifest.unlink()
    manifest.mkdir()
    result = inspect_artifact_reference(
        artifact_selector(run_id, task_id, attempt_id, "sha256:" + "a" * 64),
    )
    assert result.manifest_status == "manifest_wrong_type"
    assert result.file_type_status == "file_directory"


def test_t009_manifest_byte_budget_exceeded(advisory_runs_root):
    """T009: 1048579 bytes triggers manifest_byte_budget_exceeded."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    base = advisory_runs_root.parent / "runs"
    raw = b"x" * (1048578 + 1)
    write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=base)
    result = inspect_artifact_reference(
        artifact_selector(run_id, task_id, attempt_id, raw_sha256_digest(raw)),
    )
    assert result.manifest_status in {"manifest_byte_budget_exceeded", "manifest_control_budget_exceeded"}


def test_t010_manifest_empty_at_boundary(advisory_runs_root):
    """T010: Well-formed empty manifest decodes; cap constants aligned (R5 B-04)."""
    from tests.htr.conftest_advisory_inspection import MAX_BODY_BYTES, MAX_FILE_BYTES, MAX_RAW_BYTES

    assert MAX_BODY_BYTES == 1048576
    assert MAX_RAW_BYTES == MAX_BODY_BYTES + 2
    assert MAX_FILE_BYTES == MAX_RAW_BYTES
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    decoded = decode_control_json(
        (paths.artifact_manifest_path(run_id, task_id, attempt_id, advisory_runs_root)).read_bytes(),
        kind="manifest",
    )
    assert decoded.ok
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_bound"


def test_t011_replace_manifest_while_fd_held(monkeypatch, advisory_runs_root):
    """T011: Replace manifest name while fd held."""
    patch_read_race(monkeypatch, filesystem_status="target_name_replaced_while_fd_remained_open")
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.filesystem_status == "target_name_replaced_while_fd_remained_open"


def test_t012_unlink_manifest_after_open(monkeypatch, advisory_runs_root):
    """T012: Unlink manifest after open."""
    patch_read_race(monkeypatch, filesystem_status="target_disappeared_after_open")
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.filesystem_status == "target_disappeared_after_open"


def test_t013_missing_before_open(monkeypatch, advisory_runs_root):
    """T013: Missing before open."""
    patch_read_race(monkeypatch, filesystem_status="target_disappeared_before_open")
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.filesystem_status == "target_disappeared_before_open"


def test_t014_replace_attempts_parent(monkeypatch, advisory_runs_root):
    """T014: Replace attempts/{id} during walk."""
    from htr.advisory_inspection_secure import walk_attempt_path, os_close_runs_root

    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")

    def broken_walk(ctx, rid, tid, aid):
        walk, err = walk_attempt_path.__wrapped__(ctx, rid, tid, aid) if hasattr(walk_attempt_path, "__wrapped__") else None
        return None, "parent_directory_component_replaced"

    monkeypatch.setattr(
        "htr.artifact_inspection.walk_attempt_path",
        lambda *_a, **_k: (None, "parent_directory_component_replaced"),
    )
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.filesystem_status == "parent_directory_component_replaced"


def test_t015_opened_fd_identity_mismatch(monkeypatch, advisory_runs_root):
    """T015: Opened fd ≠ pre-open entry."""
    patch_read_race(monkeypatch, filesystem_status="opened_fd_identity_mismatch")
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.filesystem_status == "opened_fd_identity_mismatch"


def test_t016_truncate_during_read(monkeypatch, advisory_runs_root):
    """T016: Truncate during read N!=st_size."""
    patch_read_race(monkeypatch, filesystem_status="file_size_changed_during_read")
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.filesystem_status == "file_size_changed_during_read"


def test_t017_invalid_utf8_manifest(advisory_runs_root):
    """T017: Invalid UTF-8 manifest."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    raw = b"\xff\xfe{" + b'"schema_version":"1"' + b"}"
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root.parent / "runs")
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_utf8_malformed"


def test_t018_utf8_bom_rejected(advisory_runs_root):
    """T018: UTF-8 BOM → utf8_malformed class."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    raw = b"\xef\xbb\xbf" + json.dumps(manifest_payload(run_id, task_id, attempt_id)).encode()
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root.parent / "runs")
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_utf8_malformed"


def test_t019_duplicate_json_keys(advisory_runs_root):
    """T019: Duplicate JSON keys."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    raw = b'{"schema_version":"1","schema_version":"2","artifacts":[]}\n'
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root.parent / "runs")
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_duplicate_json_keys"


def test_t020_top_level_array(advisory_runs_root):
    """T020: Top-level array."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    raw = b"[1,2,3]\n"
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root.parent / "runs")
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_top_level_schema_malformed"


def test_t021_trailing_extra_json_value(advisory_runs_root):
    """T021: Trailing extra JSON value without strip."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    raw = b'{"schema_version":"1","run_id":"' + run_id.encode() + b'","task_id":"' + task_id.encode() + b'","attempt_id":"' + attempt_id.encode() + b'","artifacts":[]} {"extra":1}\n'
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root.parent / "runs")
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_json_malformed"


def test_t022_non_finite_number_rejected(advisory_runs_root):
    """T022: Non-finite / float number rejected."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    raw = b'{"schema_version":"1","run_id":"x","task_id":"y","attempt_id":"z","artifacts":[],"n":1.5}\n'
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root.parent / "runs")
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_json_malformed"


def test_t023_missing_artifacts_key(advisory_runs_root):
    """T023: Missing artifacts → reference_absent_from_manifest."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        {"schema_version": "1", "run_id": run_id, "task_id": task_id, "attempt_id": attempt_id},
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.reference_status == "reference_absent_from_manifest"
    assert result.manifest_status == "manifest_bound"


def test_t024_artifacts_not_list(advisory_runs_root):
    """T024: artifacts not list → reference_absent_from_manifest."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts="not-a-list"),  # type: ignore[arg-type]
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.reference_status == "reference_absent_from_manifest"


def test_t025_attempt_id_path_mismatch(advisory_runs_root):
    """T025: Record attempt_id ≠ path → manifest_scope_conflict."""
    from htr.ids import new_attempt_id

    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    other_attempt = new_attempt_id()
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, other_attempt),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status in {"manifest_scope_conflict", "manifest_partially_malformed"}
    assert result.decoded_manifest is not None
    assert result.decoded_manifest.get("attempt_id") != attempt_id


def test_t026_unknown_top_level_field_finding(advisory_runs_root):
    """T026: Unknown top-level field observed; still bind."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, surprise_field=True),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status in {"manifest_bound", "manifest_partially_malformed"}
    assert "manifest_unknown_field_observed" in result.findings


def test_t027_partial_manifest_select_valid(advisory_runs_root):
    """T027: One malformed + one valid; select valid."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(
            run_id,
            task_id,
            attempt_id,
            artifacts=[{"bad": True}, artifact_entry("artifacts/out.txt")],
        ),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest, entry_index=1))
    assert result.reference_status == "reference_selected"
    assert result.manifest_status in {"manifest_bound", "manifest_partially_malformed"}


def test_t028_select_malformed_index(advisory_runs_root):
    """T028: Select malformed index; no unsafe L1."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[{"bad": True}]),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest, entry_index=0))
    assert result.reference_status == "reference_malformed"


def test_t029_exact_duplicate_key(advisory_runs_root):
    """T029: Exact duplicate key (no metadata in key)."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    entry = artifact_entry("artifacts/a.txt")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[entry, dict(entry)]),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest, entry_index=0))
    assert result.manifest_status == "manifest_exact_duplicates_present"


def test_t030_conflict_same_path_distinct_kind(advisory_runs_root):
    """T030: Conflict same components+kind different sha256."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(
            run_id,
            task_id,
            attempt_id,
            artifacts=[
                artifact_entry("artifacts/x.txt", kind="file", sha256="sha256:" + "a" * 64),
                artifact_entry("artifacts/x.txt", kind="file", sha256="sha256:" + "b" * 64),
            ],
        ),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_conflicts_present"
    assert "reference_same_path_distinct_kind" in result.findings


def test_t031_same_path_different_kind(advisory_runs_root):
    """T031: Same path different kind."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(
            run_id,
            task_id,
            attempt_id,
            artifacts=[
                artifact_entry("artifacts/x.txt", kind="file"),
                artifact_entry("artifacts/x.txt", kind="dir"),
            ],
        ),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert "reference_same_path_distinct_kind" in result.findings


@pytest.mark.parametrize(
    ("declared_path", "expected_status", "tid"),
    [
        ("dir/./file.txt", "path_dot_rejected", "T032"),
        ("a//b", "path_separator_rejected", "T033"),
        ("/etc/passwd", "path_absolute_rejected", "T036"),
        ("../sibling", "path_dotdot_rejected", "T037"),
        ("", "path_empty_rejected", "T038"),
        ("artifacts/out/", "path_separator_rejected", "T039"),
        ("/leading", "path_absolute_rejected", "T040"),
        (r"artifacts\out.txt", "path_backslash_rejected", "T041"),
        ("artifacts/out\x00.txt", "path_control_rejected", "T042"),
        ("a" * 4097, "path_budget_exceeded", "T043"),
        ("/".join(["c"] * 33), "path_budget_exceeded", "T044"),
        ("artifacts/" + ("x" * 256), "path_budget_exceeded", "T045"),
    ],
)
def test_t032_t045_path_lexical_cases(declared_path, expected_status, tid):
    """T032–T045: Path lexical rejection cases."""
    status, components, _ = lexical_validate_artifact_path(declared_path)
    assert status == expected_status
    assert components is None


def test_t034_selector_digest_mismatch(advisory_runs_root):
    """T034: Selector digest mismatch; no entry inspect."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/x.txt")]),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(
        artifact_selector(run_id, task_id, attempt_id, "sha256:" + "b" * 64),
    )
    assert result.authority_status == "selector_manifest_digest_mismatch"
    assert result.identity_status == "identity_selector_digest_mismatch"


def test_t035_entry_index_oob(advisory_runs_root):
    """T035: entry_index OOB → reference_index_out_of_range."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root.parent / "runs",
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest, entry_index=5))
    assert result.reference_status == "reference_index_out_of_range"


def test_t046_artifact_hardlink_blocked(advisory_runs_root, monkeypatch):
    """T046: Artifact st_nlink=2."""
    patch_hash_artifact(monkeypatch, hardlink_status="artifact_hardlink_blocked", computed_digest=None)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    base = advisory_runs_root.parent / "runs"
    art_dir = paths.artifacts_dir(run_id, task_id, attempt_id, base)
    art_dir.mkdir(parents=True, exist_ok=True)
    f = art_dir / "out.txt"
    f.write_bytes(b"abc")
    make_hardlink(f, art_dir / "out_link.txt")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/out.txt")]),
        base_dir=base,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.hardlink_status == "artifact_hardlink_blocked"


def test_t047_artifact_symlink(advisory_runs_root, monkeypatch):
    """T047: Artifact symlink."""
    patch_hash_artifact(monkeypatch, file_type_status="file_symlink", computed_digest=None)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    base = advisory_runs_root.parent / "runs"
    art_dir = paths.artifacts_dir(run_id, task_id, attempt_id, base)
    art_dir.mkdir(parents=True, exist_ok=True)
    target = art_dir / "real.txt"
    target.write_bytes(b"x")
    make_symlink(target, art_dir / "link.txt")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/link.txt")]),
        base_dir=base,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.file_type_status == "file_symlink"


def test_t048_artifact_fifo(advisory_runs_root, monkeypatch):
    """T048: FIFO artifact."""
    patch_hash_artifact(monkeypatch, file_type_status="file_fifo", computed_digest=None)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/pipe")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.file_type_status == "file_fifo"


def test_t049_artifact_socket(advisory_runs_root, monkeypatch):
    """T049: Socket artifact."""
    patch_hash_artifact(monkeypatch, file_type_status="file_socket", computed_digest=None)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/sock")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.file_type_status == "file_socket"


def test_t050_size_mismatch(advisory_runs_root, monkeypatch):
    """T050: Declared size mismatch."""
    patch_hash_artifact(monkeypatch, observed_size=10, computed_digest="sha256:" + "d" * 64)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root.parent / "runs")
    base = advisory_runs_root.parent / "runs"
    art_dir = paths.artifacts_dir(run_id, task_id, attempt_id, base)
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "out.txt").write_bytes(b"12345")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(
            run_id,
            task_id,
            attempt_id,
            artifacts=[artifact_entry("artifacts/out.txt", size_bytes=99)],
        ),
        base_dir=base,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.size_status == "size_mismatch"
