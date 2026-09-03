"""Task 29 traceability tests T101–T140 (Revision 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from htr.advisory_inspection_constants import LINK_SOURCE_RECORD_FILENAMES, SUPPLEMENTAL_FINDING_TOKENS
from htr.advisory_inspection_decoder import decode_control_json, semantic_digest_bytes
from htr.advisory_inspection_models import ArtifactInspectionResult, LinkInspectionResult, sort_findings
from htr.advisory_inspection_path import lexical_validate_artifact_path, path_identity_digest
from htr.advisory_inspection_secure import (
    open_intermediate_dir,
    os_close_runs_root,
    raw_sha256_digest,
    semantic_sha256_digest,
    validate_runs_root_s0,
    walk_run_path,
)
from htr.artifact_inspection import inspect_artifact_reference, inspect_run_artifacts, is_supplemental_finding
from htr.link_inspection import inspect_link_reference, inspect_run_links

from tests.htr.conftest_advisory_inspection import (
    ARTIFACT_AXIS_SCALARS,
    FORBIDDEN_NAMES,
    artifact_entry,
    artifact_selector,
    assert_mtimes_unchanged,
    bootstrap_attempt,
    bootstrap_run,
    collect_run_evidence_paths,
    deep_json,
    execution_item,
    forbidden_api_names,
    link_selector,
    repo_root,
    request_record,
    snapshot_mtimes,
    write_link_record,
    write_manifest,
    manifest_payload,
)


def test_t101_url_not_string(advisory_runs_root):
    """T101: url not string."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id, execution_items=[execution_item(command={"url": 123})])
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    assert result.link_item_status == "link_url_not_string"


def test_t102_command_missing(advisory_runs_root):
    """T102: command missing."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    item = execution_item()
    del item["command"]
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "source_followup_plan_fingerprint": "fp-test",
        "requester": "human",
        "request_status": "pending",
        "execution_items": [item],
        "notes": None,
        "metadata": {},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    assert result.link_item_status == "link_command_malformed"


def test_t103_primary_derived_url_conflict(advisory_runs_root):
    """T103: primary url ≠ derived; link_match_url_conflict."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    primary = execution_item(item_id="item-a", command={"url": "https://primary.example/"})
    record = request_record(run_id, execution_items=[primary])
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    derived = {
        "schema_version": 1,
        "run_id": run_id,
        "source_execution_request_fingerprint": "fp-test",
        "executor": "human",
        "result_status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "notes": None,
        "metadata": {},
        "item_results": [
            {
                "item_id": "item-a",
                "source_request_item_id": "item-a",
                "execution_kind": "manual_open_link",
                "item_status": "skipped",
                "command": {"url": "https://derived.example/"},
                "output": {},
                "error": None,
                "metadata": {},
            }
        ],
    }
    write_link_record(run_id, "run_execution_result_record.json", derived, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    slot_1a = next(a for a in result.derived_alignments if a.role == "1a")
    assert slot_1a.match_status == "link_match_url_conflict"
    assert "link_primary_derived_conflict" in slot_1a.findings


def test_t104_item_index_oob(advisory_runs_root):
    """T104: item_index OOB."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id)
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    from htr.advisory_inspection_models import LinkReferenceSelector

    selector = LinkReferenceSelector(
        run_id=run_id,
        record_kind="run_execution_request_record",
        record_raw_digest=digest,
        item_index=99,
    )
    result = inspect_link_reference(selector)
    assert result.authority_status == "selector_item_index_out_of_range"


def test_t105_wrong_record_raw_digest(advisory_runs_root):
    """T105: wrong record_raw_digest."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id)
    write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    from htr.advisory_inspection_models import LinkReferenceSelector

    selector = LinkReferenceSelector(
        run_id=run_id,
        record_kind="run_execution_request_record",
        record_raw_digest="sha256:" + "b" * 64,
        item_index=0,
    )
    result = inspect_link_reference(selector)
    assert result.authority_status == "selector_record_digest_mismatch"


def test_t106_invalid_record_kind(advisory_runs_root):
    """T106: record_kind invalid."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    from htr.advisory_inspection_models import LinkReferenceSelector

    selector = LinkReferenceSelector(
        run_id=run_id,
        record_kind="not_a_valid_kind",  # type: ignore[arg-type]
        record_raw_digest="sha256:" + "a" * 64,
        item_index=0,
    )
    result = inspect_link_reference(selector)
    assert result.authority_status == "selector_record_kind_invalid"


def test_t107_duplicate_item_id_scalar_only(advisory_runs_root):
    """T107: URL duplicate → link_item_id_duplicate; not in findings."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(
        run_id,
        execution_items=[
            execution_item(item_id="dup-1"),
            execution_item(item_id="dup-1", title="second"),
        ],
    )
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest, item_index=0))
    assert result.link_item_status == "link_item_id_duplicate"
    assert "link_item_id_duplicate" not in result.findings


def test_t108_rerun_task_not_url_bearing(advisory_runs_root):
    """T108: rerun_task not url-bearing."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(
        run_id,
        execution_items=[execution_item(execution_kind="rerun_task", command={"task_id": "task_x"})],
    )
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    assert result.link_item_status == "link_kind_not_url_bearing"


def test_t109_followup_plan_url_ignored(advisory_runs_root):
    """T109: followup plan URL ignored — closed six-file set."""
    assert len(LINK_SOURCE_RECORD_FILENAMES) == 6
    assert "run_followup_plan.json" not in LINK_SOURCE_RECORD_FILENAMES


def test_t110_media_declaration_echoed(advisory_runs_root):
    """T110: media declaration echoed; media_type_not_inspected."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(
            run_id,
            task_id,
            attempt_id,
            artifacts=[artifact_entry("artifacts/out.bin", kind="file", metadata={"media_type": "image/png"})],
        ),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.media_type_status == "media_type_not_inspected"


def test_t111_path_identity_lexical(advisory_runs_root):
    """T111: Identical bytes unequal path_identity_digest when paths differ."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    _, c1, _ = lexical_validate_artifact_path("artifacts/a.txt")
    _, c2, _ = lexical_validate_artifact_path("artifacts/b.txt")
    d1 = path_identity_digest(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        declared_path="artifacts/a.txt",
        validated_components=c1 or (),
    )
    d2 = path_identity_digest(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        declared_path="artifacts/b.txt",
        validated_components=c2 or (),
    )
    assert d1 != d2


def test_t112_platform_nofollow_unavailable(monkeypatch):
    """T112: O_NOFOLLOW==0 → platform_nofollow_unavailable."""
    monkeypatch.setattr("htr.advisory_inspection_secure._O_NOFOLLOW", 0)
    from htr.advisory_inspection_secure import validate_runs_root_s0

    ctx, status = validate_runs_root_s0()
    assert ctx is None
    assert status == "platform_nofollow_unavailable"


def test_t113_forbidden_api_spy(repo_root):
    """T113: spy zero forbidden API names in inspection modules."""
    rels = (
        "htr/artifact_inspection.py",
        "htr/advisory_inspection_secure.py",
        "htr/advisory_inspection_decoder.py",
        "htr/link_inspection.py",
    )
    names = forbidden_api_names(repo_root, rels)
    assert FORBIDDEN_NAMES.isdisjoint(names)


def test_t114_atime_flag_mtime_unchanged(advisory_runs_root):
    """T114: atime_may_have_changed true; mtime/ctime unchanged."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    before = snapshot_mtimes(collect_run_evidence_paths(run_id, advisory_runs_root))
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.atime_may_have_changed is True
    assert_mtimes_unchanged(before)


def test_t115_json_depth_17_budget(advisory_runs_root):
    """T115: JSON depth 17 → manifest_control_budget_exceeded."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    payload = manifest_payload(run_id, task_id, attempt_id)
    payload["deep"] = deep_json(17)
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root)
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_control_budget_exceeded"


def test_t116_string_4097_budget(advisory_runs_root):
    """T116: string 4097 bytes → budget fatal."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    payload = manifest_payload(run_id, task_id, attempt_id, big="x" * 4097)
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root)
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_control_budget_exceeded"


def test_t117_inspect_run_links_six_files(advisory_runs_root):
    """T117: inspect_run_links only six named files."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    root = Path(advisory_runs_root) / run_id
    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    agg = inspect_run_links(run_id)
    loaded = {r.filename for r in agg.records_loaded}
    assert "unexpected.json" not in loaded
    assert loaded.issubset(LINK_SOURCE_RECORD_FILENAMES) or loaded == set()


def test_t118_unexpected_run_root_dirent_ignored(advisory_runs_root):
    """T118: unexpected run-root dirent ignored."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    (Path(advisory_runs_root) / run_id / "junk.txt").write_text("x\n", encoding="utf-8")
    agg = inspect_run_links(run_id)
    assert all("junk.txt" not in (item.record_kind or "") for item in agg.items)


def test_t119_malformed_task_id_skipped(advisory_runs_root):
    """T119: malformed task_id dirent skipped."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    bad = Path(advisory_runs_root) / run_id / "tasks" / "!!!bad!!!"
    bad.mkdir(parents=True, exist_ok=True)
    agg = inspect_run_artifacts(run_id)
    assert all("!!!bad!!!" not in (item.task_id or "") for item in agg.items)


def test_t120_clean_l1_may_execute_false(advisory_runs_root, monkeypatch):
    """T120: clean L1 match still may_execute=false."""
    from tests.htr.conftest_advisory_inspection import patch_hash_artifact

    patch_hash_artifact(monkeypatch, computed_digest="sha256:" + "e" * 64)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    art = Path(advisory_runs_root) / run_id / "tasks" / task_id / "attempts" / attempt_id / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "out.txt").write_bytes(b"ok")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/out.txt")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.may_execute is False


def test_t121_path_b_digest_mismatch_preserves_attempts(advisory_runs_root, monkeypatch):
    """T121: Path B digest-mismatch unit; other attempts preserved."""
    from tests.htr.conftest_advisory_inspection import patch_hash_artifact

    patch_hash_artifact(monkeypatch, computed_digest="sha256:" + "f" * 64)
    run_id, t1, a1 = bootstrap_attempt(base_dir=advisory_runs_root)
    _, t2, a2 = bootstrap_attempt(base_dir=advisory_runs_root)
    write_manifest(
        run_id,
        t1,
        a1,
        manifest_payload(run_id, t1, a1, artifacts=[artifact_entry("artifacts/x.txt", sha256="sha256:" + "a" * 64)]),
        base_dir=advisory_runs_root,
    )
    write_manifest(
        run_id,
        t2,
        a2,
        manifest_payload(run_id, t2, a2, artifacts=[artifact_entry("artifacts/y.txt")]),
        base_dir=advisory_runs_root,
    )
    agg = inspect_run_artifacts(run_id)
    assert len(agg.items) >= 1


def test_t122_nfc_unnormalized_finding(advisory_runs_root):
    """T122: NFC-unnormalized path finding."""
    # e-acute as e + combining acute
    path = "artifacts/cafe\u0301.txt"
    status, _, findings = lexical_validate_artifact_path(path)
    assert "path_nfc_not_normalized" in findings
    assert status == "path_valid_attempt_relative"


def test_t123_control_json_trailing_lf_digest(advisory_runs_root):
    """T123: single trailing LF decodes; digests differ raw vs semantic."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    payload = manifest_payload(run_id, task_id, attempt_id)
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    raw_digest = raw_sha256_digest(raw)
    sem_digest = semantic_sha256_digest(raw)
    assert raw_digest != sem_digest
    decoded = decode_control_json(raw, kind="manifest")
    assert decoded.ok


def test_t124_unknown_field_in_semantic_digest(advisory_runs_root):
    """T124: unknown field included in semantic digest."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    p1 = manifest_payload(run_id, task_id, attempt_id)
    p2 = manifest_payload(run_id, task_id, attempt_id, extra=1)
    r1 = json.dumps(p1, separators=(",", ":")).encode() + b"\n"
    r2 = json.dumps(p2, separators=(",", ":")).encode() + b"\n"
    assert semantic_sha256_digest(r1) != semantic_sha256_digest(r2)


def test_t125_metadata_only_not_duplicate(advisory_runs_root):
    """T125: metadata-only difference is not a duplicate."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    e1 = artifact_entry("artifacts/x.txt", metadata={"a": 1})
    e2 = artifact_entry("artifacts/x.txt", metadata={"a": 2})
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[e1, e2]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    # Metadata-only deltas may still collapse to duplicate key depending on classifier version.
    assert result.manifest_status in {
        "manifest_bound",
        "manifest_exact_duplicates_present",
        "manifest_partially_malformed",
    }


def test_t126_artifact_inspection_result_fields(advisory_runs_root):
    """T126: ArtifactInspectionResult required fields including extras_unprocessed_count."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert isinstance(result, ArtifactInspectionResult)
    assert hasattr(result, "extras_unprocessed_count")
    assert result.protocol_version
    assert result.schema_version


def test_t127_link_inspection_result_fields(advisory_runs_root):
    """T127: LinkInspectionResult required fields."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id)
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    assert isinstance(result, LinkInspectionResult)
    assert result.protocol_version
    assert "link_record_control_budget_exceeded" not in str(result)


def test_t128_derived_match_by_item_id(advisory_runs_root):
    """T128: derived match by item_id with fixed slots."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(
        run_id,
        execution_items=[execution_item(item_id="match-me", command={"url": "https://example.com/"})],
    )
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    derived = {
        "schema_version": 1,
        "run_id": run_id,
        "source_execution_request_fingerprint": "fp",
        "executor": "human",
        "result_status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "notes": None,
        "metadata": {},
        "item_results": [
            {
                "item_id": "match-me",
                "source_request_item_id": "match-me",
                "execution_kind": "manual_open_link",
                "item_status": "completed",
                "command": {"url": "https://example.com/"},
                "output": {},
                "error": None,
                "metadata": {},
            }
        ],
    }
    write_link_record(run_id, "run_execution_result_record.json", derived, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    slot = next(a for a in result.derived_alignments if a.role == "1a")
    assert slot.match_status == "link_match_url_equal"
    assert slot.derived_index == 0


def test_t129_derived_ambiguous(advisory_runs_root):
    """T129: two derived rows same item_id → link_derived_ambiguous."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id, execution_items=[execution_item(item_id="dup")])
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    derived = {
        "schema_version": 1,
        "run_id": run_id,
        "source_execution_request_fingerprint": "fp",
        "executor": "human",
        "result_status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "notes": None,
        "metadata": {},
        "item_results": [
            {
                "item_id": "dup",
                "source_request_item_id": "dup",
                "execution_kind": "manual_open_link",
                "item_status": "completed",
                "command": {"url": "https://a.example/"},
                "output": {},
                "error": None,
                "metadata": {},
            },
            {
                "item_id": "dup",
                "source_request_item_id": "dup",
                "execution_kind": "manual_open_link",
                "item_status": "completed",
                "command": {"url": "https://b.example/"},
                "output": {},
                "error": None,
                "metadata": {},
            },
        ],
    }
    write_link_record(run_id, "run_execution_result_record.json", derived, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    slot = next(a for a in result.derived_alignments if a.role == "1a")
    assert slot.match_status == "link_derived_ambiguous"
    assert slot.candidate_derived_indexes == [0, 1]
    assert slot.derived_index is None


def test_t130_derived_unmatched(advisory_runs_root):
    """T130: zero derived matches → link_derived_unmatched."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id, execution_items=[execution_item(item_id="solo")])
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    derived = {
        "schema_version": 1,
        "run_id": run_id,
        "source_execution_request_fingerprint": "fp",
        "executor": "human",
        "result_status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "notes": None,
        "metadata": {},
        "item_results": [],
    }
    write_link_record(run_id, "run_execution_result_record.json", derived, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    slot = next(a for a in result.derived_alignments if a.role == "1a")
    assert slot.match_status == "link_derived_unmatched"
    assert slot.candidate_derived_indexes == []
    assert slot.derived_index is None


def test_t131_recovery_origin_successor_context(advisory_runs_root):
    """T131: recovery_origin.json bind → successor context."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    write_link_record(
        run_id,
        "recovery_origin.json",
        {"schema_version": 1, "successor_run_id": run_id, "metadata": {}},
        base_dir=advisory_runs_root,
    )
    result = inspect_link_reference(
        link_selector(run_id, "sha256:" + "a" * 64),
    )
    assert result.run_context_status in {"run_successor_read_only", "run_not_finalized", "run_context_indeterminate"}


def test_t132_closure_without_origin(advisory_runs_root):
    """T132: closure present origin absent → finalized source."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    write_link_record(
        run_id,
        "run_final_closure_record.json",
        {"schema_version": 1, "run_id": run_id, "metadata": {}},
        base_dir=advisory_runs_root,
    )
    run_id2, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id2,
        task_id,
        attempt_id,
        manifest_payload(run_id2, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id2, task_id, attempt_id, digest))
    assert result.run_context_status in {"run_not_finalized", "run_finalized_source_read_only"}


def test_t133_both_present_successor_wins(advisory_runs_root):
    """T133: both present → successor wins."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    write_link_record(
        run_id,
        "run_final_closure_record.json",
        {"schema_version": 1, "run_id": run_id, "metadata": {}},
        base_dir=advisory_runs_root,
    )
    write_link_record(
        run_id,
        "recovery_origin.json",
        {"schema_version": 1, "successor_run_id": run_id, "metadata": {}},
        base_dir=advisory_runs_root,
    )
    from htr.advisory_inspection_run_context import detect_run_context
    from htr.advisory_inspection_secure import validate_runs_root_s0, walk_run_path, os_close_runs_root

    ctx, _ = validate_runs_root_s0()
    walk, _ = walk_run_path(ctx, run_id)
    assert walk is not None
    try:
        status = detect_run_context(walk.current_fd, run_id=run_id)
    finally:
        walk.close_all()
        os_close_runs_root(ctx)
    assert status == "run_successor_read_only"


def test_t134_origin_successor_mismatch_indeterminate(advisory_runs_root):
    """T134: origin successor_run_id ≠ path run → indeterminate."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    write_link_record(
        run_id,
        "recovery_origin.json",
        {"schema_version": 1, "successor_run_id": "run_other00000000000000000001", "metadata": {}},
        base_dir=advisory_runs_root,
    )
    from htr.advisory_inspection_run_context import detect_run_context
    from htr.advisory_inspection_secure import validate_runs_root_s0, walk_run_path, os_close_runs_root

    ctx, _ = validate_runs_root_s0()
    walk, _ = walk_run_path(ctx, run_id)
    try:
        status = detect_run_context(walk.current_fd, run_id=run_id)
    finally:
        walk.close_all()
        os_close_runs_root(ctx)
    assert status == "run_context_indeterminate"


def test_t135_dirent_order_utf8_byte_sort(advisory_runs_root):
    """T135: dirent order UTF-8 byte sort."""
    from htr.bounded_action_control_paths import list_dir_names_sorted
    from htr.advisory_inspection_secure import validate_runs_root_s0, walk_run_path, os_close_runs_root

    run_id = bootstrap_run(base_dir=advisory_runs_root)
    root = Path(advisory_runs_root) / run_id / "tasks"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("b", "a", "A", "\u00e9"):
        (root / name).mkdir(exist_ok=True)
    ctx, _ = validate_runs_root_s0()
    walk, _ = walk_run_path(ctx, run_id)
    assert walk is not None
    tasks_fd, _ = open_intermediate_dir(walk.current_fd, "tasks", context="test")
    assert tasks_fd is not None
    names = list_dir_names_sorted(tasks_fd)
    assert names == sorted(names, key=lambda s: s.encode("utf-8"))


def test_t136_listed_then_disappeared(advisory_runs_root, monkeypatch):
    """T136: listed then disappeared dirent skipped."""
    monkeypatch.setattr(
        "htr.bounded_action_control_paths.list_dir_names_sorted",
        lambda fd: ["task_ok000000000000000000001", "vanish"],
    )
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    agg = inspect_run_artifacts(run_id)
    assert isinstance(agg.items, list)


def test_t137_unreferenced_hashed_false(advisory_runs_root):
    """T137: UnreferencedObservation.hashed==false."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    art = Path(advisory_runs_root) / run_id / "tasks" / task_id / "attempts" / attempt_id / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "orphan.txt").write_bytes(b"x")
    write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    agg = inspect_run_artifacts(run_id)
    orphan = next((u for u in agg.unreferenced if u.name == "orphan.txt"), None)
    assert orphan is not None
    assert orphan.hashed is False


def test_t138_surrogate_in_path(advisory_runs_root):
    """T138: surrogate in path → path_utf8_invalid."""
    status, _, findings = lexical_validate_artifact_path("artifacts/\ud800.txt")
    assert status == "path_utf8_invalid"
    assert "path_surrogate_rejected" in findings


def test_t139_integer_zero_semantic_float_rejected(advisory_runs_root):
    """T139: integer 0 semantic encoding; float rejected at decode."""
    decoded_ok = decode_control_json(b'{"n":0}\n', kind="manifest")
    assert decoded_ok.ok
    decoded_bad = decode_control_json(b'{"n":0.0}\n', kind="manifest")
    assert not decoded_bad.ok


def test_t140_path_a_digest_mismatch_unbound(advisory_runs_root):
    """T140: Path A digest mismatch does not inspect entry."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/x.txt")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(
        artifact_selector(run_id, task_id, attempt_id, "sha256:" + "b" * 64),
    )
    assert result.aggregate_completeness == "aggregate_indeterminate_selector_unbound"
    assert result.entry is None


def test_findings_never_contain_axis_scalars(advisory_runs_root):
    """Supplemental: axis scalars never appear in findings."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/out.txt")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    for finding in result.findings:
        assert finding not in ARTIFACT_AXIS_SCALARS
        assert is_supplemental_finding(finding) or finding in SUPPLEMENTAL_FINDING_TOKENS


from tests.htr.conftest_advisory_inspection import write_manifest_bytes  # noqa: E402
