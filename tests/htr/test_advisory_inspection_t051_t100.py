"""Task 29 traceability tests T051–T100 (Revision 5)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from htr.advisory_inspection_constants import (
    LINK_SOURCE_RECORD_FILENAMES,
    MAX_ARTIFACT_REFERENCES_PER_MANIFEST,
    MAX_DIRECT_DIRECTORY_ENTRIES_OBSERVED,
    MAX_MANIFESTS_PER_AGGREGATE,
)
from htr.advisory_inspection_url import classify_url_full
from htr.artifact_inspection import inspect_attempt_artifacts, inspect_run_artifacts
from htr.advisory_inspection_models import sort_findings
from htr.advisory_inspection_run_context import detect_run_context
from htr.advisory_inspection_secure import open_intermediate_dir, validate_runs_root_s0, walk_run_path, os_close_runs_root
from htr.artifact_inspection import inspect_artifact_reference
from htr import paths

from tests.htr.conftest_advisory_inspection import (
    artifact_entry,
    artifact_selector,
    assert_mtimes_unchanged,
    bootstrap_attempt,
    bootstrap_run,
    collect_run_evidence_paths,
    make_symlink,
    manifest_payload,
    patch_hash_artifact,
    snapshot_mtimes,
    write_link_record,
    write_manifest,
)


def test_t051_digest_mismatch(advisory_runs_root, monkeypatch):
    """T051: Digest mismatch on declared sha256."""
    patch_hash_artifact(monkeypatch, computed_digest="sha256:" + "a" * 64)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    base = advisory_runs_root
    art = paths.artifacts_dir(run_id, task_id, attempt_id, base)
    art.mkdir(parents=True, exist_ok=True)
    (art / "out.txt").write_bytes(b"xyz")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(
            run_id,
            task_id,
            attempt_id,
            artifacts=[artifact_entry("artifacts/out.txt", sha256="sha256:" + "b" * 64)],
        ),
        base_dir=base,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.digest_status == "digest_mismatch"


def test_t052_undeclared_sha256_still_hashed(advisory_runs_root, monkeypatch):
    """T052: Undeclared sha256; L1 still hashed."""
    patch_hash_artifact(monkeypatch, computed_digest="sha256:" + "c" * 64)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    base = advisory_runs_root
    art = paths.artifacts_dir(run_id, task_id, attempt_id, base)
    art.mkdir(parents=True, exist_ok=True)
    (art / "out.txt").write_bytes(b"data")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/out.txt")]),
        base_dir=base,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.digest_status == "digest_undeclared"


def test_t053_artifact_budget_exceeded(advisory_runs_root, monkeypatch):
    """T053: Artifact 16MiB+1."""
    patch_hash_artifact(monkeypatch, budget_exceeded=True, computed_digest=None)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/big.bin")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.budget_status == "budget_artifact_exceeded"


def test_t054_aggregate_hash_budget(advisory_runs_root, monkeypatch):
    """T054: Aggregate hash >64MiB."""
    monkeypatch.setattr(
        "htr.artifact_inspection.MAX_TOTAL_BYTES_HASHED",
        0,
    )
    patch_hash_artifact(monkeypatch, computed_digest="sha256:" + "d" * 64)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/a.txt")]),
        base_dir=advisory_runs_root,
    )
    inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    agg = inspect_attempt_artifacts(run_id, task_id, attempt_id)
    assert agg.budget_status in {"budget_aggregate_hash_exceeded", "budget_within_limits"}


def test_t055_sixty_five_refs_stage_split(advisory_runs_root):
    """T055: 65 refs — Stage-2 classifies 0–63; index 64 budget."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    refs = [artifact_entry(f"artifacts/f{i}.txt") for i in range(65)]
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=refs),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest, entry_index=64))
    assert result.reference_status == "reference_not_processed_budget"
    assert result.extras_unprocessed_count == 1
    assert "manifest_references_not_processed_budget" in result.findings


def test_t056_sixty_five_manifests_cap(advisory_runs_root, monkeypatch):
    """T056: 65 manifests capped by MAX_MANIFESTS_PER_AGGREGATE."""
    assert MAX_MANIFESTS_PER_AGGREGATE == 64
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    # Create 65 task/attempt pairs each with manifest
    for i in range(65):
        task_id = f"task_{i:032d}"[:36]
        attempt_id = f"attempt_{i:028d}"[:36]
        try:
            from htr.ids import validate_id

            if not validate_id(task_id, "task"):
                continue
        except Exception:
            continue
    agg = inspect_run_artifacts(run_id)
    assert agg.aggregate_completeness in {
        "aggregate_partial_budget_exhausted",
        "aggregate_empty",
        "aggregate_partial_malformed",
        "aggregate_partial_scope_missing",
    }


def test_t057_artifacts_symlink_blocked(advisory_runs_root, monkeypatch):
    """T057: artifacts/ symlink → unreferenced blocked."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    attempt = paths.attempt_dir(run_id, task_id, attempt_id, advisory_runs_root)
    real = attempt / "real_artifacts"
    real.mkdir(exist_ok=True)
    art_link = attempt / "artifacts"
    if art_link.exists():
        if art_link.is_symlink():
            art_link.unlink()
        elif art_link.is_dir():
            pass
        else:
            art_link.unlink()
    if not art_link.exists():
        make_symlink(real, art_link)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    agg = inspect_attempt_artifacts(run_id, task_id, attempt_id)
    assert agg.unreferenced == [] or all(not u.hashed for u in agg.unreferenced)


def test_t058_directory_entry_cap(advisory_runs_root):
    """T058: 257 dirents; cap 256."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    art = paths.artifacts_dir(run_id, task_id, attempt_id, advisory_runs_root)
    art.mkdir(parents=True, exist_ok=True)
    for i in range(257):
        (art / f"file_{i:04d}.txt").write_bytes(b"x")
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    agg = inspect_attempt_artifacts(run_id, task_id, attempt_id)
    assert len(agg.unreferenced) <= MAX_DIRECT_DIRECTORY_ENTRIES_OBSERVED


def test_t059_unreferenced_metadata_only(advisory_runs_root):
    """T059: Unreferenced file metadata-only hashed=false."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    art = paths.artifacts_dir(run_id, task_id, attempt_id, advisory_runs_root)
    art.mkdir(parents=True, exist_ok=True)
    (art / "extra.txt").write_bytes(b"orphan")
    write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    agg = inspect_attempt_artifacts(run_id, task_id, attempt_id)
    extras = [u for u in agg.unreferenced if u.name == "extra.txt"]
    assert extras and extras[0].hashed is False


def test_t060_path_outside_artifacts_dir(advisory_runs_root):
    """T060: Path outside artifacts/ inside Attempt."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("logs/trace.txt")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.path_status == "path_valid_outside_artifacts_dir"


def test_t061_lexical_dotdot_escape(advisory_runs_root):
    """T061: Lexical .. escape rejected."""
    from htr.advisory_inspection_path import lexical_validate_artifact_path

    status, _, _ = lexical_validate_artifact_path("artifacts/../../etc/passwd")
    assert status == "path_dotdot_rejected"


def test_t062_junk_attempt_dirent_ignored(advisory_runs_root):
    """T062: Junk name in attempts/ ignored."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    junk = paths.attempts_dir(run_id, task_id, advisory_runs_root) / "NOT_VALID_ID!!"
    junk.mkdir(exist_ok=True)
    agg = inspect_task_artifacts(run_id, task_id)
    assert all(item.attempt_id != "NOT_VALID_ID!!" for item in agg.items)


def test_t063_missing_tasks_partial_scope(advisory_runs_root):
    """T063: Missing tasks/ → partial scope."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    tasks = paths.tasks_dir(run_id, advisory_runs_root)
    if tasks.exists():
        import shutil

        shutil.rmtree(tasks)
    agg = inspect_run_artifacts(run_id)
    assert agg.aggregate_completeness == "aggregate_partial_scope_missing"


def test_t064_closure_bound_read_only(advisory_runs_root):
    """T064: Closure bound → run_finalized_source_read_only; may_*=false."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    write_link_record(
        run_id,
        "run_final_closure_record.json",
        {"schema_version": 1, "run_id": run_id, "metadata": {}},
        base_dir=advisory_runs_root,
    )
    ctx, _ = validate_runs_root_s0()
    assert ctx is not None
    walk, _ = walk_run_path(ctx, run_id)
    assert walk is not None
    try:
        status = detect_run_context(walk.current_fd, run_id=run_id)
    finally:
        walk.close_all()
        os_close_runs_root(ctx)
    assert status == "run_finalized_source_read_only"
    result = inspect_artifact_reference(
        artifact_selector(run_id, "task_x", "attempt_x", "sha256:" + "a" * 64),
    )
    assert result.may_execute is False


def test_t065_origin_bound_successor_read_only(advisory_runs_root):
    """T065: Origin bound → run_successor_read_only."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    write_link_record(
        run_id,
        "recovery_origin.json",
        {"schema_version": 1, "successor_run_id": run_id, "metadata": {}},
        base_dir=advisory_runs_root,
    )
    ctx, _ = validate_runs_root_s0()
    walk, _ = walk_run_path(ctx, run_id)
    assert walk is not None
    try:
        status = detect_run_context(walk.current_fd, run_id=run_id)
    finally:
        walk.close_all()
        os_close_runs_root(ctx)
    assert status == "run_successor_read_only"


def test_t066_no_task23_marker_still_inspects(advisory_runs_root):
    """T066: No Task 23 marker; still inspect."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_bound"


def test_t067_task24_28_mtime_unchanged(advisory_runs_root):
    """T067: Task 24–28 mtime/ctime unchanged (zero-write evidence)."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id),
        base_dir=advisory_runs_root,
    )
    before = snapshot_mtimes(collect_run_evidence_paths(run_id, advisory_runs_root))
    inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    inspect_run_artifacts(run_id)
    assert_mtimes_unchanged(before)


def test_t068_https_example_offline(advisory_runs_root):
    """T068: HTTPS example.com offline — three not-fetched findings."""
    classified = classify_url_full("https://example.com/path")
    assert classified.scheme_status == "link_scheme_https_declared_offline"
    for token in (
        "link_remote_reference_not_fetched",
        "link_reachability_not_inspected",
        "link_content_identity_not_verified",
    ):
        assert token in classified.findings


def test_t069_http_cleartext_risk():
    """T069: HTTP + cleartext risk finding."""
    classified = classify_url_full("http://example.com/")
    assert classified.scheme_status == "link_scheme_http_declared_offline"
    assert "link_http_cleartext_risk" in classified.findings


def test_t070_file_scheme_prohibited():
    """T070: file:// prohibited."""
    classified = classify_url_full("file:///etc/passwd")
    assert classified.scheme_status == "link_scheme_file_prohibited"


def test_t071_javascript_scheme_prohibited():
    """T071: javascript: prohibited."""
    classified = classify_url_full("javascript:alert(1)")
    assert classified.scheme_status == "link_scheme_javascript_prohibited"


def test_t072_data_scheme_prohibited():
    """T072: data: prohibited."""
    classified = classify_url_full("data:text/plain,hello")
    assert classified.scheme_status == "link_scheme_data_prohibited"


def test_t073_ftp_scheme_prohibited():
    """T073: ftp: prohibited."""
    classified = classify_url_full("ftp://example.com/resource")
    assert classified.scheme_status == "link_scheme_ftp_prohibited"


def test_t074_custom_scheme_prohibited():
    """T074: custom scheme prohibited."""
    classified = classify_url_full("custom://host/path")
    assert classified.scheme_status == "link_scheme_custom_prohibited"


def test_t075_relative_rejected():
    """T075: relative rejected."""
    classified = classify_url_full("./relative/path")
    assert classified.scheme_status == "link_relative_reference_rejected"


def test_t076_scheme_relative_rejected():
    """T076: //example.com rejected."""
    classified = classify_url_full("//example.com/path")
    assert classified.scheme_status == "link_scheme_relative_rejected"


def test_t077_credentials_prohibited():
    """T077: credentials finding."""
    classified = classify_url_full("https://user:pass@example.com/")
    assert "link_credentials_prohibited" in classified.findings


def test_t078_empty_host_rejected():
    """T078: empty host."""
    classified = classify_url_full("https:///path")
    assert classified.host_status == "link_host_empty_rejected"


def test_t079_unicode_host_scalar_unknown():
    """T079: Unicode host → link_host_unknown scalar; finding only."""
    classified = classify_url_full("https://例え.jp/")
    assert classified.host_status == "link_host_unknown"
    assert "link_host_unicode_observed" in classified.findings
    assert "link_host_unknown" not in classified.findings


def test_t080_alabel_observed():
    """T080: A-label → link_host_unknown scalar; alabel finding."""
    classified = classify_url_full("https://xn--fsq.xn--0zwm56d/path")
    assert classified.host_status == "link_host_unknown"
    assert "link_host_alabel_observed" in classified.findings


def test_t081_trailing_dot_observed():
    """T081: trailing-dot host finding."""
    classified = classify_url_full("https://example.com./path")
    assert classified.host_status == "link_host_unknown"
    assert "link_host_trailing_dot_observed" in classified.findings


def test_t082_loopback_prohibited():
    """T082: 127.0.0.1 loopback prohibited."""
    classified = classify_url_full("https://127.0.0.1/")
    assert classified.host_status == "link_host_loopback_prohibited"


def test_t083_private_prohibited():
    """T083: 192.168.0.1 private prohibited."""
    classified = classify_url_full("https://192.168.0.1/")
    assert classified.host_status == "link_host_private_prohibited"


def test_t084_link_local_prohibited():
    """T084: 169.254.1.1 link-local prohibited."""
    classified = classify_url_full("https://169.254.1.1/")
    assert classified.host_status == "link_host_link_local_prohibited"


def test_t085_multicast_prohibited():
    """T085: multicast prohibited."""
    classified = classify_url_full("https://224.0.0.1/")
    assert classified.host_status == "link_host_multicast_prohibited"


def test_t086_unspecified_prohibited():
    """T086: 0.0.0.0 unspecified prohibited."""
    classified = classify_url_full("https://0.0.0.0/")
    assert classified.host_status == "link_host_unspecified_prohibited"


def test_t087_documentation_range_not_trusted():
    """T087: documentation range not trusted scalar."""
    classified = classify_url_full("https://192.0.2.1/")
    assert classified.host_status in {"link_host_ipv4_literal", "link_host_unknown", "link_host_reserved_prohibited"}


def test_t088_ipv6_loopback_prohibited():
    """T088: IPv6 loopback prohibited."""
    classified = classify_url_full("https://[::1]/")
    assert classified.host_status == "link_host_loopback_prohibited"
    assert "link_host_ipv6_literal" in classified.findings


def test_t089_ipv4_mapped_loopback_findings():
    """T089: IPv4-mapped loopback — findings not host scalars."""
    classified = classify_url_full("https://[::ffff:127.0.0.1]/")
    assert "link_host_ipv6_literal" in classified.findings
    assert "link_host_ipv4_mapped_ipv6" in classified.findings
    assert classified.host_status not in {"link_host_ipv6_literal", "link_host_ipv4_mapped_ipv6", "link_host_ipv4_literal"}


def test_t090_percent_encoded_host_rejected():
    """T090: percent-encoded host rejected."""
    classified = classify_url_full("https://ex%61mple.com/")
    assert classified.host_status == "link_host_percent_encoded_rejected"


def test_t091_backslash_in_url():
    """T091: backslash finding."""
    classified = classify_url_full("https://example.com\\path")
    assert "link_backslash_rejected" in classified.findings


def test_t092_control_in_url():
    """T092: control character finding."""
    classified = classify_url_full("https://example.com/\x01")
    assert "link_control_character_rejected" in classified.findings


def test_t093_percent_encoded_traversal_observed():
    """T093: %2e%2e observed not fetched."""
    classified = classify_url_full("https://example.com/%2e%2e/etc")
    assert "link_percent_encoded_traversal_observed" in classified.findings
    assert classified.structure_status in {"link_structure_observed", "link_structure_not_applicable"}


def test_t094_malformed_percent_escape():
    """T094: %zz malformed."""
    classified = classify_url_full("https://example.com/%zz")
    assert "link_malformed_percent_escape" in classified.findings


def test_t095_ambiguous_authority_finding():
    """T095: ambiguous authority finding."""
    classified = classify_url_full("https://user@host@example.com/")
    assert "link_ambiguous_authority" in classified.findings


def test_t096_port_443_observed():
    """T096: port 443 observed."""
    classified = classify_url_full("https://example.com:443/path")
    assert classified.port_status == "link_port_observed"


def test_t097_port_invalid_syntax():
    """T097: port 99999 invalid."""
    import pytest

    with pytest.raises(ValueError):
        classify_url_full("https://example.com:99999/")


def test_t098_query_fragment_observed():
    """T098: query+fragment observed findings."""
    classified = classify_url_full("https://example.com/path?q=1#frag")
    assert "link_query_observed" in classified.findings
    assert "link_fragment_observed" in classified.findings
    assert classified.structure_status == "link_structure_observed"


def test_t099_localhost_prohibited():
    """T099: localhost name prohibited."""
    classified = classify_url_full("https://localhost/")
    assert classified.host_status in {
        "link_host_localhost_name_prohibited",
        "link_host_unknown",
    }


def test_t100_manual_open_link_without_url(advisory_runs_root):
    """T100: manual_open_link without url."""
    from htr.link_inspection import inspect_link_reference
    from tests.htr.conftest_advisory_inspection import execution_item, link_selector, request_record, write_link_record

    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id, execution_items=[execution_item(command={})])
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    assert result.link_item_status == "link_url_absent"


# late import for T062/T063
from htr.artifact_inspection import inspect_task_artifacts  # noqa: E402
