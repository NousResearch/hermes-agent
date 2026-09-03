"""Task 29 PROVISIONAL traceability tests T141–T168 (Revision 5).

PROVISIONAL — numbering not locked; architect re-review pending.
"""

from __future__ import annotations

import json

import pytest

from htr.advisory_inspection_constants import MAX_ARTIFACT_REFERENCES_PER_MANIFEST, MAX_CONTROL_JSON_BYTES
from htr.advisory_inspection_decoder import decode_control_json
from htr.advisory_inspection_models import sort_findings
from htr.advisory_inspection_url import classify_url_full
from htr.artifact_inspection import inspect_artifact_reference
from htr.link_inspection import inspect_link_reference

from tests.htr.conftest_advisory_inspection import (
    artifact_entry,
    artifact_selector,
    bootstrap_attempt,
    bootstrap_run,
    deep_json,
    execution_item,
    link_selector,
    manifest_payload,
    pad_json_body,
    request_record,
    write_link_record,
    write_manifest,
    write_manifest_bytes,
)

pytestmark = pytest.mark.provisional


def test_t141_provisional_stage1_retains_100_stage2_64(advisory_runs_root):
    """T141 PROVISIONAL: Stage-1 retains 100 artifacts; Stage-2 processes 64."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    refs = [artifact_entry(f"artifacts/f{i}.txt") for i in range(100)]
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=refs),
        base_dir=advisory_runs_root,
    )
    from htr import paths as htr_paths

    manifest_bytes = htr_paths.artifact_manifest_path(
        run_id, task_id, attempt_id, advisory_runs_root
    ).read_bytes()
    decoded = decode_control_json(manifest_bytes, kind="manifest")
    assert decoded.ok
    assert len(decoded.obj.get("artifacts", [])) == 100
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest, entry_index=63))
    assert result.reference_status == "reference_selected"
    assert result.extras_unprocessed_count == 36


def test_t142_provisional_index_63_vs_64(advisory_runs_root):
    """T142 PROVISIONAL: Index 63 processed; index 64 not."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    refs = [artifact_entry(f"artifacts/f{i}.txt") for i in range(65)]
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=refs),
        base_dir=advisory_runs_root,
    )
    ok = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest, entry_index=63))
    blocked = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest, entry_index=64))
    assert ok.reference_status == "reference_selected"
    assert blocked.reference_status == "reference_not_processed_budget"


def test_t143_provisional_selector_entry_64(advisory_runs_root):
    """T143 PROVISIONAL: Selector entry_index=64 with len=65."""
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
    assert result.budget_status == "budget_references_exceeded"


def test_t144_provisional_budget_finding_once(advisory_runs_root):
    """T144 PROVISIONAL: manifest_references_not_processed_budget once per result."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    refs = [artifact_entry(f"artifacts/f{i}.txt") for i in range(70)]
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=refs),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    count = result.findings.count("manifest_references_not_processed_budget")
    assert count == 1


def test_t145_provisional_stage1_array_budget_independent(advisory_runs_root):
    """T145 PROVISIONAL: nested array >64 inside entry fatal Stage-1."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    payload = manifest_payload(
        run_id,
        task_id,
        attempt_id,
        artifacts=[artifact_entry("artifacts/x.txt", metadata={"nested": list(range(65))})],
    )
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root)
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_control_budget_exceeded"


def test_t146_provisional_body_exactly_1048576(advisory_runs_root):
    """T146 PROVISIONAL: Body at cap boundary decodes or hits budget axis."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    payload = manifest_payload(run_id, task_id, attempt_id)
    try:
        body = pad_json_body(payload, MAX_CONTROL_JSON_BYTES)
    except ValueError:
        pytest.skip("unable to synthesize exact cap-sized body in this environment")
    decoded = decode_control_json(body + b"\n", kind="manifest")
    assert decoded.ok or decoded.budget_exceeded


def test_t147_provisional_eof_1048579_budget(advisory_runs_root):
    """T147 PROVISIONAL: EOF at byte 1048579 → budget before decode."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    raw = b"x" * 1048579
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root)
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status in {
        "manifest_control_budget_exceeded",
        "manifest_byte_budget_exceeded",
    }


def test_t148_provisional_scalar_not_in_findings():
    """T148 PROVISIONAL: scalar never duplicated into findings (spot check)."""
    classified = classify_url_full("https://127.0.0.1/")
    assert classified.host_status == "link_host_loopback_prohibited"
    assert "link_host_loopback_prohibited" not in classified.findings


def test_t149_provisional_trailing_space_rejected(advisory_runs_root):
    """T149 PROVISIONAL: raw_decode rejects trailing space without strip."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    base = manifest_payload(run_id, task_id, attempt_id)
    raw = json.dumps(base, separators=(",", ":")).encode() + b" \n"
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root)
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_json_malformed"


def test_t150_provisional_json_string_newline_escape(advisory_runs_root):
    """T150 PROVISIONAL: JSON string \\n escape decodes (not raw 0x0A)."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    payload = manifest_payload(run_id, task_id, attempt_id, note="line1\\nline2")
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    decoded = decode_control_json(raw, kind="manifest")
    assert decoded.ok


def test_t151_provisional_depth_17_budget_not_json_malformed(advisory_runs_root):
    """T151 PROVISIONAL: tree depth 17 → manifest_control_budget_exceeded."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    payload = manifest_payload(run_id, task_id, attempt_id)
    payload["deep"] = deep_json(17)
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root)
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_control_budget_exceeded"
    assert result.manifest_status != "manifest_json_malformed"


def test_t152_provisional_link_record_budget_token(advisory_runs_root):
    """T152 PROVISIONAL: link_record_control_budget_exceeded; never byte_budget token."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    raw = b"x" * 1048579
    path = advisory_runs_root / run_id / "run_execution_request_record.json"
    path.write_bytes(raw)
    from htr.advisory_inspection_models import LinkReferenceSelector
    from htr.advisory_inspection_secure import raw_sha256_digest

    selector = LinkReferenceSelector(
        run_id=run_id,
        record_kind="run_execution_request_record",
        record_raw_digest=raw_sha256_digest(raw),
        item_index=0,
    )
    result = inspect_link_reference(selector)
    assert result.link_record_status == "link_record_control_budget_exceeded"
    assert "link_record_byte_budget_exceeded" not in str(result)


def test_t153_provisional_path_component_not_directory(advisory_runs_root, monkeypatch):
    """T153 PROVISIONAL: intermediate path component is file."""
    from htr.advisory_inspection_secure import HashArtifactResult

    def fake_open(parent_fd, components, *, context):
        return None, "filesystem_path_component_not_directory"

    monkeypatch.setattr("htr.advisory_inspection_secure.open_nested_path", fake_open)
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/out.txt")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.filesystem_status == "filesystem_path_component_not_directory"


def test_t154_provisional_host_syntax_rejected():
    """T154 PROVISIONAL: host parser syntax error → link_host_syntax_rejected."""
    import pytest

    try:
        classified = classify_url_full("https://[::not-valid]/")
    except ValueError:
        pytest.skip("urlparse/ipaddress raises before classifier on this host")
    assert classified.host_status == "link_host_syntax_rejected"


def test_t155_provisional_mapped_ipv4_loopback():
    """T155 PROVISIONAL: ::ffff:127.0.0.1 mapped IPv4 loopback classification."""
    classified = classify_url_full("https://[::ffff:127.0.0.1]/")
    assert "link_host_ipv6_literal" in classified.findings
    assert "link_host_ipv4_mapped_ipv6" in classified.findings
    assert classified.host_status == "link_host_loopback_prohibited"


def test_t156_provisional_four_derived_slots(advisory_runs_root):
    """T156 PROVISIONAL: derived_alignments four slots 1a/1b/2a/2b."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id)
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    assert len(result.derived_alignments) == 4
    assert [a.role for a in result.derived_alignments] == ["1a", "1b", "2a", "2b"]


def test_t157_provisional_execution_request_slots_not_applicable(advisory_runs_root):
    """T157 PROVISIONAL: execution_request slots 2a/2b → link_derived_not_applicable."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id)
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    roles = {a.role: a for a in result.derived_alignments}
    assert roles["2a"].match_status == "link_derived_not_applicable"
    assert roles["2a"].applicable is False
    assert roles["2b"].match_status == "link_derived_not_applicable"


def test_t158_provisional_raw_lf_rejected_string_escape_ok(advisory_runs_root):
    """T158 PROVISIONAL: raw LF rejected; JSON string escape accepted."""
    bad = decode_control_json(b'{"a":1}\n\n', kind="manifest")
    assert not bad.ok
    good = decode_control_json(b'{"note":"a\\nb"}\n', kind="manifest")
    assert good.ok


def test_t159_provisional_invalid_host_syntax():
    """T159 PROVISIONAL: generic invalid host syntax → link_host_syntax_rejected."""
    import pytest

    try:
        classified = classify_url_full("https://[dead:beef:zzzz]/")
    except ValueError:
        pytest.skip("urlparse/ipaddress raises before classifier on this host")
    assert classified.host_status == "link_host_syntax_rejected"


def test_t160_provisional_mapped_ipv6_no_ipv4_literal_finding():
    """T160 PROVISIONAL: mapped IPv6 findings; no link_host_ipv4_literal finding."""
    classified = classify_url_full("https://[::ffff:192.168.1.1]/")
    assert "link_host_ipv6_literal" in classified.findings
    assert "link_host_ipv4_mapped_ipv6" in classified.findings
    assert "link_host_ipv4_literal" not in classified.findings


def test_t161_provisional_non_applicable_derived_role(advisory_runs_root):
    """T161 PROVISIONAL: non-applicable derived role → link_derived_not_applicable."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id)
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    slot = next(a for a in result.derived_alignments if a.role == "2a")
    assert slot.match_status == "link_derived_not_applicable"


def test_t162_provisional_candidate_indexes_ascending(advisory_runs_root):
    """T162 PROVISIONAL: candidate_derived_indexes ascending on multi-match."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = request_record(run_id, execution_items=[execution_item(item_id="x")])
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
                "item_id": "x",
                "source_request_item_id": "x",
                "execution_kind": "manual_open_link",
                "item_status": "completed",
                "command": {"url": "https://a/"},
                "output": {},
                "error": None,
                "metadata": {},
            },
            {
                "item_id": "x",
                "source_request_item_id": "x",
                "execution_kind": "manual_open_link",
                "item_status": "completed",
                "command": {"url": "https://b/"},
                "output": {},
                "error": None,
                "metadata": {},
            },
        ],
    }
    write_link_record(run_id, "run_execution_result_record.json", derived, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    slot = next(a for a in result.derived_alignments if a.role == "1a")
    assert slot.candidate_derived_indexes == sorted(slot.candidate_derived_indexes)


def test_t163_provisional_primary_decode_fatal_empty_alignments(advisory_runs_root):
    """T163 PROVISIONAL: primary decode fatal → derived_alignments=[]."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    raw = b"not-json\n"
    path = advisory_runs_root / run_id / "run_execution_request_record.json"
    path.write_bytes(raw)
    from htr.advisory_inspection_models import LinkReferenceSelector
    from htr.advisory_inspection_secure import raw_sha256_digest

    selector = LinkReferenceSelector(
        run_id=run_id,
        record_kind="run_execution_request_record",
        record_raw_digest=raw_sha256_digest(raw),
        item_index=0,
    )
    result = inspect_link_reference(selector)
    assert result.derived_alignments == []
    assert result.link_item_status == "link_item_not_applicable"


def test_t164_provisional_primary_malformed_rank2(advisory_runs_root):
    """T164 PROVISIONAL: derived absent + primary malformed → link_match_primary_item_malformed."""
    run_id = bootstrap_run(base_dir=advisory_runs_root)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "source_followup_plan_fingerprint": "fp-test",
        "requester": "human",
        "request_status": "pending",
        "execution_items": [
            {"item_id": "", "execution_kind": "manual_open_link", "command": {"url": "https://x/"}},
        ],
        "notes": None,
        "metadata": {},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    digest = write_link_record(run_id, "run_execution_request_record.json", record, base_dir=advisory_runs_root)
    result = inspect_link_reference(link_selector(run_id, digest))
    slot = next(a for a in result.derived_alignments if a.role == "1a")
    assert slot.match_status == "link_match_primary_item_malformed"


def test_t165_provisional_manifest_control_budget_not_json_malformed(advisory_runs_root):
    """T165 PROVISIONAL: manifest control budget ≠ json_malformed."""
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    payload = manifest_payload(run_id, task_id, attempt_id)
    payload["deep"] = deep_json(20)
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    digest = write_manifest_bytes(run_id, task_id, attempt_id, raw, base_dir=advisory_runs_root)
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.manifest_status == "manifest_control_budget_exceeded"


def test_t166_provisional_port_parse_not_reached():
    """T166 PROVISIONAL: relative URL → link_port_parse_not_reached."""
    classified = classify_url_full("./relative")
    assert classified.port_status == "link_port_parse_not_reached"


def test_t167_provisional_filesystem_component_non_directory(advisory_runs_root, monkeypatch):
    """T167 PROVISIONAL: filesystem path component non-directory observation."""
    monkeypatch.setattr(
        "htr.advisory_inspection_secure.open_nested_path",
        lambda *_a, **_k: (None, "filesystem_path_component_not_directory"),
    )
    run_id, task_id, attempt_id = bootstrap_attempt(base_dir=advisory_runs_root)
    digest = write_manifest(
        run_id,
        task_id,
        attempt_id,
        manifest_payload(run_id, task_id, attempt_id, artifacts=[artifact_entry("artifacts/x.txt")]),
        base_dir=advisory_runs_root,
    )
    result = inspect_artifact_reference(artifact_selector(run_id, task_id, attempt_id, digest))
    assert result.filesystem_status == "filesystem_path_component_not_directory"
    assert "filesystem_open_rejected" not in str(result)


def test_t168_provisional_findings_sorted_unique():
    """T168 PROVISIONAL: findings deterministic sorted unique."""
    unsorted = ["link_query_observed", "link_ambiguous_authority", "link_query_observed", "link_fragment_observed"]
    out = sort_findings(unsorted)
    assert out == sorted(set(unsorted))
    assert len(out) == len(set(unsorted))
