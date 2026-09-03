"""Task 29 Phase I — link inspection tests (subset T100–T108)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from htr.advisory_inspection_constants import SUPPLEMENTAL_FINDING_TOKENS
from htr.advisory_inspection_models import LinkReferenceSelector
from htr.advisory_inspection_secure import raw_sha256_digest
from htr import contracts, io, paths
from htr.ids import new_run_id
from htr.link_inspection import inspect_link_reference

_FORBIDDEN_NAMES = frozenset(
    {
        "read_json",
        "evaluate_run_seal",
        "parse_strict_json_bytes",
    }
)


def _write_record(run_id: str, filename: str, record: dict) -> str:
    target = paths.run_root(run_id) / filename
    raw = json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    target.write_bytes(raw)
    return raw_sha256_digest(raw)


def _bootstrap_run() -> str:
    run_id = new_run_id()
    io.create_run_workspace(run_id)
    return run_id


def _execution_item(**overrides):
    item = {
        "item_id": "exec-1",
        "source_followup_item_id": "followup-1",
        "title": "Open dashboard",
        "execution_kind": "manual_open_link",
        "command": {"url": "https://example.com"},
        "approval_reason": None,
        "metadata": {},
    }
    item.update(overrides)
    return item


def _request_record(run_id: str, **kwargs):
    items = kwargs.pop("execution_items", [_execution_item()])
    return contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint="fp-test",
        execution_items=items,
        **kwargs,
    )


def _selector(run_id: str, digest: str, *, item_index: int = 0) -> LinkReferenceSelector:
    return LinkReferenceSelector(
        run_id=run_id,
        record_kind="run_execution_request_record",
        record_raw_digest=digest,
        item_index=item_index,
    )


def test_t100_manual_open_link_without_url():
    run_id = _bootstrap_run()
    record = _request_record(
        run_id,
        execution_items=[_execution_item(command={})],
    )
    digest = _write_record(run_id, "run_execution_request_record.json", record)
    result = inspect_link_reference(_selector(run_id, digest))
    assert result.link_item_status == "link_url_absent"


def test_t101_url_not_string():
    run_id = _bootstrap_run()
    record = _request_record(
        run_id,
        execution_items=[_execution_item(command={"url": 123})],
    )
    digest = _write_record(run_id, "run_execution_request_record.json", record)
    result = inspect_link_reference(_selector(run_id, digest))
    assert result.link_item_status == "link_url_not_string"


def test_t102_command_missing():
    run_id = _bootstrap_run()
    item = _execution_item()
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
    digest = _write_record(run_id, "run_execution_request_record.json", record)
    result = inspect_link_reference(_selector(run_id, digest))
    assert result.link_item_status == "link_command_malformed"


def test_t103_primary_url_conflict_with_derived():
    run_id = _bootstrap_run()
    primary = _execution_item(
        item_id="item-a",
        command={"url": "https://primary.example/"},
    )
    record = _request_record(run_id, execution_items=[primary])
    digest = _write_record(run_id, "run_execution_request_record.json", record)

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
    _write_record(run_id, "run_execution_result_record.json", derived)

    result = inspect_link_reference(_selector(run_id, digest))
    slot_1a = next(a for a in result.derived_alignments if a.role == "1a")
    assert slot_1a.match_status == "link_match_url_conflict"
    assert "link_primary_derived_conflict" in slot_1a.findings


def test_t104_item_index_oob():
    run_id = _bootstrap_run()
    record = _request_record(run_id)
    digest = _write_record(run_id, "run_execution_request_record.json", record)
    selector = LinkReferenceSelector(
        run_id=run_id,
        record_kind="run_execution_request_record",
        record_raw_digest=digest,
        item_index=99,
    )
    result = inspect_link_reference(selector)
    assert result.authority_status == "selector_item_index_out_of_range"


def test_t105_wrong_record_raw_digest():
    run_id = _bootstrap_run()
    record = _request_record(run_id)
    _write_record(run_id, "run_execution_request_record.json", record)
    selector = LinkReferenceSelector(
        run_id=run_id,
        record_kind="run_execution_request_record",
        record_raw_digest="sha256:" + "b" * 64,
        item_index=0,
    )
    result = inspect_link_reference(selector)
    assert result.authority_status == "selector_record_digest_mismatch"


def test_t106_invalid_record_kind():
    run_id = _bootstrap_run()
    selector = LinkReferenceSelector(
        run_id=run_id,
        record_kind="not_a_valid_kind",  # type: ignore[arg-type]
        record_raw_digest="sha256:" + "a" * 64,
        item_index=0,
    )
    result = inspect_link_reference(selector)
    assert result.authority_status == "selector_record_kind_invalid"


def test_t107_duplicate_item_id_scalar_only_not_finding():
    run_id = _bootstrap_run()
    record = _request_record(
        run_id,
        execution_items=[
            _execution_item(item_id="dup-1"),
            _execution_item(item_id="dup-1", title="second"),
        ],
    )
    digest = _write_record(run_id, "run_execution_request_record.json", record)
    result = inspect_link_reference(_selector(run_id, digest, item_index=0))
    assert result.link_item_status == "link_item_id_duplicate"
    assert "link_item_id_duplicate" not in result.findings
    assert "link_item_id_duplicate" not in SUPPLEMENTAL_FINDING_TOKENS


def test_t108_rerun_task_not_url_bearing():
    run_id = _bootstrap_run()
    record = _request_record(
        run_id,
        execution_items=[_execution_item(execution_kind="rerun_task", command={"task_id": "task_x"})],
    )
    digest = _write_record(run_id, "run_execution_request_record.json", record)
    result = inspect_link_reference(_selector(run_id, digest))
    assert result.link_item_status == "link_kind_not_url_bearing"


def test_t107_non_url_bearing_duplicate_stays_kind_not_url_bearing():
    run_id = _bootstrap_run()
    record = _request_record(
        run_id,
        execution_items=[
            _execution_item(execution_kind="rerun_task", command={"task_id": "t1"}, item_id="dup-x"),
            _execution_item(execution_kind="rerun_task", command={"task_id": "t2"}, item_id="dup-x"),
        ],
    )
    digest = _write_record(run_id, "run_execution_request_record.json", record)
    result = inspect_link_reference(_selector(run_id, digest, item_index=0))
    assert result.link_item_status == "link_kind_not_url_bearing"


def test_derived_alignments_four_fixed_slots_when_bound():
    run_id = _bootstrap_run()
    record = _request_record(run_id)
    digest = _write_record(run_id, "run_execution_request_record.json", record)
    result = inspect_link_reference(_selector(run_id, digest))
    assert len(result.derived_alignments) == 4
    assert [a.role for a in result.derived_alignments] == ["1a", "1b", "2a", "2b"]
    assert result.derived_alignments[2].match_status == "link_derived_not_applicable"
    assert result.derived_alignments[3].match_status == "link_derived_not_applicable"


def _collect_source_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_forbidden_api_spy_link_inspection():
    repo_root = Path(__file__).resolve().parents[2]
    names = _collect_source_names(repo_root / "htr" / "link_inspection.py")
    assert _FORBIDDEN_NAMES.isdisjoint(names)
