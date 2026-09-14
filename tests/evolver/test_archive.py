from __future__ import annotations

import json

from evolver.archive import PathologyArchive, record_from_fix_result


def test_fix_result_adapter_preserves_canonical_task_trace_failure_contract(tmp_path):
    failed = {
        "prompt_index": 7,
        "conversations": [
            {"from": "human", "value": "repair the parser"},
            {"from": "gpt", "value": "attempt"},
        ],
        "completed": False,
        "metadata": {"model": "test-model", "timestamp": "2026-09-14T00:00:00+00:00"},
    }
    successful = dict(failed, completed=True)

    record = record_from_fix_result(failed)
    assert record is not None
    assert (record.task.task_id, record.task.input, record.failure_class) == (
        "task-7",
        "repair the parser",
        "incomplete",
    )
    assert record.trace.trace_id == record.trace.spans[0].trace_id
    assert record.trace.spans[1].parent_span_id == record.trace.spans[0].span_id
    assert record_from_fix_result(successful) is None

    source = tmp_path / "fix_results" / "runs.jsonl"
    source.parent.mkdir()
    source.write_text("\n".join(json.dumps(row) for row in (failed, successful)) + "\n", encoding="utf-8")
    archive = PathologyArchive(tmp_path / "pathologies.jsonl")
    assert archive.ingest_fix_results(source, harness_version="factory-v1") == 1
    loaded = list(archive)
    assert len(loaded) == 1
    assert loaded[0].task.harness_version == "factory-v1"
    assert loaded[0].to_dict() == list(PathologyArchive(archive.path))[0].to_dict()
