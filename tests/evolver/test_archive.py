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
    assert (record.task.input, record.failure_class) == ("repair the parser", "incomplete")
    assert record_from_fix_result(failed, task_id="producer-task").task.task_id == "producer-task"
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


def test_ingest_accepts_native_producer_rows_without_losing_error_or_task_identity(tmp_path):
    mini_error = {
        "conversations": [],
        "completed": False,
        "api_calls": 0,
        "error": "sandbox setup failed",
        "metadata": {"timestamp": "2026-09-14T01:02:03"},
    }
    mini_failure = {
        "conversations": [{"from": "human", "value": "repair mini"}],
        "completed": False,
        "metadata": {"model": "mini-model", "timestamp": "2026-09-14T01:02:04"},
    }
    batch_failure = {
        "prompt_index": 0,
        "conversations": [{"from": "human", "value": "repair batch A"}],
        "completed": False,
        "metadata": {"batch_num": 3, "model": "batch-model", "timestamp": "2026-09-14T01:02:05"},
    }
    other_batch_failure = {
        **batch_failure,
        "conversations": [{"from": "human", "value": "repair batch B"}],
    }

    first_source = tmp_path / "run-a" / "batch_3.jsonl"
    first_source.parent.mkdir()
    first_source.write_text(
        "\n".join(json.dumps(row) for row in (mini_error, mini_failure, batch_failure)) + "\n",
        encoding="utf-8",
    )
    second_source = tmp_path / "run-b" / "batch_3.jsonl"
    second_source.parent.mkdir()
    second_source.write_text(json.dumps(other_batch_failure) + "\n", encoding="utf-8")

    archive = PathologyArchive(tmp_path / "pathologies.jsonl")
    assert archive.ingest_fix_results(tmp_path) == 4
    records = list(archive)
    assert records[0].failure_class == "runner_error"
    assert records[0].trace.spans[0].attributes["hermes.content"] == "sandbox setup failed"
    assert all(record.observed_at.endswith("+00:00") for record in records)
    batch_records = [record for record in records if record.task.input.startswith("repair batch")]
    assert len({record.task.task_id for record in batch_records}) == 2


def test_ingest_assigns_distinct_repeatable_ids_to_trace_less_runner_errors(tmp_path):
    source = tmp_path / "fix_results" / "errors.jsonl"
    source.parent.mkdir()
    rows = (
        {"completed": False, "error": "sandbox setup failed", "conversations": []},
        {"completed": False, "error": "sandbox cleanup failed", "conversations": []},
    )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    first = PathologyArchive(tmp_path / "first.jsonl")
    second = PathologyArchive(tmp_path / "second.jsonl")
    assert first.ingest_fix_results(tmp_path / "fix_results", harness_version="factory-v1") == 2
    assert second.ingest_fix_results(tmp_path / "fix_results", harness_version="factory-v1") == 2

    first_ids = [record.task.task_id for record in first]
    assert len(set(first_ids)) == 2
    assert first_ids == [record.task.task_id for record in second]

    separate_root_ids = []
    for root_name, error in (("run-a", "sandbox launch failed"), ("run-b", "sandbox teardown failed")):
        separate_source = tmp_path / root_name / "errors.jsonl"
        separate_source.parent.mkdir()
        separate_source.write_text(
            json.dumps({"completed": False, "error": error, "conversations": []}) + "\n",
            encoding="utf-8",
        )
        separate_archive = PathologyArchive(tmp_path / f"{root_name}.jsonl")
        assert separate_archive.ingest_fix_results(separate_source.parent, harness_version="factory-v1") == 1
        separate_root_ids.append(next(iter(separate_archive)).task.task_id)

    assert len(set(separate_root_ids)) == 2
