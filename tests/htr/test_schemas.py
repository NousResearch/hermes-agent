import pytest

from htr.schemas import validate


def test_validate_run_manifest():
    validate(
        {
            "run_id": "run_20260718_abcd12",
            "created_at": "2026-07-18T07:00:00+00:00",
            "status": "created",
        },
        "run_manifest",
    )


def test_validate_task_status():
    validate(
        {
            "task_id": "task_20260718_abcd12",
            "run_id": "run_20260718_abcd12",
            "status": "created",
            "attempts": [],
        },
        "task_status",
    )


def test_validate_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing fields"):
        validate({"run_id": "run_20260718_abcd12"}, "run_manifest")


def test_validate_rejects_wrong_types():
    with pytest.raises(ValueError, match="attempts must be a list"):
        validate(
            {
                "task_id": "task_20260718_abcd12",
                "run_id": "run_20260718_abcd12",
                "status": "created",
                "attempts": "nope",
            },
            "task_status",
        )


def test_validate_task_card():
    validate(
        {
            "schema_version": "1",
            "run_id": "run_20260718_abcd12",
            "task_id": "task_20260718_abcd12",
            "title": "Demo",
            "instruction": "Do the thing",
            "created_at": "2026-07-18T08:00:00+00:00",
            "created_by": "architect",
            "inputs": {},
            "constraints": {},
            "acceptance": {},
            "metadata": {},
        },
        "task_card",
    )


def test_validate_attempt_result():
    validate(
        {
            "schema_version": "1",
            "run_id": "run_20260718_abcd12",
            "task_id": "task_20260718_abcd12",
            "attempt_id": "att_20260718_abcd12",
            "created_at": "2026-07-18T08:00:00+00:00",
            "produced_by": "worker",
            "summary": "done",
            "outputs": {},
            "artifacts": [],
            "metrics": {},
            "metadata": {},
        },
        "attempt_result",
    )


def test_validate_artifact_entry_rejects_bad_metadata():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        validate(
            {
                "path": "artifacts/a.txt",
                "kind": "file",
                "created_at": "2026-07-18T08:00:00+00:00",
                "metadata": [],
            },
            "artifact_entry",
        )
