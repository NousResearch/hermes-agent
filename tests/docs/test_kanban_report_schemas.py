import json
from pathlib import Path

from hermes_cli import kanban_db as kb

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "docs" / "kanban" / "reports"


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _assert_required_fields(schema: dict, example: dict, prefix: str = "") -> None:
    for field in schema.get("required", []):
        assert field in example, f"{prefix}{field} is required"

    for name, subschema in schema.get("properties", {}).items():
        if name not in example:
            continue
        value = example[name]
        location = f"{prefix}{name}."
        if "const" in subschema:
            assert value == subschema["const"], f"{prefix}{name} must match const"
        if "enum" in subschema:
            assert value in subschema["enum"], f"{prefix}{name} must be in enum"
        if subschema.get("type") == "object":
            assert isinstance(value, dict), f"{prefix}{name} must be object"
            _assert_required_fields(subschema, value, location)
        if subschema.get("type") == "array":
            assert isinstance(value, list), f"{prefix}{name} must be array"
            item_schema = subschema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    if item_schema.get("type") == "object":
                        assert isinstance(item, dict), f"{prefix}{name}[{index}] must be object"
                        _assert_required_fields(item_schema, item, f"{prefix}{name}[{index}].")
                    elif "enum" in item_schema:
                        assert item in item_schema["enum"], f"{prefix}{name}[{index}] must be in enum"


def test_worker_handoff_example_matches_required_schema_fields_and_marker():
    schema = _load_json(REPORTS / "schemas" / "worker-handoff-report.schema.json")
    example = _load_json(REPORTS / "samples" / "worker-handoff-report.example.json")

    _assert_required_fields(schema, example)

    assert example["changed_files"]
    assert example["pre_existing_dirty_files"]
    assert example["commands"]
    assert all(isinstance(command["exit_code"], int) for command in example["commands"])
    assert example["schema_changed"] is True
    assert "OMX" in example["route_used"]

    review = example["review"]
    assert review["required"] is True
    assert review["marker"] == f"review_required: {review['report_path']}"
    assert review["marker"].startswith("review_required: ")
    assert review["summary"]


def test_reviewer_verdict_example_matches_required_schema_fields_and_enums():
    schema = _load_json(REPORTS / "schemas" / "reviewer-verdict.schema.json")
    example = _load_json(REPORTS / "samples" / "reviewer-verdict.example.json")

    _assert_required_fields(schema, example)

    assert set(schema["properties"]["verdict"]["enum"]) == {
        "PASS",
        "REQUEST_CHANGES",
        "PARTIAL",
        "BLOCKED",
        "NEEDS_HUMAN_APPROVAL",
    }
    assert example["review_task_id"]
    assert example["source_task_id"]
    assert example.get("task_id", example["source_task_id"]) == example["source_task_id"]
    assert isinstance(example["safe_to_merge"], bool)
    assert isinstance(example["safe_to_deploy"], bool)
    assert isinstance(example["blocking_findings"], list)
    assert isinstance(example["non_blocking_findings"], list)
    assert example["evidence"]
    assert all(isinstance(command["exit_code"], int) for command in example["evidence"])
    assert example["commands"]
    assert all(isinstance(command["exit_code"], int) for command in example["commands"])
    assert example["next_action"]["type"] in schema["properties"]["next_action"]["properties"]["type"]["enum"]
    assert example["verdict"] in schema["properties"]["verdict"]["enum"]


def test_reviewer_verdict_schema_covers_transition_engine_gate_fields():
    schema = _load_json(REPORTS / "schemas" / "reviewer-verdict.schema.json")
    properties = schema["properties"]

    assert {
        "review_task_id",
        "source_task_id",
        "safe_to_merge",
        "safe_to_deploy",
        "blocking_findings",
        "non_blocking_findings",
        "evidence",
        "next_action",
    } <= set(schema["required"])
    assert "Legacy alias for source_task_id" in properties["task_id"]["description"]
    assert "required" in properties["human_approval"]["required"]
    assert "reason" in properties["human_approval"]["required"]
    assert {
        "deploy",
        "restart",
        "delete_data",
        "credential_change",
    } <= set(properties["production_actions"]["properties"])
    assert kb._HIGH_RISK_NEXT_ACTION_TYPES <= set(properties["next_action"]["properties"]["type"]["enum"])


def test_all_kanban_report_json_files_are_valid_json():
    for path in sorted(REPORTS.rglob("*.json")):
        assert _load_json(path), f"{path} should contain a JSON object"
