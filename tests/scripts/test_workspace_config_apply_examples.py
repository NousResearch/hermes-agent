from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "workspace_config_apply_examples.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_workspace_config_apply_examples", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_schema(tmp_path: Path) -> Path:
    project = tmp_path / "api"
    project.mkdir()
    (project / ".env.example").write_text("DATABASE_URL=\n# local comment\n", encoding="utf-8")
    schema = {
        "root": str(tmp_path),
        "summary": {"projects": 1, "keys": 3},
        "projects": [
            {
                "project": str(project),
                "entries": [
                    {
                        "key": "DATABASE_URL",
                        "status": "present_in_example",
                        "action": "keep",
                        "secret": False,
                        "confidence": "high",
                    },
                    {
                        "key": "STRIPE_SECRET_KEY",
                        "status": "missing_from_example",
                        "action": "review_add_to_env_example",
                        "secret": True,
                        "confidence": "medium",
                    },
                    {
                        "key": "UNUSED_SECRET",
                        "status": "unused_env",
                        "action": "review_remove_or_document",
                        "secret": True,
                        "confidence": "medium",
                    },
                ],
            }
        ],
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return schema_path


def test_dry_run_does_not_write_env_example(tmp_path):
    module = _load_module()
    schema_path = _write_schema(tmp_path)
    example = tmp_path / "api" / ".env.example"
    before = example.read_text(encoding="utf-8")

    result = module.apply_examples(schema_path, apply=False, report_dir=None)

    assert result["summary"]["planned_env_example_writes"] == 1
    assert result["summary"]["applied_env_example_writes"] == 0
    assert example.read_text(encoding="utf-8") == before


def test_apply_writes_managed_block_and_preserves_local_content(tmp_path):
    module = _load_module()
    schema_path = _write_schema(tmp_path)
    example = tmp_path / "api" / ".env.example"

    result = module.apply_examples(schema_path, apply=True, report_dir=None)
    text = example.read_text(encoding="utf-8")

    assert result["summary"]["applied_env_example_writes"] == 1
    assert "DATABASE_URL=" in text
    assert "# local comment" in text
    assert "# hermes-managed:start env-example" in text
    assert "STRIPE_SECRET_KEY=" in text
    assert "UNUSED_SECRET" not in text


def test_existing_managed_block_is_normalized_once(tmp_path):
    module = _load_module()
    schema_path = _write_schema(tmp_path)
    example = tmp_path / "api" / ".env.example"
    example.write_text(
        "DATABASE_URL=\n# hermes-managed:start env-example\nOLD_KEY=\n# hermes-managed:end env-example\n",
        encoding="utf-8",
    )

    module.apply_examples(schema_path, apply=True, report_dir=None)
    text = example.read_text(encoding="utf-8")

    assert text.count("# hermes-managed:start env-example") == 1
    assert "OLD_KEY=" in text
    assert "STRIPE_SECRET_KEY=" in text


def test_new_keys_are_merged_with_existing_managed_keys(tmp_path):
    module = _load_module()
    schema_path = _write_schema(tmp_path)
    example = tmp_path / "api" / ".env.example"
    example.write_text(
        "DATABASE_URL=\n# hermes-managed:start env-example\nEXISTING_MANAGED_KEY=\n# hermes-managed:end env-example\n",
        encoding="utf-8",
    )

    module.apply_examples(schema_path, apply=True, report_dir=None)
    text = example.read_text(encoding="utf-8")

    assert "EXISTING_MANAGED_KEY=" in text
    assert "STRIPE_SECRET_KEY=" in text
    assert text.count("# hermes-managed:start env-example") == 1


def test_no_eligible_keys_keeps_existing_managed_block(tmp_path):
    module = _load_module()
    schema_path = _write_schema(tmp_path)
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    for entry in data["projects"][0]["entries"]:
        if entry["key"] == "STRIPE_SECRET_KEY":
            entry["status"] = "present_in_example"
    schema_path.write_text(json.dumps(data), encoding="utf-8")
    example = tmp_path / "api" / ".env.example"
    example.write_text(
        "DATABASE_URL=\n# hermes-managed:start env-example\nSTRIPE_SECRET_KEY=\n# hermes-managed:end env-example\n",
        encoding="utf-8",
    )

    result = module.apply_examples(schema_path, apply=False, report_dir=None)

    assert result["summary"]["planned_env_example_writes"] == 0
    assert "STRIPE_SECRET_KEY=" in example.read_text(encoding="utf-8")


def test_report_generation_has_no_secret_values(tmp_path):
    module = _load_module()
    schema_path = _write_schema(tmp_path)
    report_dir = tmp_path / "reports"

    result = module.apply_examples(schema_path, apply=True, report_dir=report_dir)
    rendered = json.dumps(result, ensure_ascii=False)
    for path in report_dir.iterdir():
        rendered += path.read_text(encoding="utf-8")

    assert "STRIPE_SECRET_KEY" in rendered
    assert "super-secret-password" not in rendered
    assert "hidden-value" not in rendered
    assert (report_dir / "env-example-remediation.json").exists()
