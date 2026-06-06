from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "workspace_config_schema.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_workspace_config_schema", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_project(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".env").write_text(
        "DATABASE_URL=postgres://user:super-secret-password@localhost/app\n"
        "UNUSED_SECRET=hidden-value\n",
        encoding="utf-8",
    )
    (root / ".env.example").write_text("DATABASE_URL=\nEXAMPLE_ONLY_FLAG=false\n", encoding="utf-8")
    (root / ".gitignore").write_text(".env\n.env.*\n!.env.example\n", encoding="utf-8")
    (root / "app.py").write_text(
        "import os\n"
        "print(os.getenv('DATABASE_URL'))\n"
        "print(os.getenv('STRIPE_SECRET_KEY'))\n"
        "print(os.getenv('REVIEW_FLAG'))\n",
        encoding="utf-8",
    )


def test_schema_classifies_keys_into_review_buckets(tmp_path):
    module = _load_module()
    _write_project(tmp_path / "api")

    result = module.generate_workspace_schema(tmp_path, output_dir=None)

    entries = result["projects"][0]["entries"]
    statuses = {entry["key"]: entry["status"] for entry in entries}
    assert statuses["DATABASE_URL"] == "present_in_example"
    assert statuses["STRIPE_SECRET_KEY"] == "missing_from_example"
    assert statuses["UNUSED_SECRET"] == "unused_env"
    assert statuses["EXAMPLE_ONLY_FLAG"] == "review"


def test_schema_marks_likely_secrets_without_values(tmp_path):
    module = _load_module()
    _write_project(tmp_path / "api")

    result = module.generate_workspace_schema(tmp_path, output_dir=None)
    rendered = json.dumps(result, ensure_ascii=False)
    secret_entry = next(
        entry
        for entry in result["projects"][0]["entries"]
        if entry["key"] == "STRIPE_SECRET_KEY"
    )

    assert secret_entry["secret"] is True
    assert "super-secret-password" not in rendered
    assert "hidden-value" not in rendered


def test_schema_writes_json_markdown_and_queue(tmp_path):
    module = _load_module()
    _write_project(tmp_path / "api")
    output_dir = tmp_path / "reports"

    result = module.generate_workspace_schema(tmp_path, output_dir=output_dir)

    schema_json = output_dir / "env-schema.json"
    schema_md = output_dir / "env-schema.md"
    queue_md = output_dir / "next-remediation-queue.md"
    assert schema_json.exists()
    assert schema_md.exists()
    assert queue_md.exists()
    assert "STRIPE_SECRET_KEY" in queue_md.read_text(encoding="utf-8")
    assert "super-secret-password" not in schema_json.read_text(encoding="utf-8")
    assert result["summary"]["missing_from_example"] == 2


def test_text_report_is_compact(tmp_path):
    module = _load_module()
    _write_project(tmp_path / "api")

    result = module.generate_workspace_schema(tmp_path, output_dir=None)
    rendered = module.render_result(result)

    assert "Workspace env schema" in rendered
    assert "missing_from_example" in rendered
    assert "super-secret-password" not in rendered
