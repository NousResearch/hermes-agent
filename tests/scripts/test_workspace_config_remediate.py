from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "workspace_config_remediate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_workspace_config_remediate", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_project(root: Path, *, gitignore: str = "dist/\n") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".env").write_text(
        "DATABASE_URL=postgres://user:super-secret-password@localhost/app\nUNUSED_SECRET=hidden-value\n",
        encoding="utf-8",
    )
    (root / ".env.example").write_text("DATABASE_URL=\n", encoding="utf-8")
    (root / ".gitignore").write_text(gitignore, encoding="utf-8")
    (root / "app.py").write_text(
        "import os\nprint(os.getenv('DATABASE_URL'))\nprint(os.getenv('STRIPE_SECRET_KEY'))\n",
        encoding="utf-8",
    )


def test_dry_run_does_not_write_gitignore(tmp_path):
    module = _load_module()
    project = tmp_path / "api"
    _write_project(project)
    before = (project / ".gitignore").read_text(encoding="utf-8")

    result = module.remediate_workspace(tmp_path, apply=False, report_dir=None)

    assert result["summary"]["planned_gitignore_writes"] == 1
    assert result["summary"]["applied_gitignore_writes"] == 0
    assert (project / ".gitignore").read_text(encoding="utf-8") == before


def test_apply_writes_managed_gitignore_block_without_removing_local_lines(tmp_path):
    module = _load_module()
    project = tmp_path / "api"
    _write_project(project, gitignore="dist/\n# local rule\n")

    result = module.remediate_workspace(tmp_path, apply=True, report_dir=None)
    text = (project / ".gitignore").read_text(encoding="utf-8")

    assert result["summary"]["applied_gitignore_writes"] == 1
    assert "dist/" in text
    assert "# local rule" in text
    assert "# hermes-managed:start workspace-config" in text
    assert ".env" in text
    assert "!.env.example" in text


def test_existing_managed_block_is_replaced_once(tmp_path):
    module = _load_module()
    project = tmp_path / "api"
    _write_project(
        project,
        gitignore="# hermes-managed:start workspace-config\nold\n# hermes-managed:end workspace-config\n",
    )

    module.remediate_workspace(tmp_path, apply=True, report_dir=None)
    text = (project / ".gitignore").read_text(encoding="utf-8")

    assert text.count("# hermes-managed:start workspace-config") == 1
    assert "old" not in text


def test_report_generation_does_not_include_secret_values(tmp_path):
    module = _load_module()
    _write_project(tmp_path / "api")
    report_dir = tmp_path / "reports"

    result = module.remediate_workspace(tmp_path, apply=False, report_dir=report_dir)

    summary = (report_dir / "summary.json").read_text(encoding="utf-8")
    registry = json.loads((report_dir / "env-registry.json").read_text(encoding="utf-8"))
    markdown = (report_dir / "remediation-report.md").read_text(encoding="utf-8")
    rendered = json.dumps(result, ensure_ascii=False) + summary + json.dumps(registry) + markdown
    assert "super-secret-password" not in rendered
    assert "hidden-value" not in rendered
    assert "DATABASE_URL" in rendered
    assert "STRIPE_SECRET_KEY" in rendered
