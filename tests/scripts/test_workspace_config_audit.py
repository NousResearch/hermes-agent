from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "workspace_config_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_workspace_config_audit", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_project(root: Path, *, gitignore: str = "") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".env").write_text(
        "\n".join(
            [
                "DATABASE_URL=postgres://user:super-secret-password@localhost/app",
                "UNUSED_SECRET=unused-super-secret",
                "BROKEN LINE WITHOUT EQUALS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / ".env.example").write_text(
        "OPENAI_API_KEY=\nVITE_PUBLIC_URL=http://localhost:3000\n",
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(gitignore, encoding="utf-8")
    (root / "app.py").write_text(
        "import os\nDATABASE_URL = os.getenv('DATABASE_URL')\n",
        encoding="utf-8",
    )
    (root / "web.ts").write_text(
        "console.log(process.env.OPENAI_API_KEY, process.env['STRIPE_SECRET_KEY'], import.meta.env.VITE_PUBLIC_URL)\n"
        "const label = `${name}-${value}`\n",
        encoding="utf-8",
    )


def test_project_audit_reports_env_drift_without_secret_values(tmp_path):
    module = _load_module()
    _write_project(tmp_path)

    report = module.inspect_project(tmp_path)

    finding_codes = {finding["code"] for finding in report["findings"]}
    assert "gitignore-env-missing" in finding_codes
    assert "env-key-unused" in finding_codes
    assert "code-key-missing-from-env-example" in finding_codes

    unused = [finding for finding in report["findings"] if finding["code"] == "env-key-unused"]
    assert any("UNUSED_SECRET" in finding["evidence"] for finding in unused)

    rendered = module.render_report(report, "json")
    assert "super-secret-password" not in rendered
    assert "unused-super-secret" not in rendered
    assert json.loads(rendered)["project"] == str(tmp_path)


def test_gitignore_env_patterns_satisfy_read_only_check(tmp_path):
    module = _load_module()
    _write_project(tmp_path, gitignore=".env\n.env.*\n!.env.example\n")

    report = module.inspect_project(tmp_path)

    finding_codes = {finding["code"] for finding in report["findings"]}
    assert "gitignore-env-missing" not in finding_codes


def test_tracked_secret_env_file_is_critical(tmp_path):
    module = _load_module()
    _write_project(tmp_path, gitignore="")
    module.subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=module.subprocess.DEVNULL)
    module.subprocess.run(["git", "add", ".env"], cwd=tmp_path, check=True, stdout=module.subprocess.DEVNULL)

    report = module.inspect_project(tmp_path)

    tracked = [finding for finding in report["findings"] if finding["code"] == "tracked-env-file"]
    assert tracked
    assert tracked[0]["severity"] == "critical"


def test_workspace_audit_discovers_multiple_projects(tmp_path):
    module = _load_module()
    _write_project(tmp_path / "api", gitignore=".env\n.env.*\n!.env.example\n")
    _write_project(tmp_path / "web")

    report = module.inspect_workspace(tmp_path)

    project_names = {Path(project["project"]).name for project in report["projects"]}
    assert project_names == {"api", "web"}
    assert report["summary"]["projects"] == 2
    assert report["summary"]["findings"] >= 1


def test_workspace_audit_skips_symlink_loops(tmp_path):
    module = _load_module()
    project = tmp_path / "project"
    _write_project(project, gitignore=".env\n.env.*\n!.env.example\n")
    loop = project / "loop"
    try:
        loop.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        return

    report = module.inspect_workspace(tmp_path)

    project_names = {Path(project_report["project"]).name for project_report in report["projects"]}
    assert project_names == {"project"}


def test_template_placeholders_are_not_reported_as_env_keys(tmp_path):
    module = _load_module()
    _write_project(tmp_path, gitignore=".env\n.env.*\n!.env.example\n")

    report = module.inspect_project(tmp_path)

    assert "name" not in report["code_refs"]
    assert "value" not in report["code_refs"]


def test_text_report_is_human_readable_and_redacted(tmp_path):
    module = _load_module()
    _write_project(tmp_path)

    rendered = module.render_report(module.inspect_project(tmp_path), "text")

    assert "Workspace config audit" in rendered
    assert "gitignore-env-missing" in rendered
    assert "super-secret-password" not in rendered


def test_markdown_workspace_report_has_summary_and_no_secret_values(tmp_path):
    module = _load_module()
    _write_project(tmp_path / "api")
    _write_project(tmp_path / "web", gitignore=".env\n.env.*\n!.env.example\n")

    rendered = module.render_report(module.inspect_workspace(tmp_path), "markdown", max_findings=2)

    assert "# Workspace Config Audit" in rendered
    assert "| Projects | Projects With Findings | Total Findings |" in rendered
    assert "api" in rendered
    assert "super-secret-password" not in rendered
    assert "more finding(s) omitted" in rendered


def test_summary_report_omits_finding_detail(tmp_path):
    module = _load_module()
    _write_project(tmp_path / "api")

    report = module.inspect_workspace(tmp_path)
    rendered = module.render_report(report, "summary")

    assert "Workspace config audit summary" in rendered
    assert "gitignore-env-missing" not in rendered
    assert str(report["summary"]["projects"]) in rendered
