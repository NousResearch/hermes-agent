from __future__ import annotations

import json
import importlib.util
import os
import shutil
import socketserver
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "media"
    / "youtube-automation-agent"
    / "scripts"
    / "youtube_automation_helper.py"
)
SKILL_DIR = SCRIPT_PATH.parents[1]
SKILL_PATH = SKILL_DIR / "SKILL.md"


def load_helper_module():
    spec = importlib.util.spec_from_file_location(
        "youtube_automation_helper_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_ready_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for rel in [
        "schedules/daily-automation.js",
        "utils/credential-manager.js",
        "config/credentials.json",
        "config/tokens.json",
    ]:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    for rel in ["index.js", "setup.js", "test.js"]:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("console.log('ok')\n", encoding="utf-8")
    (repo / ".env").write_text("# test\n", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "package.json").write_text(
        json.dumps({"scripts": {"start": "node index.js"}}), encoding="utf-8"
    )
    return repo


def run_helper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_inspect_reports_missing_script_targets(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "schedules").mkdir(parents=True)
    (repo / "utils").mkdir(parents=True)
    (repo / "config").mkdir(parents=True)

    (repo / "index.js").write_text("console.log('ok')\n", encoding="utf-8")
    (repo / "setup.js").write_text("console.log('setup')\n", encoding="utf-8")
    (repo / "test.js").write_text("console.log('test')\n", encoding="utf-8")
    (repo / "schedules" / "daily-automation.js").write_text("module.exports = {}\n", encoding="utf-8")
    (repo / "utils" / "credential-manager.js").write_text("module.exports = {}\n", encoding="utf-8")
    (repo / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "start": "node index.js",
                    "workflow:daily": "node workflows/daily-content-pipeline.js",
                    "db:init": "node database/init.js",
                }
            }
        ),
        encoding="utf-8",
    )

    proc = run_helper("inspect", "--repo", str(repo), "--json")
    assert proc.returncode == 1

    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "blocked"
    assert {item["target"] for item in payload["missing_script_targets"]} == {
        "workflows/daily-content-pipeline.js",
        "database/init.js",
    }


def test_inspect_reports_missing_package_metadata_without_traceback(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    proc = run_helper("inspect", "--repo", str(repo), "--json")

    assert proc.returncode == 1
    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "blocked"
    assert payload["package_metadata"] == {
        "ok": False,
        "error": "package.json is missing",
    }


def test_inspect_reports_malformed_package_metadata_without_traceback(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text("{not valid json", encoding="utf-8")

    proc = run_helper("inspect", "--repo", str(repo), "--json")

    assert proc.returncode == 1
    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "blocked"
    assert payload["package_metadata"]["ok"] is False
    assert payload["package_metadata"]["error"].startswith("package.json is malformed:")


def test_inspect_reports_non_utf8_package_metadata_without_traceback(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_bytes(b"\xff")

    proc = run_helper("inspect", "--repo", str(repo), "--json")

    assert proc.returncode == 1
    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "blocked"
    assert payload["package_metadata"] == {
        "ok": False,
        "error": "package.json is malformed: invalid UTF-8 at byte 0",
    }


@pytest.mark.parametrize("scripts", [[], {"start": 42}])
def test_inspect_reports_invalid_scripts_metadata_without_traceback(
    tmp_path: Path, scripts: object
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text(
        json.dumps({"scripts": scripts}), encoding="utf-8"
    )

    proc = run_helper("inspect", "--repo", str(repo), "--json")

    assert proc.returncode == 1
    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "blocked"
    assert payload["package_metadata"]["ok"] is False
    assert payload["package_metadata"]["error"].startswith(
        "package.json is malformed:"
    )


def test_inspect_reports_missing_node_as_needs_setup(tmp_path: Path, monkeypatch):
    repo = make_ready_repo(tmp_path)
    helper = load_helper_module()
    monkeypatch.setattr(helper.shutil, "which", lambda command: None)

    report = helper.inspect_repo(repo)

    assert report["verdict"] == "needs-setup"
    node_check = next(item for item in report["checks"] if item["label"] == "command:node")
    assert node_check == {"label": "command:node", "ok": False, "detail": "node not found"}


def test_inspect_reports_failed_node_syntax_as_blocked(tmp_path: Path, monkeypatch):
    repo = make_ready_repo(tmp_path)
    helper = load_helper_module()
    monkeypatch.setattr(helper.shutil, "which", lambda command: "node")
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=1, stdout="", stderr="SyntaxError"
        ),
    )

    report = helper.inspect_repo(repo)

    assert report["verdict"] == "blocked"
    assert report["syntax_checks"]
    assert all(item["ok"] is False for item in report["syntax_checks"])


def test_skill_frontmatter_and_sections_follow_external_skill_contract():
    content = SKILL_PATH.read_text(encoding="utf-8")
    frontmatter = content.split("---", 2)[1]
    fields = {
        key.strip(): value.strip()
        for line in frontmatter.splitlines()
        if ":" in line and not line.startswith(" ")
        for key, value in [line.split(":", 1)]
    }

    description = fields["description"]
    assert len(description) <= 60
    assert description.endswith(".")
    assert fields["author"].startswith("Haithum Abdelfattah (@darkzOGx)")
    assert "platforms" in fields

    expected_sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [content.index(section) for section in expected_sections]
    assert positions == sorted(positions)


def test_named_profile_install_uses_active_hermes_home(tmp_path: Path):
    named_home = tmp_path / ".hermes" / "profiles" / "creator"
    installed_skill = named_home / "skills" / "media" / "youtube-automation-agent"
    shutil.copytree(SKILL_DIR, installed_skill)

    documented_files = [SKILL_PATH, *sorted((SKILL_DIR / "references").glob("*.md"))]
    documentation = "\n".join(path.read_text(encoding="utf-8") for path in documented_files)
    assert "~/.hermes/skills" not in documentation
    assert "$HERMES_HOME" in documentation
    assert "$SKILL_DIR/scripts/youtube_automation_helper.py" in documentation

    env = os.environ.copy()
    env["HERMES_HOME"] = str(named_home)
    installed_script = installed_skill / "scripts" / "youtube_automation_helper.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(installed_script),
            "init-run",
            "--channel",
            "Profile Channel",
            "--niche",
            "testing",
            "--audience",
            "creators",
            "--style",
            "educational",
            "--topic",
            "profile workflow",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    workspace = Path(json.loads(proc.stdout)["workspace"])
    assert workspace.parent == installed_skill / "data" / "runs"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            body = json.dumps({"status": "healthy", "initialized": True}).encode()
            self.send_response(200)
        elif self.path == "/schedule":
            body = b"[]"
            self.send_response(200)
        elif self.path == "/analytics":
            body = b"{}"
            self.send_response(200)
        else:
            body = b"not found"
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class _NonObjectHealthHandler(_Handler):
    def do_GET(self):  # noqa: N802
        if self.path != "/health":
            return super().do_GET()
        body = b"[]"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_probe_reports_healthy_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"

        proc = run_helper("probe", "--base-url", base_url, "--json")
        server.shutdown()
        thread.join(timeout=2)

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["healthy"] is True
    assert [item["endpoint"] for item in payload["endpoints"]] == ["/health", "/schedule", "/analytics"]
    assert all(item["ok"] for item in payload["endpoints"])


def test_probe_reports_non_object_health_json_as_unhealthy():
    with socketserver.TCPServer(("127.0.0.1", 0), _NonObjectHealthHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"

        proc = run_helper("probe", "--base-url", base_url, "--json")
        server.shutdown()
        thread.join(timeout=2)

    assert proc.returncode == 1
    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["healthy"] is False
    health = next(item for item in payload["endpoints"] if item["endpoint"] == "/health")
    assert health["ok"] is False


def test_init_run_and_stage_brief(tmp_path: Path):
    workspace = tmp_path / "run.json"
    proc = run_helper(
        "init-run",
        "--channel",
        "Ladera Labs",
        "--niche",
        "AI productivity",
        "--audience",
        "founders",
        "--style",
        "educational",
        "--frequency",
        "daily",
        "--topic",
        "agent workflows",
        "--output",
        str(workspace),
        "--json",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["workspace"] == str(workspace)
    assert payload["run"]["current_stage"] == "strategy"

    brief_proc = run_helper("brief", "--workspace", str(workspace), "--json")
    assert brief_proc.returncode == 0
    brief_payload = json.loads(brief_proc.stdout)
    assert brief_payload["brief"]["stage"] == "strategy"
    assert "selected topic" in brief_payload["brief"]["prompt"].lower()


def test_complete_stage_advances_to_next_stage(tmp_path: Path):
    workspace = tmp_path / "run.json"
    init_proc = run_helper(
        "init-run",
        "--channel",
        "Ladera Labs",
        "--niche",
        "AI productivity",
        "--audience",
        "founders",
        "--style",
        "educational",
        "--frequency",
        "daily",
        "--output",
        str(workspace),
        "--json",
    )
    assert init_proc.returncode == 0

    complete_proc = run_helper(
        "complete-stage",
        "--workspace",
        str(workspace),
        "--stage",
        "strategy",
        "--notes",
        "Selected workflow automation angle",
        "--artifacts-json",
        '{"selected_topic":"AI workflow automation","content_type":"Explainer"}',
        "--json",
    )
    assert complete_proc.returncode == 0
    payload = json.loads(complete_proc.stdout)
    assert payload["completed_stage"] == "strategy"
    assert payload["next_stage"] == "script"

    status_proc = run_helper("status", "--workspace", str(workspace), "--json")
    status_payload = json.loads(status_proc.stdout)
    assert status_payload["current_stage"] == "script"
    assert status_payload["stages"]["strategy"]["artifacts"]["selected_topic"] == "AI workflow automation"

    export_proc = run_helper("export", "--workspace", str(workspace), "--json")
    export_payload = json.loads(export_proc.stdout)
    assert export_payload["completed_stages"] == ["strategy"]
    assert "strategy" in export_payload["deliverables"]
