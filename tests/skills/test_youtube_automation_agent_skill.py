from __future__ import annotations

import json
import socketserver
import subprocess
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "media"
    / "youtube-automation-agent"
    / "scripts"
    / "youtube_automation_helper.py"
)


def run_helper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT_PATH), *args],
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
