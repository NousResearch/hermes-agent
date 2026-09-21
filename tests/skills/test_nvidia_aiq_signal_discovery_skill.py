from __future__ import annotations

import re
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "optional-skills/mlops/nvidia-aiq-signal-discovery/SKILL.md"


def _parse_frontmatter(frontmatter: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, parsed)]
    pending_key: str | None = None
    pending_parent: dict[str, Any] | None = None
    for raw_line in frontmatter.splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            assert pending_key is not None
            assert pending_parent is not None
            if not isinstance(pending_parent.get(pending_key), list):
                pending_parent[pending_key] = []
            item: dict[str, Any] = {}
            pending_parent[pending_key].append(item)
            stack.append((indent, item))
            line = line[2:]
            parent = item
        key, _, value = line.partition(":")
        if not value.strip():
            child: dict[str, Any] = {}
            assert isinstance(parent, dict)
            parent[key] = child
            pending_key = key
            pending_parent = parent
            stack.append((indent, child))
            continue
        assert isinstance(parent, dict)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            parent[key] = [part.strip() for part in value[1:-1].split(",")]
        else:
            parent[key] = value
        pending_key = key
        pending_parent = parent
    return parsed


def _read() -> tuple[dict[str, Any], str]:
    content = SKILL.read_text(encoding="utf-8")
    match = re.search(r"\n---\s*\n", content[3:])
    assert match
    frontmatter = _parse_frontmatter(content[3 : match.start() + 3])
    return frontmatter, content


def test_skill_has_governed_frontmatter_and_sections() -> None:
    frontmatter, content = _read()
    assert frontmatter["name"] == "nvidia-aiq-signal-discovery"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    metadata = frontmatter["metadata"]
    assert isinstance(metadata, dict)
    hermes = metadata["hermes"]
    assert isinstance(hermes, dict)
    config = hermes["config"]
    assert isinstance(config, list)
    assert config[0]["key"] == "AIQ_SERVER_URL"
    for section in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert section in content


def test_skill_declares_no_send_and_native_bridge() -> None:
    _, content = _read()
    assert "scripts/bridge_nvidia_signal_result.py" in content
    assert "does not place orders" in content
    assert "research_only" in content
    assert "AIQ_SERVER_URL" in content


def test_aiq_client_exercises_submit_status_and_download(tmp_path, monkeypatch) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("aiq_client", REPO / "optional-skills/mlops/nvidia-aiq-signal-discovery/scripts/aiq_client.py")
    client = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(client)
    seen: list[tuple[str, str, object | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            size = int(self.headers["Content-Length"])
            seen.append((self.command, self.path, json.loads(self.rfile.read(size))))
            self._reply({"job_id": "job/opaque?1"})

        def do_GET(self):  # noqa: N802
            seen.append((self.command, self.path, None))
            self._reply({"status": "completed", "result": True})

        def _reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        request_file = tmp_path / "request.json"
        request_file.write_text('{"signal": "日本語"}\n', encoding="utf-8")
        output_file = tmp_path / "result.json"

        monkeypatch.setattr(sys, "argv", ["aiq_client.py", "--server", base, "submit", str(request_file)])
        assert client.main() == 0
        monkeypatch.setattr(sys, "argv", ["aiq_client.py", "--server", base, "status", "job/opaque?1"])
        assert client.main() == 0
        monkeypatch.setattr(
            sys,
            "argv",
            ["aiq_client.py", "--server", base, "download", "job/opaque?1", str(output_file)],
        )
        assert client.main() == 0
        assert json.loads(output_file.read_text(encoding="utf-8"))["result"] is True
        assert seen[0][2] == {"signal": "日本語"}
        assert seen[1][1] == "/v1/jobs/job%2Fopaque%3F1"
        assert seen[2][1] == "/v1/jobs/job%2Fopaque%3F1/result"
    finally:
        server.shutdown()
