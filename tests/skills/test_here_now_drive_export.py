from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


DRIVE_SCRIPT = Path("optional-skills/productivity/here-now/scripts/drive.sh")


class DriveHandler(BaseHTTPRequestHandler):
    files: list[str] = []
    downloaded_paths: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/drives/drv_test/files":
            body = json.dumps({"files": [{"path": path} for path in self.files]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        prefix = "/api/v1/drives/drv_test/files/"
        if parsed.path.startswith(prefix):
            self.downloaded_paths.append(parsed.path[len(prefix) :])
            body = b"drive contents\n"
            self.send_response(200)
            self.send_header("content-type", "application/octet-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def _run_export(tmp_path: Path, remote_path: str) -> subprocess.CompletedProcess[str]:
    DriveHandler.files = [remote_path]
    DriveHandler.downloaded_paths = []
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_file = bin_dir / "file"
    fake_file.write_text("#!/usr/bin/env sh\necho application/octet-stream\n")
    fake_file.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    server = ThreadingHTTPServer(("127.0.0.1", 0), DriveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return subprocess.run(
            [
                "bash",
                str(DRIVE_SCRIPT),
                "--token",
                "drv_live_test",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--allow-nonherenow-base-url",
                "export",
                "drv_test",
                "safe",
                "--to",
                str(tmp_path / "export"),
            ],
            text=True,
            capture_output=True,
            timeout=15,
            env=env,
        )
    finally:
        server.shutdown()
        server.server_close()


def test_drive_export_rejects_traversal_paths(tmp_path: Path) -> None:
    result = _run_export(tmp_path, "safe/../../pwned-by-drive-export.txt")

    assert result.returncode != 0
    assert "refusing unsafe Drive export path" in result.stderr
    assert not (tmp_path / "pwned-by-drive-export.txt").exists()
    assert DriveHandler.downloaded_paths == []


def test_drive_export_writes_safe_paths_under_target(tmp_path: Path) -> None:
    result = _run_export(tmp_path, "safe/nested/ok.txt")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "export" / "nested" / "ok.txt").read_text() == "drive contents\n"
    assert not (tmp_path / "nested" / "ok.txt").exists()
