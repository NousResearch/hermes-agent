"""Regression coverage for Desktop's ephemeral dashboard session token.

Desktop launches ``hermes serve`` with a freshly generated
HERMES_DASHBOARD_SESSION_TOKEN. A user's ~/.hermes/.env may also contain a
stable dashboard token for remote-dashboard workflows. The normal env loader
lets .env override shell exports, but Desktop's per-process token must win or
Electron's local API calls 401.
"""

from __future__ import annotations

import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _request_status(port: int, token: str | None) -> int:
    headers = {}
    if token is not None:
        headers["X-Hermes-Session-Token"] = token
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/sessions?limit=1",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as exc:
        exc.read()
        return exc.code


def test_desktop_session_token_survives_dotenv_override(tmp_path: Path):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        f"HERMES_DASHBOARD_SESSION_TOKEN={secrets.token_urlsafe(32)}\n",
        encoding="utf-8",
    )

    desktop_token = secrets.token_urlsafe(32)
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
            "HERMES_DESKTOP": "1",
            "HERMES_DASHBOARD_SESSION_TOKEN": desktop_token,
        }
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--isolated",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        port = None
        output: list[str] = []
        deadline = time.time() + 25
        assert proc.stdout is not None
        while time.time() < deadline and port is None:
            line = proc.stdout.readline()
            if line:
                output.append(line.rstrip())
                match = re.search(r"HERMES_BACKEND_READY port=(\d+)", line)
                if match:
                    port = int(match.group(1))
            elif proc.poll() is not None:
                break

        assert port is not None, "backend did not become ready: " + " | ".join(output[-8:])
        assert _request_status(port, None) == 401
        assert _request_status(port, desktop_token) == 200
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
