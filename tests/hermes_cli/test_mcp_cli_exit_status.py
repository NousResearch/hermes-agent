"""Process-level exit-status coverage for ``hermes mcp test``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _run_hermes(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _write_mcp_config(home: Path, servers: dict[str, dict[str, object]]) -> None:
    home.mkdir()
    # JSON is valid YAML and avoids coupling this process-level CLI test to a YAML writer.
    (home / "config.yaml").write_text(json.dumps({"mcp_servers": servers}), encoding="utf-8")


def test_mcp_test_missing_server_returns_nonzero_exit_status(tmp_path: Path) -> None:
    result = _run_hermes(tmp_path / "hermes", "mcp", "test", "missing")

    assert result.returncode == 1
    assert "Server 'missing' not found in config." in result.stdout


def test_mcp_test_connection_failure_returns_nonzero_exit_status(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    _write_mcp_config(home, {"broken": {"command": str(tmp_path / "does-not-exist")}})

    result = _run_hermes(home, "mcp", "test", "broken")

    assert result.returncode == 1
    assert "Connection failed" in result.stdout


def test_mcp_test_success_returns_zero_exit_status(tmp_path: Path) -> None:
    server = tmp_path / "mcp_server.py"
    server.write_text(
        "import json\n"
        "import sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    request_id = request.get('id')\n"
        "    if request_id is None:\n"
        "        continue\n"
        "    if request.get('method') == 'initialize':\n"
        "        result = {'protocolVersion': request['params']['protocolVersion'], "
        "'capabilities': {'tools': {}}, 'serverInfo': {'name': 'fixture', 'version': '1'}}\n"
        "    elif request.get('method') == 'tools/list':\n"
        "        result = {'tools': [{'name': 'ping', 'description': 'Ping', "
        "'inputSchema': {'type': 'object', 'properties': {}}}]}\n"
        "    else:\n"
        "        result = {}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': request_id, 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    home = tmp_path / "hermes"
    _write_mcp_config(home, {
        "fixture": {"command": sys.executable, "args": [str(server)], "connect_timeout": 10},
    })

    result = _run_hermes(home, "mcp", "test", "fixture")

    assert result.returncode == 0, result.stderr
    assert "Connected" in result.stdout
    assert "Tools discovered: 1" in result.stdout
