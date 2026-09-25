"""Every Hermes process type reads and writes through the remote backend with the same credential
(design §4.4 step 6, G3): a CLI ``config set`` in one process is seen by a gateway-boot process
and a TUI-gateway (dashboard) process, each of which fetched on its own at import time."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .stub_plane import INSTANCE, TOKEN, StubPlane, remote_env

REPO = Path(__file__).resolve().parents[3]

READ_SNIPPET = """
import json, sys
import {module}
from hermes_cli.config import load_config
print("RESULT=" + json.dumps(load_config()["display"]["personality"]))
"""


def _run(args, env, timeout=120):
    return subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)


@pytest.fixture
def plane_env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    with StubPlane() as plane:
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG", "TZ",
                                                           "PYTHONHASHSEED", "HERMES_TEST_ISOLATION"}}
        env.update(remote_env(plane), HERMES_HOME=str(home), PYTHONPATH=str(REPO), HERMES_YOLO_MODE="",
                   PYTHONDONTWRITEBYTECODE="1")
        plane.upper = {"display": {"personality": "concise"}}
        yield plane, env, home


def _result(proc):
    assert proc.returncode == 0, proc.stderr[-3000:]
    # tui_gateway.server moves print() to stderr (stdout is its JSON-RPC channel), so read both.
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.startswith("RESULT=")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(lines[-1][len("RESULT="):])


def test_cli_gateway_and_dashboard_share_the_remote_config(plane_env):
    plane, env, home = plane_env

    cli = _run([sys.executable, "-m", "hermes_cli.main", "config", "set", "display.personality", "pirate"], env)
    assert cli.returncode == 0, cli.stderr[-3000:]
    assert plane.profile("default")["values"] == {"display": {"personality": "pirate"}}

    gateway = _run([sys.executable, "-c", READ_SNIPPET.format(module="gateway.run")], env)
    assert _result(gateway) == "pirate"
    dashboard = _run([sys.executable, "-c", READ_SNIPPET.format(module="tui_gateway.server")], env)
    assert _result(dashboard) == "pirate"

    gets = [r for r in plane.requests if r["method"] == "GET"]
    assert len(gets) >= 3  # each process fetched for itself at boot
    assert {r["auth"] for r in plane.requests} == {f"Bearer {TOKEN}"}
    assert {r["instance"] for r in plane.requests} == {INSTANCE}
    assert not (home / "config.yaml").exists()


def test_process_exits_non_zero_when_plane_is_down(plane_env, monkeypatch):
    plane, env, home = plane_env
    plane.fail_status = 403  # a refusal is not retried, so this is fast
    proc = _run([sys.executable, "-m", "hermes_cli.main", "config", "get", "display.personality"], env)
    assert proc.returncode != 0
    assert "Hermes does not start without its remote config" in proc.stderr
    assert "concise" not in proc.stdout  # no default or cached value was served
