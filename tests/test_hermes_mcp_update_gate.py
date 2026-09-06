"""Marker identity and early MCP startup use only stdlib against a scratch home."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import hermes_mcp_update_gate as gate


@pytest.mark.parametrize("alive,created,status", [
    (False, None, "stale"), (True, 99.9, "matching"),
    (True, 100.9, "matching"), (True, 101, "stale"),
    (True, None, "unknown"),
])
def test_pid_identity_contract(alive, created, status):
    assert gate.pid_identity_status(123, 100, pid_alive=lambda pid: alive,
                                    pid_create_time=lambda pid: created) == status


@pytest.mark.parametrize("created", [99.9, None])
def test_live_or_unknown_owner_never_expires(tmp_path, created):
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\n")
    assert gate.live_update_marker_owner(marker, now=10000, pid_alive=lambda pid: True,
                                         pid_create_time=lambda pid: created) is not None
    assert marker.exists()


@pytest.mark.parametrize("body", ["", "oops", "123\n100\nextra\n"])
def test_malformed_marker_blocks_exact_mcp_only(tmp_path, body):
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text(body)
    assert gate.should_quiesce_mcp_bridge(argv=["python", "-m", gate.MCP_MAIN_MODULE], marker=marker)
    assert not gate.should_quiesce_mcp_bridge(argv=["python", "script.py", "-m", gate.MCP_MAIN_MODULE], marker=marker)
    assert marker.read_text() == body


def test_dead_owner_reader_does_not_delete_marker(tmp_path):
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\n")
    assert gate.live_update_marker_owner(marker, now=101, pid_alive=lambda pid: False) is None
    assert marker.exists()


def test_tagged_bridge_keeps_gate_closed_during_claim_gap(tmp_path):
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\nhandoff-bridge\n")
    assert gate.live_update_marker_owner(marker, now=110, pid_alive=lambda pid: False) is not None
    assert gate.live_update_marker_owner(marker, now=131, pid_alive=lambda pid: False) is None


def test_real_module_launch_exits_before_native_preload(tmp_path):
    (tmp_path / gate.MARKER_NAME).write_text(f"{os.getpid()}\n{int(time.time())}\n")
    env = dict(os.environ, HERMES_HOME=str(tmp_path))
    child = subprocess.run([sys.executable, "-v", "-m", gate.MCP_MAIN_MODULE],
                           cwd=Path(__file__).resolve().parents[1], env=env,
                           capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr
    assert "paused while" in child.stderr
    assert "import 'agent.jiter_preload'" not in child.stderr
    assert "import 'psutil'" not in child.stderr


def test_unrelated_agent_import_is_not_gated_by_broken_marker(tmp_path):
    (tmp_path / gate.MARKER_NAME).write_text("")
    env = dict(os.environ, HERMES_HOME=str(tmp_path))
    child = subprocess.run([sys.executable, "-c", "import agent; print('ready')"],
                           cwd=Path(__file__).resolve().parents[1], env=env,
                           capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "ready"


def test_running_bridge_watcher_exits_when_update_marker_appears(tmp_path):
    env = dict(os.environ, HERMES_HOME=str(tmp_path))
    script = """
import os, sys, threading, time
from pathlib import Path
from agent.transports.hermes_tools_mcp_server import _watch_for_update_quiesce
sys.orig_argv = [sys.executable, '-m', 'agent.transports.hermes_tools_mcp_server']
def claim():
    (Path(os.environ['HERMES_HOME']) / '.hermes-update-in-progress').write_text(
        str(os.getpid()) + chr(10) + str(int(time.time())) + chr(10))
threading.Timer(0.1, claim).start()
print('watching', flush=True)
_watch_for_update_quiesce()
raise RuntimeError('watcher unexpectedly returned')
"""
    child = subprocess.run([sys.executable, "-c", script],
                           cwd=Path(__file__).resolve().parents[1], env=env,
                           capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "watching"
