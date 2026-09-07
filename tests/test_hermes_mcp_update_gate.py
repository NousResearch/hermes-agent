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


def test_proven_owner_never_expires(tmp_path):
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\n")
    assert gate.live_update_marker_owner(marker, now=10000, pid_alive=lambda pid: True,
                                         pid_create_time=lambda pid: 99.9) is not None
    assert marker.exists()


def test_unprovable_owner_expires_at_the_age_ceiling(tmp_path):
    """Upstream main unlinks a marker past 20 min. The PR kept unknown owners
    blocking forever, so one recycled PID disabled the bridge permanently."""
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\n")
    inside = 100 + gate.UPDATE_MARKER_MAX_AGE_SECONDS - 60
    beyond = 100 + gate.UPDATE_MARKER_MAX_AGE_SECONDS + 60
    unknown = dict(pid_alive=lambda pid: True, pid_create_time=lambda pid: None)
    assert gate.live_update_marker_owner(marker, now=inside, **unknown) is not None
    assert gate.live_update_marker_owner(marker, now=beyond, **unknown) is None
    assert marker.exists(), "readers never mutate; the lock owner reclaims"


def test_posix_creation_time_is_provable_without_third_party_packages(tmp_path):
    """B7: _pid_create_time returned None unconditionally off Windows, so every
    live PID stayed `unknown` on macOS/Linux and no marker ever aged out."""
    created = gate.pid_create_time(os.getpid())
    if os.name != "nt" and not sys.platform.startswith(("linux", "darwin")):
        pytest.skip(f"no stdlib creation-time source for {sys.platform}")
    assert created is not None, f"{sys.platform} must prove its own creation time"
    assert 0 < created <= time.time() + 1
    assert gate.pid_identity_status(os.getpid(), int(time.time()) + 5) == "matching"


@pytest.mark.parametrize("body", ["", "oops", "123\n100\nextra\n"])
def test_marker_that_names_nobody_does_not_gate_the_bridge(tmp_path, body):
    """A body naming no owner proves no update is running. Gating on it is how
    one torn write made the bridge exit 0 at import until a human intervened."""
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text(body)
    with pytest.raises(gate.UpdateMarkerUnhealthyError) as excinfo:
        gate.live_update_marker_owner(marker)
    assert str(marker) in str(excinfo.value), "the message must name the file to delete"
    assert not gate.should_quiesce_mcp_bridge(argv=["python", "-m", gate.MCP_MAIN_MODULE], marker=marker)
    assert marker.read_text() == body, "readers never mutate the marker"


def test_unreadable_marker_still_fails_closed_for_the_exact_mcp_argv(tmp_path):
    """Opposite of the case above: a marker we could not READ might be hiding a
    real update, so that one alone keeps gating -- and only this entry point."""
    marker = tmp_path / gate.MARKER_NAME
    marker.mkdir()  # a directory raises OSError, not "names nobody"
    assert gate.should_quiesce_mcp_bridge(argv=["python", "-m", gate.MCP_MAIN_MODULE], marker=marker)
    assert not gate.should_quiesce_mcp_bridge(
        argv=["python", "script.py", "-m", gate.MCP_MAIN_MODULE], marker=marker)


def test_unexpected_internal_failure_does_not_gate(tmp_path):
    """`except Exception: return True` turned any bug in here into a silent,
    permanent bridge outage. Only a proven-unreadable marker may gate."""
    marker = tmp_path / gate.MARKER_NAME
    marker.write_text("123\n100\n")

    def explode(pid):
        raise RuntimeError("probe is broken")

    assert not gate.should_quiesce_mcp_bridge(
        argv=["python", "-m", gate.MCP_MAIN_MODULE], marker=marker,
        pid_alive=lambda pid: True, pid_create_time=explode, now=10 ** 9)


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


def test_bridge_watcher_survives_a_transient_probe_failure(tmp_path):
    """The 0.5s watcher used to have no guard: one raise killed the thread and
    the bridge then held the venv for the rest of the session, silently."""
    env = dict(os.environ, HERMES_HOME=str(tmp_path))
    script = """
import os, sys, threading, time
from pathlib import Path
import agent.transports.hermes_tools_mcp_server as bridge
sys.orig_argv = [sys.executable, '-m', 'agent.transports.hermes_tools_mcp_server']
calls = []
real = bridge._update_quiesce_requested
def flaky():
    calls.append(1)
    if len(calls) < 3:
        raise OSError('marker is being rewritten')
    return real()
bridge._update_quiesce_requested = flaky
def claim():
    (Path(os.environ['HERMES_HOME']) / '.hermes-update-in-progress').write_text(
        str(os.getpid()) + chr(10) + str(int(time.time())) + chr(10))
threading.Timer(1.6, claim).start()
print('watching', flush=True)
bridge._watch_for_update_quiesce()
raise RuntimeError('watcher unexpectedly returned')
"""
    child = subprocess.run([sys.executable, "-c", script],
                           cwd=Path(__file__).resolve().parents[1], env=env,
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "watching"
