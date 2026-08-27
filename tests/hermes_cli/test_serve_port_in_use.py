"""#93608 — port bind conflict must be machine-readable, not a bare exit 1.

The reporter's repro: something already listens on the serve port (another
``hermes serve``, the gateway, anything) → ``hermes serve`` printed only
uvicorn's ``ERROR: [Errno 98/10048] error while attempting to bind on
address`` and exited 1 — indistinguishable from a broken backend for the
desktop spawn and for scripts.

Contract under test:

* conflict → single stdout sentinel ``BACKEND_PORT_IN_USE port=<port>`` +
  a human hint line + exit code 75 (EX_TEMPFAIL, the repo's existing
  transient-condition convention) — and NO ``HERMES_BACKEND_READY``.
* free explicit port → boots and announces ``HERMES_BACKEND_READY`` exactly
  as before (contract untouched).
* ``--port 0`` (ephemeral) → probe skipped, boots, announces the
  OS-assigned port.
"""

import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX serve-runner path under test"
)


# ---------------------------------------------------------------------------
# Unit: the probe itself
# ---------------------------------------------------------------------------


def test_probe_detects_held_socket():
    from hermes_cli.web_server import _port_bind_conflict

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        assert _port_bind_conflict("127.0.0.1", port) is True
    finally:
        holder.close()


def test_probe_free_port_is_clean():
    from hermes_cli.web_server import _port_bind_conflict

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    assert _port_bind_conflict("127.0.0.1", port) is False


def test_probe_skips_ephemeral_port_zero():
    from hermes_cli.web_server import _port_bind_conflict

    # port 0 can never conflict — must short-circuit False, never bind.
    assert _port_bind_conflict("127.0.0.1", 0) is False


def test_addr_in_use_error_classification():
    import errno

    from hermes_cli.web_server import _is_addr_in_use_error

    assert _is_addr_in_use_error(OSError(errno.EADDRINUSE, "in use")) is True
    assert _is_addr_in_use_error(OSError(98, "linux")) is True
    assert _is_addr_in_use_error(OSError(10048, "winsock")) is True
    assert _is_addr_in_use_error(OSError(errno.EACCES, "denied")) is False


def test_exit_code_is_distinct_tempfail():
    from hermes_cli.web_server import PORT_IN_USE_EXIT_CODE

    assert PORT_IN_USE_EXIT_CODE == 75  # EX_TEMPFAIL — repo convention
    assert PORT_IN_USE_EXIT_CODE != 1


# ---------------------------------------------------------------------------
# E2E: real start_server in a subprocess (temp HERMES_HOME)
# ---------------------------------------------------------------------------


def _spawn_serve(port: int, tmp_path: Path, merge_stderr: bool = True) -> subprocess.Popen:
    home = tmp_path / "hermes_home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        HERMES_HOME=str(home),
        HERMES_SERVE_HEADLESS="1",
        PYTHONUNBUFFERED="1",
    )
    # Ensure a stray desktop-parent env can't arm the watchdog/reaper paths.
    for k in ("HERMES_DESKTOP", "HERMES_PARENT_PID", "HERMES_PARENT_START_MARKER",
              "HERMES_PARENT_NONCE"):
        env.pop(k, None)
    code = (
        "from hermes_cli.web_server import start_server\n"
        f"start_server(host='127.0.0.1', port={port}, open_browser=False, "
        "headless=True)\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        # merge_stderr=False: the caller only asserts on stdout. Discard
        # stderr instead of piping it — with the serve-path stdout redirect
        # active ALL server logging lands on stderr, and an unread stderr
        # pipe can fill (~64KB) and block the child before it ever emits
        # the READY sentinel, turning the test into a timeout flake.
        stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL,
        text=True,
    )


def _read_until(proc: subprocess.Popen, token: str, timeout: float = 120.0):
    """Collect stdout lines until ``token`` appears, process exits, or timeout."""
    lines: list[str] = []
    hit = threading.Event()

    def _pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if token in line:
                hit.set()
                return

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    hit.wait(timeout)
    return hit.is_set(), lines


def test_conflict_emits_sentinel_and_exit_75(tmp_path):
    """Reporter's exact repro shape: a real held LISTEN socket on the port."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        proc = _spawn_serve(port, tmp_path)
        try:
            out, _ = proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            out = ""
            pytest.fail("serve did not exit on a port conflict")
    finally:
        holder.close()

    assert proc.returncode == 75, f"exit={proc.returncode}\n{out}"
    assert f"BACKEND_PORT_IN_USE port={port}" in out
    assert f"Port {port}" in out  # human hint line
    assert "HERMES_BACKEND_READY" not in out  # never claimed ready
    # exactly one machine sentinel line
    assert out.count("BACKEND_PORT_IN_USE") == 1


def test_free_port_boots_and_announces_ready(tmp_path):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    proc = _spawn_serve(port, tmp_path)
    try:
        ready, lines = _read_until(proc, "HERMES_BACKEND_READY")
        out = "".join(lines)
        assert ready, f"no READY sentinel; output:\n{out}"
        assert f"HERMES_BACKEND_READY port={port}" in out
        assert "BACKEND_PORT_IN_USE" not in out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_ephemeral_port_zero_unaffected(tmp_path):
    proc = _spawn_serve(0, tmp_path)
    try:
        ready, lines = _read_until(proc, "HERMES_BACKEND_READY")
        out = "".join(lines)
        assert ready, f"no READY sentinel; output:\n{out}"
        # OS-assigned non-zero port announced
        ready_line = next(l for l in lines if "HERMES_BACKEND_READY" in l)
        announced = int(ready_line.strip().rsplit("port=", 1)[1])
        assert announced > 0
        assert "BACKEND_PORT_IN_USE" not in out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_ready_sentinel_arrives_on_stdout_not_stderr(tmp_path):
    """#96282 — the Desktop spawn watches child.stdout for the READY sentinel.

    ``tui_gateway.server`` (imported on the serve startup path since 6d4e851d8
    for the flush-on-SIGTERM handlers) redirects ``sys.stdout`` to
    ``sys.stderr`` at import time. If the sentinel is printed through the
    redirected ``sys.stdout``, it lands on stderr and the desktop times out
    after 90s against a perfectly healthy backend. The sentinel must reach
    the real stdout (fd 1).

    The other E2E tests here merge stderr into stdout, which is exactly how
    the regression slipped past CI — this test keeps the streams separate.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    proc = _spawn_serve(port, tmp_path, merge_stderr=False)
    try:
        # stdout only: the sentinel MUST arrive here (desktop watches this pipe)
        ready, lines = _read_until(proc, "HERMES_BACKEND_READY")
        out = "".join(lines)
        assert ready, (
            f"READY sentinel not on stdout (desktop boot would time out); "
            f"stdout:\n{out}"
        )
        assert f"HERMES_BACKEND_READY port={port}" in out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
