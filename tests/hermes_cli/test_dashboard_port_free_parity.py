"""``hermes dashboard`` / ``hermes serve`` for a user who never moved to the gateway.

Naming this home's gateway as the holder of the requested port costs a round trip on the gateway's
control socket (bounded at 2 s against a slow gateway), so it is asked only once the port is known
to be held. A free port starts exactly as before and never reaches the control socket, even while
this home's gateway runs (on another port). A port held by anything other than this home's gateway
keeps the old answer word for word, and exit 75. Asking a home with no gateway only connects: its
``gateway.lock`` keeps its mode and bytes. Every CLI test here passes unchanged on the code before the
gateway check existed; the last test calls the new helper directly.
"""
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli.gateway_runtime_discovery import _socket_path
from tests.hermes_cli.live_gateway import (
    AUDITED, REPO_ROOT, audited_launcher, free_port, home_env, live_gateway, unix_connects, write_config,
)

pytestmark = pytest.mark.platforms("linux")

_OLD_HINT = ("  Port {port} on 127.0.0.1 is already in use — likely another 'hermes serve' / 'hermes dashboard' "
             "backend or the Hermes gateway. Stop the other process, or pass --port <other> "
             "(--port 0 picks a free ephemeral port).")


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    with live_gateway(tmp_path_factory.mktemp("gateway-elsewhere")) as live:
        dist = live.user / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><title>old dashboard</title>", encoding="utf-8")
        live.env["HERMES_WEB_DIST"] = str(dist)
        yield live


@pytest.fixture
def plain_env(tmp_path):
    """A home that never ran a gateway, but holds the ``gateway.lock`` an older Hermes left (mode 0644)."""
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    write_config(home)
    lock = home / "gateway.lock"
    lock.write_bytes(b"")
    lock.chmod(0o644)
    return home_env(home, user)


def _snapshot(home):
    """``gateway.*`` files of *home*: mode and bytes (what discovery's missing-owner path repairs)."""
    return {path.name: (path.stat().st_mode, path.read_bytes()) for path in home.glob("gateway*")}


def _control_socket(gateway) -> str:
    return str(_socket_path(gateway.home))


def _accepts(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _boot_and_stop(env: dict, log, *argv: str, port: int) -> str:
    """Start the audited CLI, wait until *port* accepts, stop it; returns its output."""
    proc = subprocess.Popen([sys.executable, "-c", AUDITED, str(log), *argv, "--port", str(port)], cwd=REPO_ROOT,
                            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    lines: list[str] = []
    threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True).start()
    try:
        deadline = time.monotonic() + 90
        while not _accepts(port):
            assert proc.poll() is None and time.monotonic() < deadline, "".join(lines)
            time.sleep(0.05)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    return "".join(lines)


@pytest.mark.parametrize("command", [["serve"], ["dashboard", "--no-open"]], ids=["serve", "dashboard"])
def test_a_free_port_starts_without_asking_the_gateway(gateway, tmp_path, command):
    log = tmp_path / "unix-connects.jsonl"
    _boot_and_stop(gateway.env, log, *command, port=free_port())
    assert _control_socket(gateway) not in unix_connects(log)


def test_status_with_its_port_free_never_asks_the_gateway(gateway, tmp_path):
    """``--status`` through the installed launcher, the port it names free, this home's gateway running."""
    log = tmp_path / "unix-connects.jsonl"
    proc = subprocess.run([audited_launcher(tmp_path, log), "dashboard", "--status", "--port", str(free_port())], cwd=REPO_ROOT,
                          env=gateway.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
                          check=False)
    assert proc.returncode == 0, proc.stdout
    assert not any(line.startswith("The Hermes gateway") for line in proc.stdout.splitlines()), proc.stdout
    assert _control_socket(gateway) not in unix_connects(log)


def _held_port_answer(env: dict) -> tuple[int, str, int]:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "serve", "--port", str(port)],
                                cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=120, check=False)
    return result.returncode, result.stdout, port


def test_a_port_held_by_another_process_keeps_the_old_answer_and_the_homes_files(plain_env):
    """No gateway runs: asking for one only connects, it never judges or narrows ``gateway.lock``."""
    home = Path(plain_env["HERMES_HOME"])
    before = _snapshot(home)
    code, out, port = _held_port_answer(plain_env)
    assert code == 75, out
    assert out.splitlines() == [f"BACKEND_PORT_IN_USE port={port}", _OLD_HINT.format(port=port)], out
    assert _snapshot(home) == before


def test_a_port_held_by_another_process_beside_this_homes_gateway_keeps_the_old_answer(gateway):
    code, out, port = _held_port_answer(gateway.env)
    assert code == 75, out
    assert out.splitlines() == [f"BACKEND_PORT_IN_USE port={port}", _OLD_HINT.format(port=port)], out


def test_asking_a_home_with_no_gateway_leaves_its_lock_alone(plain_env):
    from hermes_cli.web_server_port_holder import gateway_holding_port, running_gateway
    home = Path(plain_env["HERMES_HOME"])
    before = _snapshot(home)
    assert running_gateway(home) is None
    assert gateway_holding_port(9119, home, host="127.0.0.1") is None
    assert _snapshot(home) == before
    assert oct((home / "gateway.lock").stat().st_mode & 0o777) == "0o644"
