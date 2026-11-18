"""``hermes dashboard --status`` on a host whose gateway serves the Webapp.

The status scan matched its own ``hermes dashboard --status`` process (it parses as a dashboard on the
port it names) and the reachability probe answered for the gateway's listener on that port, so the command
listed itself as a running backend and never mentioned the gateway. It now leaves itself out and adds one
line naming the gateway that serves the Webapp. A host running a real old backend and no gateway gets the
same listing as before (control), except that the launcher is no longer listed beside the backend when
``--status`` names the backend's port.

The installed ``hermes`` runs the CLI under a launcher process whose argv is the same ``hermes dashboard
--status``; the tests launch through such a launcher (a script named ``hermes`` that runs the CLI as its
child, not ``exec``), which is the process the old scan listed. The scan is machine-wide (not scoped to a
home), so assertions name PIDs rather than counting lines.
"""
import contextlib
import os
import secrets
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli.gateway_runtime_discovery import _socket_path
from tests.hermes_cli.live_gateway import (
    REPO_ROOT, audited_launcher, free_port, home_env, live_gateway, unix_connects, write_config,
)

pytestmark = pytest.mark.platforms("linux")


def _launcher(directory) -> str:
    """A ``hermes`` launcher that keeps running while the CLI runs as its child."""
    path = directory / "hermes"
    path.write_text(f'#!/bin/sh\n"{sys.executable}" -m hermes_cli.main "$@"\n', encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _status(env: dict, *argv: str) -> tuple[int, int, str]:
    """``hermes dashboard --status`` in *env* through the launcher: (launcher pid, exit code, output)."""
    proc = subprocess.Popen([_launcher(Path(env["HOME"])), "dashboard", "--status", *argv],
                            cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = proc.communicate(timeout=120)
    return proc.pid, proc.returncode, out


def test_status_beside_the_gateway_leaves_itself_out_and_names_the_gateway(tmp_path):
    with live_gateway(tmp_path) as gateway:
        pid, code, out = _status(gateway.env, "--port", str(gateway.port))
    assert code == 0, out
    assert f"PID {pid} " not in out, out
    assert "--status" not in out, out
    assert "The Hermes gateway" in out and "serves the Webapp for this profile at" in out, out
    assert f":{gateway.port}" in out, out


def _wait_ready(proc: subprocess.Popen, lines: list[str]) -> bool:
    hit = threading.Event()

    def pump():
        for line in proc.stdout:
            lines.append(line)
            if "HERMES_BACKEND_READY" in line:
                hit.set()

    threading.Thread(target=pump, daemon=True).start()
    return hit.wait(120)


@contextlib.contextmanager
def _start_backend(env: dict, port: int, *host: str):
    """A real ``hermes serve`` on *port* (``--host`` *host* when given), yielded once it is ready."""
    argv = [*(("--host", *host) if host else ()), "--port", str(port)]
    backend = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "serve", *argv],
                               cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines: list[str] = []
    try:
        assert _wait_ready(backend, lines), "".join(lines)
        yield backend.pid
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=30)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait(timeout=5)


@pytest.fixture(scope="module")
def old_backend(tmp_path_factory):
    """A real ``hermes serve`` in a home that never ran a gateway: (env, backend pid, its port)."""
    root = tmp_path_factory.mktemp("old-backend")
    home, user = root / "state", root / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    write_config(home)
    env = home_env(home, user)
    port = free_port()
    with _start_backend(env, port) as pid:
        yield env, pid, port


def _lists(out: str, pid: int, port: int) -> bool:
    return any(line.startswith(f"    PID {pid} [serve]: ") and f"--port {port}" in line for line in out.splitlines())


def test_a_real_old_backend_is_listed_as_before_and_no_gateway_is_mentioned(old_backend):
    """Control: an old backend runs in a home with no gateway; the port --status names is free."""
    env, backend_pid, port = old_backend
    pid, code, out = _status(env, "--port", str(free_port()))
    assert code == 0, out
    assert "hermes dashboard/serve process(es) running:" in out, out
    assert _lists(out, backend_pid, port), out
    assert f"PID {pid} " not in out, out
    assert not any(line.startswith("The Hermes gateway") for line in out.splitlines()), out


def test_status_naming_the_old_backends_port_lists_the_backend_and_not_itself(old_backend):
    """No gateway anywhere: ``--status --port <the backend's>`` used to list its own launcher as a second
    backend (the probe on that port answered for the real one)."""
    env, backend_pid, port = old_backend
    pid, code, out = _status(env, "--port", str(port))
    assert code == 0, out
    assert _lists(out, backend_pid, port), out
    assert f"PID {pid} " not in out, out
    assert not any(line.startswith("The Hermes gateway") for line in out.splitlines()), out


def test_status_naming_a_listed_backends_port_never_asks_this_homes_gateway(tmp_path):
    """This home's gateway runs on another port while an old backend serves port Q: the held port
    ``--status --port Q`` names is the listed backend's, so the gateway is never asked about it."""
    with live_gateway(tmp_path) as gateway:
        port = free_port()
        with _start_backend(gateway.env, port) as backend_pid:
            log = tmp_path / "unix-connects.jsonl"
            proc = subprocess.run([audited_launcher(tmp_path, log), "dashboard", "--status", "--port", str(port)],
                                  cwd=REPO_ROOT, env=gateway.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=120, check=False)
            control = str(_socket_path(gateway.home))
    out = proc.stdout
    assert proc.returncode == 0, out
    assert _lists(out, backend_pid, port), out
    assert not any(line.startswith("The Hermes gateway") for line in out.splitlines()), out
    assert control not in unix_connects(log)


def _lists_at(out: str, pid: int, port: int) -> bool:
    return any(line.startswith(f"    PID {pid} [serve]: ") and f"--port {port}" in line for line in out.splitlines())


def test_a_backend_on_another_address_of_the_port_does_not_hide_the_gateway_the_probe_reached(tmp_path):
    """The gateway serves the Webapp on 127.0.0.1:P and an old backend runs on 127.0.0.2:P. ``--status
    --host 0.0.0.0 --port P`` probes 127.0.0.1:P, which is the gateway, not the backend: the backend is
    listed and the gateway is named."""
    with live_gateway(tmp_path) as gateway:
        # 127.0.0.2 is not 127.0.0.1, so the bind needs a sign-in: a throwaway deployment token.
        backend_env = {**gateway.env, "HERMES_DASHBOARD_SESSION_TOKEN": secrets.token_urlsafe(24)}
        with _start_backend(backend_env, gateway.port, "127.0.0.2") as backend_pid:
            pid, code, out = _status(gateway.env, "--host", "0.0.0.0", "--port", str(gateway.port))
    assert code == 0, out
    assert _lists_at(out, backend_pid, gateway.port), out
    assert f"PID {pid} " not in out, out
    assert any(line.startswith("The Hermes gateway") and f":{gateway.port}" in line for line in out.splitlines()), out


def test_status_names_the_gateway_at_the_address_its_probe_connected_to(tmp_path):
    """``--host ip6-localhost`` resolves to ::1, where the gateway's Webapp listens. The gateway is
    asked about the peer the probe connected to (::1), not the spelling of ``--host``."""
    import socket as _socket
    try:
        if not any(info[4][0] == "::1" for info in _socket.getaddrinfo("ip6-localhost", 1, type=_socket.SOCK_STREAM)):
            pytest.skip("ip6-localhost does not resolve to ::1 here")
    except OSError:
        pytest.skip("no ip6-localhost on this host")
    with live_gateway(tmp_path, webapp_host="::1") as gateway:
        pid, code, out = _status(gateway.env, "--host", "ip6-localhost", "--port", str(gateway.port))
    assert code == 0, out
    assert f"PID {pid} " not in out, out
    assert any(line.startswith("The Hermes gateway") for line in out.splitlines()), out


def test_a_process_with_no_listener_is_still_listed_but_does_not_hide_the_gateway(tmp_path):
    """A live process whose command line reads ``hermes serve --port P`` but holds no listener (stuck, or
    still starting) is listed as before; the gateway that really answered on P is named too."""
    stand_in = tmp_path / "bin"
    stand_in.mkdir()
    script = stand_in / "hermes"
    script.write_text("#!/bin/sh\nsleep 120 &\nwait\n", encoding="utf-8")
    script.chmod(0o755)
    with live_gateway(tmp_path) as gateway:
        stuck = subprocess.Popen([str(script), "serve", "--port", str(gateway.port)], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            pid, code, out = _status(gateway.env, "--port", str(gateway.port))
        finally:
            os.killpg(stuck.pid, signal.SIGTERM)
            stuck.wait(timeout=10)
    assert code == 0, out
    assert f"PID {pid} " not in out, out
    assert any(line.startswith(f"    PID {stuck.pid} [serve]: ") for line in out.splitlines()), out
    assert any(line.startswith("The Hermes gateway") for line in out.splitlines()), out
