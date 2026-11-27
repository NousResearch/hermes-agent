"""A platform whose plugin is not installed yet never holds a gateway back, and arrives without a restart.

Live: a real ``hermes gateway run`` on a disposable root. Another installer (the detached default-plugin
pass) is played by a process holding the unattended-installer lock; Home Assistant (a platform that
left core) is the platform owed a plugin, configured by ``HASS_TOKEN``. Outbound HTTPS goes to a closed
port, so an install attempt fails at once and reports why.
"""
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gateway.test_multiplex_hot_serve_live import control, start_daemon, stop, tail
from tests.gateway.test_platform_hot_start import _accepts, _closed_port, _free_port
from tests.waiting import wait_until

_HOLDER = """
import fcntl, sys, time
from pathlib import Path
lock = Path(sys.argv[1])
lock.parent.mkdir(parents=True, exist_ok=True)
fh = lock.open("a+")
fcntl.flock(fh, fcntl.LOCK_EX)
Path(sys.argv[2]).touch()
time.sleep(600)
"""


def _hold_installer_lock(root: Path, tmp_path: Path) -> subprocess.Popen:
    held = tmp_path / "held"
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(root / "cache" / "default-plugins.lock"),
                               str(held)])
    wait_until(held.exists, timeout=30, what="the other installer to hold the lock")
    return holder


def _gateway(tmp_path: Path, dotenv: str, *, proxy: str = ""):
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: false\n", encoding="utf-8")
    (root / ".env").write_text(dotenv, encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    closed = proxy or _closed_port()
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(repo), PYTHONUNBUFFERED="1",
               HERMES_GATEWAY_LOCK_DIR=str(tmp_path / "locks"), HTTPS_PROXY=closed, HTTP_PROXY=closed,
               https_proxy=closed, http_proxy=closed, NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
    return root, {"root": root, "env": env, "log": tmp_path / "gateway.log"}


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_a_slow_plugin_install_never_holds_the_start_back(tmp_path):
    """The start used to finish the install (a clone that hangs, or the other installer's whole pass)
    before it served anything: minutes of nothing. The ready platforms now serve within seconds."""
    port = _free_port()
    with socket.socket() as blackhole:  # accepts the proxy connection and never answers it
        blackhole.bind(("127.0.0.1", 0))
        blackhole.listen(8)
        proxy = f"http://127.0.0.1:{blackhole.getsockname()[1]}"
        root, mux = _gateway(tmp_path, f"WEBHOOK_ENABLED=true\nWEBHOOK_PORT={port}\nHASS_TOKEN=t\n", proxy=proxy)
        _serves_while_the_install_waits(root, mux, port, tmp_path)


def _serves_while_the_install_waits(root: Path, mux: dict, port: int, tmp_path: Path) -> None:
    holder = _hold_installer_lock(root, tmp_path)
    proc = None
    try:
        proc, _desc = start_daemon(mux, timeout=45)
        wait_until(lambda: _accepts(port), timeout=15, interval=0.1, what="webhook listening")
        assert proc.poll() is None, tail(mux)
    finally:
        holder.kill()
        holder.wait(timeout=10)
        if proc is not None:
            stop(proc, expect=None)


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_credentials_set_on_a_running_gateway_for_a_missing_plugin_say_how_it_arrives(tmp_path):
    """``hermes config set`` of a credential whose platform's plugin is missing, gateway running: the CLI
    printed only "✓ Set" while nothing served the platform. It now says what the running gateway did
    about it (installing, the install outcome) or, when it did not install it, the command that does."""
    port = _free_port()
    _root, mux = _gateway(tmp_path, f"WEBHOOK_ENABLED=true\nWEBHOOK_PORT={port}\n")
    proc, _desc = start_daemon(mux)
    try:
        done = subprocess.run([sys.executable, "-m", "hermes_cli.main", "config", "set", "HASS_TOKEN", "t"],
                              env=mux["env"], cwd=Path(__file__).resolve().parents[2], capture_output=True,
                              text=True, timeout=120, check=False)
        assert done.returncode == 0, (done.stdout, done.stderr)
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        assert lines[0].startswith("✓ Set HASS_TOKEN"), done.stdout
        said = "\n".join(lines[1:])
        assert "installing the plugin" in said or "hermes plugins install homeassistant" in said, (done.stdout, tail(mux))
        assert _accepts(port) and proc.poll() is None, tail(mux)  # nothing else went down meanwhile
    finally:
        stop(proc, expect=None)
