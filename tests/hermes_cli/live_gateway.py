"""A real gateway (``python -m gateway.run``) in a temp home with the Webapp's listener on a fixed port,
plus the CLI as a child process of that home: the old ``hermes dashboard`` / ``hermes serve`` entry points
run against what a migrated host really has. No model is reached (the model points at a closed port and
no turn is submitted)."""
from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.gateway.fixtures.local_recovery_probe import child_env
from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import wait_until

REPO_ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@dataclass
class LiveGateway:
    home: Path
    user: Path
    port: int
    process: subprocess.Popen
    env: dict

    def cli(self, *argv: str, stdin: str | None = None, extra_env: dict | None = None,
            timeout: float = 120) -> subprocess.CompletedProcess:
        """``hermes <argv>`` in this home, stdout and stderr merged."""
        return run_cli({**self.env, **(extra_env or {})}, *argv, stdin=stdin, timeout=timeout)


def run_cli(env: dict, *argv: str, stdin: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    """``hermes <argv>`` with *env*, stdout and stderr merged."""
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", *argv], cwd=REPO_ROOT, env=env, input=stdin,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)


# ``python -c AUDITED LOG ARGV...`` runs ``hermes ARGV...`` and appends every AF_UNIX path the CLI
# connects to (the gateway's control socket among them) to LOG, one JSON string per line.
AUDITED = """
import json, runpy, socket, sys
log = open(sys.argv[1], "a", buffering=1, encoding="utf-8")
def hook(event, args):
    if event == "socket.connect" and args[0].family == socket.AF_UNIX:
        log.write(json.dumps(str(args[1])) + "\\n")
sys.addaudithook(hook)
sys.argv = ["hermes", *sys.argv[2:]]
runpy.run_module("hermes_cli.main", run_name="__main__", alter_sys=True)
"""


def audited_launcher(directory: Path, log: Path) -> str:
    """A ``hermes`` launcher (kept running while the CLI runs as its child) for the audited CLI."""
    path = directory / "hermes"
    path.write_text(f'#!/bin/sh\n"{sys.executable}" -c \'{AUDITED}\' "{log}" "$@"\n', encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def unix_connects(log: Path) -> list[str]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


def home_env(home: Path, user: Path) -> dict:
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(REPO_ROOT),
               PYTHONUNBUFFERED="1", HERMES_GATEWAY_LOCK_DIR=str(user / "locks"),
               # The suite's no-fetch rule: a fresh home's first boot would otherwise start the detached
               # default-plugin installer, which fetches the live catalog.
               HERMES_DISABLE_LAZY_INSTALLS="1")
    return env


def write_config(home: Path, extra: dict | None = None) -> None:
    config = {"gateway": {"multiplex_profiles": False},
              "model": {"provider": "custom", "default": "no-inference", "base_url": "http://127.0.0.1:1/v1"},
              "auxiliary": {"title_generation": {"enabled": False}}, **(extra or {})}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


@contextlib.contextmanager
def live_gateway(tmp_path: Path, *, webapp: bool = True, webapp_host: str | None = None,
                 config: dict | None = None, prepare=None):
    """Boot the gateway; yields once its control socket says ``ready``. Stops it by its own pid.
    *webapp_host* sets ``webapp.host`` (the default is 127.0.0.1); *config* adds top-level keys.
    *prepare* ``(home, env)`` runs before the gateway starts (to set up sign-in, for example)."""
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    port = free_port()
    block = {"enabled": True, "port": port, **({"host": webapp_host} if webapp_host else {})}
    write_config(home, {**({"webapp": block} if webapp else {}), **(config or {})})
    env = home_env(home, user)
    if prepare is not None:
        prepare(home, env)
    log_path = tmp_path / "gateway.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=REPO_ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            def ready():
                if process.poll() is not None:
                    raise AssertionError(f"gateway exited {process.returncode}: {log_path.read_text()}")
                with contextlib.suppress(OSError, ValueError, AssertionError):
                    return control(home, "identify").get("state") == "ready"
                return False
            wait_until(ready, timeout=60, what=f"gateway ready ({log_path})")
            yield LiveGateway(home=home, user=user, port=port, process=process, env=env)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
