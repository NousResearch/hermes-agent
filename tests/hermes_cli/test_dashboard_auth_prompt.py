"""The interactive sign-in setup ``hermes serve --host 0.0.0.0`` offers on a real terminal when no sign-in
provider is configured: what it prints and what it stores in ``dashboard.basic_auth``.

These pin the prompt as it is before ``hermes dashboard password`` shares its storage: the same transcript,
a signing secret written whenever ``config.yaml`` has none (also when ``HERMES_DASHBOARD_BASIC_AUTH_SECRET``
is set in the environment), the plugin re-enable line before the closing lines, and a backend that then
signs in with the typed password. The child runs in its own session with a pseudo-terminal as stdin/stdout/stderr and no controlling
terminal, so ``getpass`` reads the pseudo-terminal too.
"""
import json
import os
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import hermes_yaml as yaml

from tests.gateway.fixtures.local_recovery_probe import child_env

pytestmark = pytest.mark.platforms("linux")

REPO_ROOT = Path(__file__).resolve().parents[2]
USER, PASSWORD = "prompt-admin", "throwaway-Prompt-Pw-41c2"  # invented for this test


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _home(tmp_path, extra: dict | None = None):
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    config = {"model": {"provider": "custom", "default": "no-inference", "base_url": "http://127.0.0.1:1/v1"},
              "auxiliary": {"title_generation": {"enabled": False}}, **(extra or {})}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    env = child_env()
    env.update(HOME=str(user), HERMES_HOME=str(home), PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1",
               HERMES_DISABLE_LAZY_INSTALLS="1", TERM="dumb", PROMPT_TOOLKIT_NO_CPR="1")
    return home, env


class _Terminal:
    """``hermes serve`` on a pseudo-terminal: read what it prints, type answers."""

    def __init__(self, env: dict, port: int):
        self.master, slave = os.openpty()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "serve", "--host", "0.0.0.0", "--port", str(port)],
            cwd=REPO_ROOT, env=env, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        self.output = ""

    def wait_for(self, text: str, timeout: float = 120) -> None:
        deadline = time.monotonic() + timeout
        while text not in self.output:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"never printed {text!r}:\n{self.output}"
            ready, _, _ = select.select([self.master], [], [], remaining)
            if ready:
                try:
                    chunk = os.read(self.master, 65536)
                except OSError:
                    chunk = b""
                assert chunk, f"exited before printing {text!r}:\n{self.output}"
                self.output += chunk.decode("utf-8", "replace").replace("\r\n", "\n")

    def answer(self, prompt: str, reply: str) -> None:
        self.wait_for(prompt)
        os.write(self.master, reply.encode() + b"\r")

    def close(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        os.close(self.master)


def _set_up_through_the_prompt(env: dict, port: int) -> _Terminal:
    terminal = _Terminal(env, port)
    terminal.answer("Choice [1]: ", "1")
    terminal.answer("Username [admin]: ", USER)
    terminal.answer("Password: ", PASSWORD)
    terminal.answer("Confirm password: ", PASSWORD)
    terminal.wait_for("Sign in at the dashboard with these credentials.")
    return terminal


def _basic_auth(home) -> dict:
    return (yaml.safe_load((home / "config.yaml").read_text()).get("dashboard") or {}).get("basic_auth") or {}


def _closing_lines(output: str) -> list[str]:
    lines = [line.rstrip() for line in output.split("\n")]
    start = next(i for i, line in enumerate(lines) if "Confirm password:" in line) + 1
    end = next(i for i, line in enumerate(lines) if "Sign in at the dashboard with these credentials." in line)
    return [line for line in lines[start:end + 1] if line]


def test_the_prompt_stores_a_hash_and_a_signing_secret_and_the_backend_signs_in_with_it(tmp_path):
    home, env = _home(tmp_path)
    port = _free_port()
    terminal = _set_up_through_the_prompt(env, port)
    try:
        assert _closing_lines(terminal.output) == [
            f"  ✓ Username/password auth configured (user: {USER}).",
            "    Saved to config.yaml under dashboard.basic_auth.",
            "    Sign in at the dashboard with these credentials."]
        section = _basic_auth(home)
        assert sorted(section) == ["password_hash", "secret", "username"]  # the emptied plaintext is not written
        assert section["username"] == USER
        assert section["password_hash"].startswith("scrypt$") and section["secret"]
        assert PASSWORD not in (home / "config.yaml").read_text() and PASSWORD not in terminal.output
        terminal.wait_for("HERMES_BACKEND_READY")
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20) as client:
            login = client.post("/auth/password-login",
                                json={"provider": "basic", "username": USER, "password": PASSWORD, "next": "/"})
            assert login.status_code == 200
            assert client.get("/api/auth/me").json().get("user_id") == USER
    finally:
        terminal.close()


def test_with_the_secret_in_the_environment_and_basic_disabled_the_prompt_still_writes_a_secret(tmp_path):
    home, env = _home(tmp_path, {"plugins": {"disabled": ["basic"]}})
    env["HERMES_DASHBOARD_BASIC_AUTH_SECRET"] = "an-invented-environment-signing-secret-0123456789"
    terminal = _set_up_through_the_prompt(env, _free_port())
    try:
        assert _closing_lines(terminal.output) == [
            "  ✓ Re-enabled the bundled 'basic' auth plugin (was in plugins.disabled)",
            f"  ✓ Username/password auth configured (user: {USER}).",
            "    Saved to config.yaml under dashboard.basic_auth.",
            "    Sign in at the dashboard with these credentials."]
        assert _basic_auth(home)["secret"]
        assert "basic" not in ((yaml.safe_load((home / "config.yaml").read_text()).get("plugins") or {})
                               .get("disabled") or [])
    finally:
        terminal.close()
