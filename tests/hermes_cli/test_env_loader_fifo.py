"""Mounted secret environments must survive bootstrap and repeated runtime reads."""

import errno
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest


def _serve_once(path, payload, stop):
    """A bounded dummy secret producer; never reads the user's mounted environment."""
    while not stop.is_set():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != errno.ENXIO:
                raise
            stop.wait(0.01)
            continue
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return


@pytest.mark.platforms("posix")
def test_mounted_environment_survives_bootstrap_and_runtime_reads(tmp_path):
    env_path = tmp_path / ".env"
    os.mkfifo(env_path, 0o600)
    (tmp_path / "config.yaml").write_text(
        "model: {provider: custom, default: dummy, base_url: 'http://localhost:9/v1'}\n"
        "platform_toolsets: {cli: []}\n", encoding="utf-8")
    stop = threading.Event()
    writer = threading.Thread(
        target=_serve_once, args=(env_path, b"HERMES_FIFO_TEST_SECRET=dummy-value\n", stop))
    writer.start()
    code = """
import os
import faulthandler
faulthandler.dump_traceback_later(20)
from pathlib import Path
from hermes_cli.env_loader import load_hermes_dotenv
home = Path(os.environ["HERMES_HOME"])
load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
from agent.secret_scope import build_profile_secret_scope
from hermes_cli.config import load_env, reload_env, sanitize_env_file, save_env_value, remove_env_value
for _ in range(3):
    assert load_env()["HERMES_FIFO_TEST_SECRET"] == "dummy-value"
    assert build_profile_secret_scope(home)["HERMES_FIFO_TEST_SECRET"] == "dummy-value"
    reload_env()
assert sanitize_env_file() == 0
save_env_value("HERMES_FIFO_TEST_SECRET", "must-not-persist")
assert remove_env_value("HERMES_FIFO_TEST_SECRET") is False
assert load_env()["HERMES_FIFO_TEST_SECRET"] == "dummy-value"
assert (home / ".env").is_fifo()
import asyncio
from types import SimpleNamespace
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_controls import AuthorityConnection

class Transport:
    def write(self, frame):
        return True

async def create():
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
        _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
        _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(home), instance_id="fixture")
    connection = AuthorityConnection(owner, Transport(), {"user_id": "owner"})
    try:
        for index in range(2):
            reply = await connection.dispatch({"jsonrpc": "2.0", "id": index, "method": "session.create",
                "params": {"source": "cli", "cwd": str(home), "toolsets": [], "request_id": f"fifo-{index}"}})
            assert "error" not in reply, reply.get("error")
            assert reply["result"]["session_id"]
    finally:
        await connection.close()
        owner.db.close()

asyncio.run(create())
print("mounted environment ready")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "HERMES_HOME": str(tmp_path)}, capture_output=True,
            text=True, timeout=30, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Mounted environment child stalled:\n{exc.stderr!r}")
    finally:
        stop.set()
        writer.join(2)
    assert result.returncode == 0, result.stderr
    assert "mounted environment ready" in result.stdout
    assert env_path.is_fifo()
    assert not list(tmp_path.glob(".env_*.tmp"))
