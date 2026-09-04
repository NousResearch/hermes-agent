"""Credential discovery must not consume a mounted secret-provider FIFO."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("host", [
    pytest.param("linux", marks=pytest.mark.linux_only),
    pytest.param("macos", marks=pytest.mark.macos_only),
])
def test_fifo_discovery_returns_inherited_credentials_without_a_writer(tmp_path, host):
    fifo = tmp_path / ".env"
    os.mkfifo(fifo, 0o600)
    before = fifo.stat()
    env = dict(os.environ, HERMES_HOME=str(tmp_path), OPENAI_API_KEY="inherited-test-key")
    code = """
import signal
signal.alarm(5)
from hermes_cli.config import get_env_value_prefer_dotenv, load_env
assert load_env() == {}
assert get_env_value_prefer_dotenv('OPENAI_API_KEY') == 'inherited-test-key'
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], env=env,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("credential discovery blocked opening the writerless FIFO")
    assert result.returncode == 0, "credential discovery did not finish before the FIFO watchdog: " + result.stderr
    assert fifo.stat().st_ino == before.st_ino
    assert fifo.stat().st_mode == before.st_mode


def test_regular_dotenv_still_overrides_inherited_credentials(tmp_path, monkeypatch):
    from hermes_cli.config import get_env_value_prefer_dotenv, invalidate_env_cache, load_env

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-test-key")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=stored-test-key\n", encoding="utf-8")
    invalidate_env_cache()
    assert load_env()["OPENAI_API_KEY"] == "stored-test-key"
    assert get_env_value_prefer_dotenv("OPENAI_API_KEY") == "stored-test-key"
