"""A bare llama.cpp alias resolves to the profile's managed local server, and only to it.

``resolve_provider_client("llamacpp")`` (the fallback chain's chokepoint) and the cached auxiliary
client used to fall through the generic custom branch: with no explicit base_url they returned
whichever cloud provider held a key, posting the local GGUF slug there (#119227), or nothing at all
("Fallback to llamacpp failed: provider not configured"). The managed endpoint here is real: a
``server.json`` naming a live child process, read through the same ownership-checked resolver the
main ladder uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import psutil
import pytest

from agent import auxiliary_client as aux
from hermes_cli.local_runtime import detect
from hermes_cli.local_runtime.supervisor import state_path

MODEL = "SmolLM2-135M-Instruct-Q4_K_M"


@pytest.fixture
def managed_server(monkeypatch):
    """Write server.json for a real supervised child; yields write(port, key)."""
    monkeypatch.setattr(detect, "DEFAULT_PROBE_PORTS", ())  # no stray :8080 server
    monkeypatch.setenv("GEMINI_API_KEY", "cloud-key-must-not-be-used")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                             stdin=subprocess.DEVNULL)
    proc = psutil.Process(child.pid)

    def write(port: int, key: str) -> None:
        state_path().parent.mkdir(parents=True, exist_ok=True)
        state_path().write_text(json.dumps({
            "base_url": f"http://127.0.0.1:{port}/v1", "api_key": key, "pid": child.pid,
            "create_time": proc.create_time(), "executable": proc.exe(),
            "owner_pid": os.getpid(), "owner_create_time": psutil.Process().create_time(),
        }))

    aux._client_cache.clear()
    yield write
    child.kill()
    child.wait()
    state_path().unlink(missing_ok=True)
    aux._client_cache.clear()


def test_fallback_resolution_uses_live_managed_server_or_nothing(managed_server):
    managed_server(18001, "key-a")
    client, model = aux.resolve_provider_client("llamacpp", model=MODEL, raw_codex=True)
    assert str(client.base_url).startswith("http://127.0.0.1:18001/v1"), client.base_url
    assert client.api_key == "key-a" and model == MODEL

    state_path().unlink()  # server off: not configured, never the credentialed cloud provider
    assert aux.resolve_provider_client("llamacpp", model=MODEL, raw_codex=True) == (None, None)


def test_cached_client_follows_managed_endpoint_restart(managed_server):
    managed_server(18001, "key-a")
    first, _ = aux._get_cached_client("llama.cpp", MODEL, task="compression")
    assert str(first.base_url).startswith("http://127.0.0.1:18001/v1"), first.base_url
    assert aux._get_cached_client("llama.cpp", MODEL, task="compression")[0] is first

    managed_server(18002, "key-b")  # restart on a new port with a fresh per-install key
    second, _ = aux._get_cached_client("llama.cpp", MODEL, task="compression")
    assert str(second.base_url).startswith("http://127.0.0.1:18002/v1") and second.api_key == "key-b"

    state_path().unlink()
    assert aux._get_cached_client("llama.cpp", MODEL, task="compression") == (None, None)
