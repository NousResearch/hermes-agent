"""Offline prompt-size must inspect the configured runtime's context budget."""
import socket

import pytest

from agent.system_prompt import build_system_prompt_parts
from hermes_cli.prompt_size import _build_inspection_agent


@pytest.mark.parametrize("provider,port,window", [("custom", 8080, 65536), ("lmstudio", 1234, 32768)])
def test_offline_inspection_honors_custom_model_context_for_project_files(tmp_path, monkeypatch, provider, port, window):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  default: forge-synthetic-model\n  provider: {provider}\n"
        f"  base_url: http://127.0.0.1:{port}/v1\n  context_length: {window}\n",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text("x" * 35000, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    attempts = []

    def blocked_connect(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError("prompt-size attempted network I/O")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    agent = _build_inspection_agent("cli")
    assert agent.context_compressor.context_length == window
    assert len(build_system_prompt_parts(agent)["context"]) < 30000
    assert attempts == []


def test_custom_endpoint_without_context_pin_is_not_probed(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: forge-synthetic-model\n  provider: custom\n"
        "  base_url: http://127.0.0.1:8080/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    attempts = []

    def blocked_connect(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError("prompt-size attempted network I/O")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    assert _build_inspection_agent("cli").context_compressor.context_length > 0
    assert attempts == []
