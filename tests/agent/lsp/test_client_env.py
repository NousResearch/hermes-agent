from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient
from agent.lsp.protocol import LSPProtocolError


async def _capture_spawn_env(monkeypatch: pytest.MonkeyPatch, client: LSPClient) -> dict[str, str]:
    captured: dict[str, str] = {}

    async def fake_create_subprocess_exec(*_args, **kwargs):
        captured.update(kwargs["env"])
        raise FileNotFoundError("stop after capturing env")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(LSPProtocolError):
        await client.start()

    return captured


@pytest.mark.asyncio
async def test_lsp_subprocess_env_strips_parent_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-parent-anthropic")
    monkeypatch.setenv("GH_TOKEN", "gh-parent-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-parent-token")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "dashboard-parent-token")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    client = LSPClient(
        server_id="mock",
        workspace_root=str(tmp_path),
        command=["missing-lsp"],
    )

    env = await _capture_spawn_env(monkeypatch, client)

    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "GH_TOKEN" not in env
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert "HERMES_DASHBOARD_SESSION_TOKEN" not in env
    assert env.get("PATH") == os.environ.get("PATH", "")


@pytest.mark.asyncio
async def test_lsp_explicit_env_overrides_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-openai")

    client = LSPClient(
        server_id="mock",
        workspace_root=str(tmp_path),
        command=["missing-lsp"],
        env={
            "OPENAI_API_KEY": "explicit-lsp-key",
            "CUSTOM_LSP_SETTING": "enabled",
        },
    )

    env = await _capture_spawn_env(monkeypatch, client)

    assert env["OPENAI_API_KEY"] == "explicit-lsp-key"
    assert env["CUSTOM_LSP_SETTING"] == "enabled"
