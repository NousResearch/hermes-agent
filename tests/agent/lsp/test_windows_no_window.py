from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.lsp import client as client_module
from agent.lsp.client import LSPClient
from hermes_cli import _subprocess_compat


_CREATE_NO_WINDOW = 0x08000000


class _EOFStream:
    async def readline(self) -> bytes:
        return b""

    async def readuntil(self, _separator: bytes) -> bytes:
        raise asyncio.IncompleteReadError(partial=b"", expected=None)


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.stdout = _EOFStream()
        self.stderr = _EOFStream()


@pytest.mark.asyncio
async def test_lsp_spawn_hides_background_console_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_argv: tuple[str, ...] = ()
    captured_kwargs: dict[str, object] = {}

    async def fake_create_subprocess_exec(*argv: str, **kwargs: object) -> _FakeProcess:
        nonlocal captured_argv, captured_kwargs
        captured_argv = argv
        captured_kwargs = kwargs
        return _FakeProcess()

    monkeypatch.setattr(client_module.sys, "platform", "win32")
    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(
        client_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    client = LSPClient(
        server_id="typescript",
        workspace_root=str(tmp_path),
        command=["typescript-language-server.cmd", "--stdio"],
    )

    await client._spawn()
    await asyncio.sleep(0)

    assert captured_argv[:3] == ("cmd.exe", "/c", "typescript-language-server.cmd")
    assert captured_kwargs.get("creationflags") == _CREATE_NO_WINDOW
    assert "start_new_session" not in captured_kwargs


@pytest.mark.asyncio
async def test_lsp_spawn_keeps_separate_session_on_posix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_kwargs: dict[str, object] = {}

    async def fake_create_subprocess_exec(*_argv: str, **kwargs: object) -> _FakeProcess:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return _FakeProcess()

    monkeypatch.setattr(client_module.sys, "platform", "linux")
    monkeypatch.setattr(
        client_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    client = LSPClient(
        server_id="typescript",
        workspace_root=str(tmp_path),
        command=["typescript-language-server", "--stdio"],
    )

    await client._spawn()
    await asyncio.sleep(0)

    assert captured_kwargs.get("start_new_session") is True
    assert "creationflags" not in captured_kwargs
