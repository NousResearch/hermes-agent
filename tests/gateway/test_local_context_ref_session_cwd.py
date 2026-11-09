"""``@`` references on a local (CLI/TUI/Desktop) session resolve in that session's cwd.

The inbound prepare step runs before the turn binds the session policy, so the process
``TERMINAL_CWD`` there is the gateway daemon's own folder (``~`` or ``MESSAGING_CWD``), never the
folder the client was launched from.
"""
from types import SimpleNamespace

import pytest

from gateway import session_policy
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


async def _context_length(_source, _session_key):
    return 128000


async def _expand(source, text):
    runner = object.__new__(GatewayRunner)
    runner._inbound_model_context_length = _context_length
    return await runner._expand_inbound_context_references(source, "key", text)


@pytest.fixture(autouse=True)
def workspaces(tmp_path, monkeypatch):
    project, daemon = tmp_path / "project", tmp_path / "daemon-home"
    for folder, body in ((project, "PROJECT-NOTES"), (daemon, "DECOY-NOTES")):
        folder.mkdir()
        (folder / "notes.txt").write_text(body, encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(daemon))
    policies = {"local-chat": SimpleNamespace(cwd=str(project))}
    monkeypatch.setattr(session_policy, "policy_for_source",
                        lambda runner, source: policies.get(source.chat_id) if source.platform == Platform.LOCAL else None)


@pytest.mark.asyncio
async def test_local_session_reference_reads_the_session_workspace():
    expanded = await _expand(SessionSource(platform=Platform.LOCAL, chat_id="local-chat"), "read @file:notes.txt")
    assert expanded is not None and "PROJECT-NOTES" in expanded
    assert "DECOY-NOTES" not in expanded


@pytest.mark.asyncio
async def test_messaging_reference_keeps_the_terminal_cwd():
    expanded = await _expand(SessionSource(platform=Platform.TELEGRAM, chat_id="123"), "read @file:notes.txt")
    assert expanded is not None and "DECOY-NOTES" in expanded
