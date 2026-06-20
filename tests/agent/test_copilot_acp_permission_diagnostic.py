"""The ACP permission-denial path must surface a one-time diagnostic.

Hermes denies the ACP agent's native tool-permission requests by design (it
drives tools via injected markers, not the agent's own toolset). But if the
agent is launched in a permission mode that routes EVERY tool through that path
(Claude Agent SDK "dontAsk"/"default", Copilot CLI without bypass), the silent
denial leaves a do-nothing agent with no diagnostic. These tests lock the
contract: the denial still happens (security/design unchanged) AND a single
actionable warning is emitted on the first denial, not one-per-call spam.
"""

import logging

from agent.copilot_acp_client import CopilotACPClient, _permission_denied


def _make_client() -> CopilotACPClient:
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        command="/bin/true",
        args=["--acp", "--stdio"],
    )


def test_permission_denied_payload_unchanged():
    """The deny response shape (cancelled outcome) must not regress."""
    resp = _permission_denied("req-1")
    assert resp["id"] == "req-1"
    assert resp["result"]["outcome"]["outcome"] == "cancelled"


def test_first_denial_warns_with_actionable_cause(caplog):
    client = _make_client()
    msg = {"jsonrpc": "2.0", "id": 1, "method": "session/request_permission", "params": {}}

    sent: list[str] = []

    class _FakeStdin:
        def write(self, data):
            sent.append(data)

        def flush(self):
            pass

    class _FakeProc:
        stdin = _FakeStdin()

    with caplog.at_level(logging.WARNING):
        handled = client._handle_server_message(
            msg, process=_FakeProc(), cwd="/tmp", text_parts=None, reasoning_parts=None
        )

    assert handled is True
    # denial still sent
    assert any("cancelled" in s for s in sent)
    # exactly one actionable warning, naming the fix
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "bypass" in warnings[0].message.lower()


def test_repeated_denials_warn_only_once(caplog):
    client = _make_client()
    msg = {"jsonrpc": "2.0", "id": 1, "method": "session/request_permission", "params": {}}

    class _FakeStdin:
        def write(self, data):
            pass

        def flush(self):
            pass

    class _FakeProc:
        stdin = _FakeStdin()

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            client._handle_server_message(
                msg, process=_FakeProc(), cwd="/tmp", text_parts=None, reasoning_parts=None
            )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "denial warning must fire once, not per-call"
    assert client._permission_denied_count == 5
