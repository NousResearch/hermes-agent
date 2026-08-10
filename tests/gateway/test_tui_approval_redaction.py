"""Regression test for TUI approval-prompt credential redaction (#48456).

Follow-up to #50767, which redacted the chat-platform and SSE/API approval
transports. The TUI JSON-RPC transport is the third egress: three
`register_gateway_notify` callbacks in `tui_gateway/server.py` emit the raw
`approval_data` (with an unredacted `command`) to the TUI client. They route
through the module-level `_emit_approval_request` helper, which redacts
`payload["command"]` via `agent.redact.redact_sensitive_text(force=True)`
before emitting.

Importing `gateway.run` from this path is deliberately avoided: a long-lived
dashboard/TUI process can already have a stale `agent.turn_context` in
`sys.modules`, and `gateway.run`'s import chain then raises ImportError.
Approval notify treats that as hard-block ("Failed to send approval request").
"""

import inspect

import pytest


class TestTuiApprovalEmitRedaction:
    def test_emit_approval_request_redacts_command_in_payload(self, monkeypatch):
        from tui_gateway import server as tui_server

        emitted = {}
        monkeypatch.setattr(
            tui_server,
            "_emit",
            lambda event, sid, payload=None: emitted.update(
                {"event": event, "sid": sid, "payload": payload}
            ),
        )
        raw = (
            "curl -H 'Authorization: Bearer "
            "ghp_0123456789abcdefghijklmnopqrstuvwx' https://api.github.com"
        )
        tui_server._emit_approval_request(
            "sess-1", {"command": raw, "description": "x"}
        )

        assert emitted["event"] == "approval.request"
        cmd = emitted["payload"]["command"]
        assert "ghp_0123456789abcdefghijklmnopqrstuvwx" not in cmd
        assert emitted["payload"]["description"] == "x"
        assert "github.com" in cmd

    def test_emit_approval_request_source_avoids_gateway_run_import(self):
        """TUI emit path must not import gateway.run (stale-process ImportError)."""
        from tui_gateway import server as tui_server

        source = inspect.getsource(tui_server._emit_approval_request)
        assert "from gateway.run import" not in source
        assert "agent.redact" in source
        assert "redact_sensitive_text" in source
