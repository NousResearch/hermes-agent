"""Regression test for TUI approval-prompt credential redaction (#48456).

Follow-up to #50767, which redacted the chat-platform and SSE/API approval
transports. The TUI JSON-RPC transport is the third egress: three
`register_gateway_notify` callbacks in `tui_gateway/server.py` emit the raw
`approval_data` (with an unredacted `command`) to the TUI client. They now
route through the module-level `_emit_approval_request` helper, which redacts
`payload["command"]` via the shared `gateway.run._redact_approval_command` seam
before emitting.
"""

import inspect

import pytest


class TestTuiApprovalEmitRedaction:
    def test_emit_approval_request_redacts_command_in_payload(self, monkeypatch):
        from tui_gateway import server as tui_server

        emitted = {}
        monkeypatch.setattr(
            tui_server, "_emit",
            lambda event, sid, payload=None: emitted.update(
                {"event": event, "sid": sid, "payload": payload}
            ),
        )
        raw = "curl -H 'Authorization: token ghp_01...6789' https://api.github.com"
        tui_server._emit_approval_request("sess-1", {"command": raw, "description": "x"})

        assert emitted["event"] == "approval.request"
        # credential removed, non-command field + command structure preserved
        assert "ghp_01...6789" not in emitted["payload"]["command"]
        assert emitted["payload"]["description"] == "x"
        assert "github.com" in emitted["payload"]["command"]

    def test_emit_approval_request_handles_missing_command(self, monkeypatch):
        from tui_gateway import server as tui_server

        emitted = {}
        monkeypatch.setattr(
            tui_server, "_emit",
            lambda event, sid, payload=None: emitted.update({"payload": payload}),
        )
        tui_server._emit_approval_request("s", {"description": "no command here"})
        assert emitted["payload"] == {"description": "no command here"}
        tui_server._emit_approval_request("s", None)
        assert emitted["payload"] == {}

    def test_no_raw_command_emit_in_approval_registrations(self):
        """Every register_gateway_notify approval callback must route through the
        redacting `_emit_approval_request` helper — no registration may emit the
        raw payload via `_emit("approval.request", ...)` directly. The ONLY
        allowed raw emit is inside the helper itself."""
        from tui_gateway import server as tui_server

        src = inspect.getsource(tui_server)
        raw_emits = src.count('_emit("approval.request"')
        assert raw_emits == 1, (
            f'expected exactly 1 raw _emit("approval.request") (inside the '
            f"redacting helper), found {raw_emits} — a registration may be "
            f"emitting the unredacted command"
        )
        assert "_emit_approval_request(sid, data)" in src, (
            "registration lambdas must route through _emit_approval_request"
        )
