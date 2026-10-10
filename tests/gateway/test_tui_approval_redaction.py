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
    @staticmethod
    def _sent(monkeypatch):
        """Capture the ``approval`` server request frame ``_emit_approval_request`` sends."""
        from tui_gateway import server as tui_server, server_requests

        sent = {}
        monkeypatch.setattr(server_requests, "send_async",
                            lambda method, sid, params, on_result: (sent.update(method=method, sid=sid, params=params),
                                                                    lambda reason: None)[1])
        monkeypatch.setattr(tui_server, "_sessions", {"sess-1": {"session_key": "key-1"}})
        return tui_server, sent

    def test_emit_approval_request_redacts_command_in_payload(self, monkeypatch):
        tui_server, sent = self._sent(monkeypatch)
        raw = "curl -H 'Authorization: token ghp_01...6789' https://api.github.com"
        tui_server._emit_approval_request("sess-1", {"command": raw, "description": "x"})

        assert sent["method"] == "approval" and sent["sid"] == "sess-1"
        # credential removed, non-command field + command structure preserved
        assert "ghp_01...6789" not in sent["params"]["command"]
        assert sent["params"]["description"] == "x"
        assert "github.com" in sent["params"]["command"]

    @pytest.mark.parametrize(
        ("allow_session", "allow_permanent", "expected"),
        [
            (True, True, ["once", "session", "always", "deny"]),
            (True, False, ["once", "session", "deny"]),
            (False, False, ["once", "deny"]),
        ],
    )
    def test_emit_approval_request_honors_allowed_scopes(
        self, monkeypatch, allow_session, allow_permanent, expected
    ):
        tui_server, sent = self._sent(monkeypatch)
        tui_server._emit_approval_request(
            "sess-1",
            {"allow_permanent": allow_permanent, "allow_session": allow_session, "command": "<write to AGENTS.md>"},
        )

        assert sent["params"]["choices"] == expected
