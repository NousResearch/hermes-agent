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

    def test_emit_approval_request_surfaces_hidden_characters_in_folder_and_site_fields(self, monkeypatch):
        """The card shows ``path``, ``host``, ``label`` and ``risk`` as it shows ``command`` and
        ``description``: hidden characters in them become visible before the request is sent."""
        tui_server, sent = self._sent(monkeypatch)
        tui_server._emit_approval_request("sess-1", {
            "kind": "network", "host": "evil.example\u202emoc.buhtig", "label": "Web\u202e search",
            "path": "C:\\docs\\a\u202etxt.exe", "risk": "ok\u202e", "description": "d\u202e"})

        assert "\\u202e" in sent["params"]["description"]
        for field in ("path", "host", "label", "risk"):
            assert "\u202e" not in sent["params"][field], field
            assert "\\u202e" in sent["params"][field], field

    def test_emit_approval_request_leaves_ordinary_unicode_in_folder_and_site_fields(self, monkeypatch):
        """Accented letters, CJK and an emoji in a folder or site field reach the card as they came."""
        tui_server, sent = self._sent(monkeypatch)
        plain = {"path": "C:\\Users\\zoë\\Fotos días 📁\\日本語", "host": "bücher.example",
                 "label": "Recherche web 🔎 (bücher.example)", "risk": "L’agent peut lire ce dossier — 読み取りのみ。"}
        tui_server._emit_approval_request("sess-1", {"kind": "access", **plain})

        assert {field: sent["params"][field] for field in plain} == plain

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
