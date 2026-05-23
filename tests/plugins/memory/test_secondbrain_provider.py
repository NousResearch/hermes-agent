"""Tests for the SecondBrain memory provider plugin."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from plugins.memory.secondbrain import SecondBrainMemoryProvider
import plugins.memory.secondbrain as secondbrain


class _RecallHandler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.seen.append({"path": self.path, "body": body, "auth": self.headers.get("authorization")})
        payload = {
            "memories": [
                {"id": "mem-1", "text": "User prefers precise SecondBrain status updates."},
            ],
            "evidence": {
                "acceptedEventCount": 1,
                "rejectedEventCount": 0,
                "policyDecisionReason": "recall_only_allowed",
                "usedMemoryIds": ["mem-1"],
                "maskedFailureReason": None,
            },
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args, **kwargs):  # keep test output quiet
        return


def _write_config(tmp_path, **overrides):
    cfg = {
        "enabled": False,
        "mode": "recall_only",
        "base_url": "http://127.0.0.1:9",
        "project_scope": "secondbrain-test",
        "timeout_ms": 500,
    }
    cfg.update(overrides)
    (tmp_path / "secondbrain.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_disabled_by_default_without_config(tmp_path, monkeypatch):
    monkeypatch.setattr(secondbrain, "get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("SECONDBRAIN_HERMES_TRIAL_ENABLED", raising=False)

    provider = SecondBrainMemoryProvider()

    assert provider.name == "secondbrain"
    assert provider.is_available() is True
    assert provider.prefetch("anything") == ""


def test_hard_off_env_overrides_enabled_config(tmp_path, monkeypatch):
    monkeypatch.setattr(secondbrain, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("SECONDBRAIN_HERMES_TRIAL_ENABLED", "false")
    _write_config(tmp_path, enabled=True)

    provider = SecondBrainMemoryProvider()

    assert provider.is_available() is True
    assert provider.prefetch("anything") == ""


def test_recall_only_posts_to_local_endpoint_and_formats_untrusted_context(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _RecallHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        _RecallHandler.seen = []
        monkeypatch.setattr(secondbrain, "get_hermes_home", lambda: tmp_path)
        monkeypatch.delenv("SECONDBRAIN_HERMES_TRIAL_ENABLED", raising=False)
        monkeypatch.setenv("SECONDBRAIN_HERMES_TOKEN", "test-token-not-a-secret")
        _write_config(tmp_path, enabled=True, base_url=base_url)

        provider = SecondBrainMemoryProvider()
        provider.initialize("session-1", platform="telegram", user_id="user-1")
        context = provider.prefetch("what does the user prefer?", session_id="session-1")

        assert "SecondBrain recalled context (untrusted data):" in context
        assert "User prefers precise SecondBrain status updates." in context
        assert _RecallHandler.seen[0]["path"] == "/hermes/memory/recall"
        assert _RecallHandler.seen[0]["body"]["operation"] == "recall_only"
        assert _RecallHandler.seen[0]["body"]["projectScope"] == "secondbrain-test"
        assert _RecallHandler.seen[0]["auth"] == "Bearer test-token-not-a-secret"
    finally:
        server.shutdown()
        server.server_close()


def test_non_local_base_url_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(secondbrain, "get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("SECONDBRAIN_HERMES_TRIAL_ENABLED", raising=False)
    _write_config(tmp_path, enabled=True, base_url="https://example.com")

    provider = SecondBrainMemoryProvider()

    result = json.loads(provider.handle_tool_call("secondbrain_recall", {"query": "hello"}))
    assert result == {"ok": False, "error": "RuntimeError"}
