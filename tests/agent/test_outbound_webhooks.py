"""Tests for the outbound webhook dispatcher (agent.outbound_webhooks).

Covers config parsing, matcher behaviour, HMAC signing, payload shape,
idempotent registration on the plugin manager, and end-to-end delivery
against a real local HTTP server (no mocks on the network path).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent import outbound_webhooks


# ── helpers ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_registration_state():
    _strip_outbound_callbacks()
    outbound_webhooks.reset_for_tests()
    yield
    _strip_outbound_callbacks()
    outbound_webhooks.reset_for_tests()


def _strip_outbound_callbacks():
    """Remove outbound-webhook callbacks from the shared plugin manager.

    ``reset_for_tests`` clears the idempotence set but the manager singleton
    keeps previously-registered callbacks; without this, a target registered
    in one test would fire (real network!) in every later test in this file.
    """
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    for event, callbacks in list(manager._hooks.items()):
        manager._hooks[event] = [
            cb for cb in callbacks
            if not getattr(cb, "__name__", "").startswith("outbound_webhook[")
        ]


class _CapturingHandler(BaseHTTPRequestHandler):
    """Records every POST (path, headers, body) on the server instance."""

    def do_POST(self):  # noqa: N802 — http.server naming
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.captured.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": body,
            }
        )
        status = getattr(self.server, "respond_status", 200)
        self.send_response(status)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002 — http.server naming
        pass


@pytest.fixture()
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    server.captured = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _url(server: HTTPServer, path: str = "/hook") -> str:
    return f"http://127.0.0.1:{server.server_address[1]}{path}"


def _cfg(*entries):
    return {"hooks": {"outbound": list(entries)}}


# ── config parsing ────────────────────────────────────────────────────────


class TestParseConfig:
    def test_missing_block_is_empty(self):
        assert outbound_webhooks.iter_configured_targets({}) == []
        assert outbound_webhooks.iter_configured_targets(None) == []
        assert outbound_webhooks.iter_configured_targets({"hooks": {}}) == []
        assert outbound_webhooks.iter_configured_targets(
            {"hooks": "not-a-dict"}
        ) == []

    def test_non_list_outbound_is_empty(self):
        assert outbound_webhooks.iter_configured_targets(
            {"hooks": {"outbound": {"url": "https://x"}}}
        ) == []

    def test_valid_entry_parses(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com/hook",
                    "events": ["on_session_end"],
                    "name": "ci",
                    "timeout": 5,
                }
            )
        )
        assert len(targets) == 1
        t = targets[0]
        assert t.url == "https://example.com/hook"
        assert t.events == ["on_session_end"]
        assert t.name == "ci"
        assert t.timeout == 5

    def test_missing_url_skipped(self):
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"events": ["on_session_end"]})
        ) == []

    def test_non_http_url_skipped(self):
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"url": "ftp://example.com", "events": ["on_session_end"]})
        ) == []
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"url": "file:///etc/passwd", "events": ["on_session_end"]})
        ) == []

    def test_unknown_events_filtered_known_kept(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com/hook",
                    "events": ["on_session_end", "not_a_real_event"],
                }
            )
        )
        assert len(targets) == 1
        assert targets[0].events == ["on_session_end"]

    def test_all_unknown_events_skips_entry(self):
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"url": "https://example.com", "events": ["bogus"]})
        ) == []

    def test_empty_events_skips_entry(self):
        assert outbound_webhooks.iter_configured_targets(
            _cfg({"url": "https://example.com", "events": []})
        ) == []

    def test_timeout_clamped(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "timeout": 9999,
                }
            )
        )
        assert targets[0].timeout == outbound_webhooks.MAX_TIMEOUT_SECONDS

    def test_bad_timeout_falls_back_to_default(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "timeout": "soon",
                }
            )
        )
        assert targets[0].timeout == outbound_webhooks.DEFAULT_TIMEOUT_SECONDS

    def test_matcher_dropped_for_non_tool_events(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "matcher": "terminal",
                }
            )
        )
        assert targets[0].matcher is None

    def test_matcher_kept_for_tool_events(self):
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["post_tool_call"],
                    "matcher": "terminal|delegate_task",
                }
            )
        )
        assert targets[0].matcher == "terminal|delegate_task"

    def test_secret_env_wins_over_literal(self, monkeypatch):
        monkeypatch.setenv("MY_HOOK_SECRET", "from-env")
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "secret_env": "MY_HOOK_SECRET",
                    "secret": "literal",
                }
            )
        )
        assert targets[0].secret == "from-env"

    def test_unset_secret_env_means_unsigned(self, monkeypatch):
        monkeypatch.delenv("MISSING_SECRET_VAR", raising=False)
        targets = outbound_webhooks.iter_configured_targets(
            _cfg(
                {
                    "url": "https://example.com",
                    "events": ["on_session_end"],
                    "secret_env": "MISSING_SECRET_VAR",
                }
            )
        )
        assert targets[0].secret is None


# ── matcher behaviour ─────────────────────────────────────────────────────


class TestMatcher:
    def _target(self, matcher):
        return outbound_webhooks.WebhookTarget(
            url="https://example.com",
            events=["post_tool_call"],
            matcher=matcher,
        )

    def test_no_matcher_matches_everything(self):
        assert self._target(None).matches_tool("terminal")
        assert self._target(None).matches_tool(None)

    def test_regex_fullmatch(self):
        t = self._target("terminal|delegate_task")
        assert t.matches_tool("terminal")
        assert t.matches_tool("delegate_task")
        assert not t.matches_tool("terminal_extra")
        assert not t.matches_tool(None)

    def test_invalid_regex_falls_back_to_equality(self):
        t = self._target("terminal(")
        assert t.matches_tool("terminal(")
        assert not t.matches_tool("terminal")


# ── payload shape ─────────────────────────────────────────────────────────


class TestPayload:
    def test_top_level_shape_matches_shell_hooks_wire(self):
        body = outbound_webhooks._serialize_payload(
            "post_tool_call",
            {
                "tool_name": "terminal",
                "args": {"command": "ls"},
                "session_id": "sess_1",
                "status": "ok",
                "duration_ms": 42,
            },
        )
        payload = json.loads(body)
        assert payload["hook_event_name"] == "post_tool_call"
        assert payload["tool_name"] == "terminal"
        assert payload["tool_input"] == {"command": "ls"}
        assert payload["session_id"] == "sess_1"
        assert payload["extra"]["status"] == "ok"
        assert payload["extra"]["duration_ms"] == 42
        assert payload["delivery_id"]
        assert payload["timestamp"].endswith("Z")

    def test_unserialisable_values_stringified(self):
        body = outbound_webhooks._serialize_payload(
            "on_session_end", {"weird": object()}
        )
        payload = json.loads(body)
        assert isinstance(payload["extra"]["weird"], str)


# ── registration ──────────────────────────────────────────────────────────


class TestRegistration:
    def test_registration_is_idempotent(self):
        cfg = _cfg(
            {"url": "https://example.com/hook", "events": ["on_session_end"]}
        )
        first = outbound_webhooks.register_from_config(cfg)
        second = outbound_webhooks.register_from_config(cfg)
        assert len(first) == 1
        assert second == []

    def test_safe_mode_skips_registration(self, monkeypatch):
        monkeypatch.setenv("HERMES_SAFE_MODE", "1")
        cfg = _cfg(
            {"url": "https://example.com/hook", "events": ["on_session_end"]}
        )
        assert outbound_webhooks.register_from_config(cfg) == []

    def test_callbacks_never_return_directives(self, http_server):
        """Outbound webhooks are notify-only — invoke_hook must see None."""
        cfg = _cfg({"url": _url(http_server), "events": ["pre_tool_call"]})
        outbound_webhooks.register_from_config(cfg)

        from hermes_cli.plugins import get_plugin_manager

        results = get_plugin_manager().invoke_hook(
            "pre_tool_call", tool_name="terminal", args={"command": "ls"},
        )
        assert outbound_webhooks.flush()
        # No block/context directives from the webhook callback.
        assert results == []
        assert len(http_server.captured) == 1


# ── E2E delivery against a real HTTP server ──────────────────────────────


class TestDelivery:
    def test_delivery_with_hmac_signature(self, http_server):
        secret = "s3cret"
        cfg = _cfg(
            {
                "url": _url(http_server),
                "events": ["on_session_end"],
                "secret": secret,
                "name": "e2e",
            }
        )
        registered = outbound_webhooks.register_from_config(cfg)
        assert len(registered) == 1

        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().invoke_hook(
            "on_session_end",
            session_id="sess_e2e",
            completed=True,
            interrupted=False,
            model="test-model",
            platform="cli",
        )
        assert outbound_webhooks.flush()

        assert len(http_server.captured) == 1
        req = http_server.captured[0]

        payload = json.loads(req["body"])
        assert payload["hook_event_name"] == "on_session_end"
        assert payload["session_id"] == "sess_e2e"
        assert payload["extra"]["completed"] is True
        assert payload["extra"]["model"] == "test-model"

        assert req["headers"]["X-Hermes-Event"] == "on_session_end"
        assert req["headers"]["X-Hermes-Delivery"]
        expected = hmac.new(
            secret.encode(), req["body"], hashlib.sha256
        ).hexdigest()
        assert req["headers"]["X-Hermes-Signature-256"] == f"sha256={expected}"

    def test_unsigned_delivery_has_no_signature_header(self, http_server):
        cfg = _cfg({"url": _url(http_server), "events": ["on_session_end"]})
        outbound_webhooks.register_from_config(cfg)

        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().invoke_hook("on_session_end", session_id="s")
        assert outbound_webhooks.flush()

        assert len(http_server.captured) == 1
        assert "X-Hermes-Signature-256" not in http_server.captured[0]["headers"]

    def test_matcher_filters_tool_events(self, http_server):
        cfg = _cfg(
            {
                "url": _url(http_server),
                "events": ["post_tool_call"],
                "matcher": "terminal",
            }
        )
        outbound_webhooks.register_from_config(cfg)

        from hermes_cli.plugins import get_plugin_manager

        manager = get_plugin_manager()
        manager.invoke_hook(
            "post_tool_call", tool_name="web_search", args={}, status="ok",
        )
        manager.invoke_hook(
            "post_tool_call", tool_name="terminal", args={}, status="ok",
        )
        assert outbound_webhooks.flush()

        assert len(http_server.captured) == 1
        payload = json.loads(http_server.captured[0]["body"])
        assert payload["tool_name"] == "terminal"

    def test_4xx_not_retried(self, http_server):
        http_server.respond_status = 400
        target = outbound_webhooks.WebhookTarget(
            url=_url(http_server), events=["on_session_end"],
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}",
        )
        outbound_webhooks._deliver(delivery)
        assert len(http_server.captured) == 1

    def test_connection_error_does_not_raise(self):
        target = outbound_webhooks.WebhookTarget(
            # Port 9 (discard) — nothing listening.
            url="http://127.0.0.1:9/unreachable",
            events=["on_session_end"],
            timeout=1,
        )
        delivery = outbound_webhooks._build_delivery(
            "on_session_end", target, b"{}",
        )
        # Must swallow the failure (logged), never raise into the agent loop.
        outbound_webhooks._deliver(delivery)
