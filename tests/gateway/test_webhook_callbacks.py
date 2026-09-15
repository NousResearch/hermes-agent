"""Callback SSRF-guard and envelope tests (Task 13, #4386/#73828)."""

from __future__ import annotations

import asyncio
import json

from gateway.platforms import webhook_callbacks as wc


def _envelope() -> dict:
    return wc.build_callback_envelope(
        execution_id="e1",
        event_id="ev1",
        status="completed",
        output="done",
        error=None,
        attempt=1,
    )


class TestValidateCallbackUrl:
    def test_public_https_ok(self, monkeypatch):
        monkeypatch.setattr(
            wc.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (wc.socket.AF_INET, wc.socket.SOCK_STREAM, wc.socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
            ],
        )
        ok, _ = wc.validate_callback_url("https://example.com/done")
        assert ok is True

    def test_private_host_blocked(self):
        ok, reason = wc.validate_callback_url("http://10.0.0.5/cb")
        assert ok is False
        assert "SSRF" in reason

    def test_loopback_blocked(self):
        ok, reason = wc.validate_callback_url("http://127.0.0.1/cb")
        assert ok is False
        assert "SSRF" in reason

    def test_metadata_blocked(self):
        ok, _ = wc.validate_callback_url("http://169.254.169.254/latest/meta-data")
        assert ok is False

    def test_non_http_blocked(self):
        ok, reason = wc.validate_callback_url("file:///etc/passwd")
        assert ok is False
        assert "http(s)" in reason

    def test_missing_host_blocked(self):
        ok, _ = wc.validate_callback_url("https://")
        assert ok is False

    def test_any_unsafe_dns_answer_fails_closed(self, monkeypatch):
        monkeypatch.setattr(
            wc.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (wc.socket.AF_INET, wc.socket.SOCK_STREAM, wc.socket.IPPROTO_TCP, "", ("93.184.216.34", 443)),
                (wc.socket.AF_INET, wc.socket.SOCK_STREAM, wc.socket.IPPROTO_TCP, "", ("127.0.0.1", 443)),
            ],
        )
        ok, reason = wc.validate_callback_url("https://example.com/done")
        assert ok is False
        assert "SSRF" in reason


class TestEnvelope:
    def test_envelope_shape(self):
        env = _envelope()
        assert env["execution_id"] == "e1"
        assert env["status"] == "completed"
        assert env["attempt"] == 1
        assert env["error"] is None

    def test_envelope_serializes(self):
        env = wc.build_callback_envelope(
            execution_id="e1", event_id="ev1", status="failed",
            output=None, error="boom", attempt=2,
        )
        json.dumps(env)  # must not raise

    def test_malformed_envelope_rejected_before_dns(self, monkeypatch):
        def should_not_resolve(*_args, **_kwargs):
            raise AssertionError("DNS must not run for an invalid envelope")

        monkeypatch.setattr(wc.socket, "getaddrinfo", should_not_resolve)
        assert wc.deliver_callback("https://example.com/done", None, {}) is False


class TestSigning:
    def test_signature_is_sha256(self):
        body = b'{"x":1}'
        sig = wc._sign(body, "secret")
        assert sig.startswith("sha256=")


class TestPinnedDelivery:
    def test_delivery_dials_the_validated_address_without_second_dns(self, monkeypatch):
        calls = []

        def resolve(*_args, **_kwargs):
            calls.append("resolve")
            return [
                (wc.socket.AF_INET, wc.socket.SOCK_STREAM, wc.socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
            ]

        opened = []

        def open_pinned(destination, body, headers, *, timeout):
            opened.append((destination, body, headers, timeout))
            return 204

        monkeypatch.setattr(wc.socket, "getaddrinfo", resolve)
        monkeypatch.setattr(wc, "_open_pinned", open_pinned)

        assert wc.deliver_callback("https://example.com/done?x=1", "secret", _envelope()) is True
        assert calls == ["resolve"]
        destination = opened[0][0]
        assert destination.connect_host == "93.184.216.34"
        assert destination.hostname == "example.com"
        assert destination.request_target == "/done?x=1"
        assert opened[0][2]["X-Hermes-Signature-256"].startswith("sha256=")

    def test_redirect_is_terminal_and_not_retried(self, monkeypatch):
        monkeypatch.setattr(
            wc.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (wc.socket.AF_INET, wc.socket.SOCK_STREAM, wc.socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
            ],
        )
        attempts = []

        def redirected(*_args, **_kwargs):
            attempts.append(1)
            return 302

        monkeypatch.setattr(wc, "_open_pinned", redirected)
        assert wc.deliver_callback("https://example.com/done", None, _envelope()) is False
        assert attempts == [1]

    def test_async_wrapper_moves_sync_transport_to_thread(self, monkeypatch):
        calls = []

        def fake_deliver(url, secret, envelope, *, timeout):
            calls.append((url, secret, envelope, timeout))
            return True

        monkeypatch.setattr(wc, "deliver_callback", fake_deliver)
        result = asyncio.run(
            wc.deliver_callback_async(
                "https://example.com/done",
                "secret",
                _envelope(),
                timeout=3,
            )
        )
        assert result is True
        assert calls[0][3] == 3


class TestDeliverRefusesPrivate:
    def test_deliver_refuses_private(self):
        assert wc.deliver_callback("http://127.0.0.1:9/cb", None, _envelope()) is False
