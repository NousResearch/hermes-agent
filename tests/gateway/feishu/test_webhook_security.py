"""Webhook security contract tests.

Tests the webhook deployment guards (rate limit, anomaly tracking,
signature delegation) as a black box. The Hermes-side guards live in
``gateway/platforms/feishu/webhook_guard.py`` (rate limiter, anomaly
tracker, body-size / Content-Type / read-timeout / JSON-parse). Signature
verification, verification_token, and URL-verification challenge are
owned by ``lark_oapi.channel.handle_webhook_request`` (SDK).

These tests inject a stub ``handle_request`` so the assertions stay
focused on the Hermes-owned guards. The stub re-implements the
``sha256(timestamp+nonce+encrypt_key+body)`` algorithm so 200/401/429/
anomaly assertions can run without spinning up a real SDK channel.

Locked-down constants (matching ``feishu/webhook_guard.py``):
  - Rate limit:        120 requests per 60-second window (composite key:
                       ``{app_id}:{path}:{remote_ip}``)
  - Anomaly threshold: 25 (warning log every N consecutive errors)
  - Body limit:        1 MB
"""

import asyncio
import dataclasses
import hashlib
import hmac
import json
import time
from unittest.mock import Mock

import pytest

pytest.importorskip("lark_oapi.channel")


def _signed_headers(*, body: bytes, encrypt_key: str, timestamp: str | None = None,
                    nonce: str = "n_test") -> dict:
    """Construct the headers Hermes' webhook validation expects.

    sha256(timestamp + nonce + encrypt_key + body_str) — the same
    algorithm Hermes used inline historically; reproduced here so the
    stub handler can validate signatures without involving the real SDK.
    """
    if timestamp is None:
        timestamp = str(int(time.time()))
    body_str = body.decode("utf-8", errors="replace")
    content = f"{timestamp}{nonce}{encrypt_key}{body_str}"
    signature = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "x-lark-request-timestamp": timestamp,
        "x-lark-request-nonce": nonce,
        "x-lark-signature": signature,
        "Content-Type": "application/json",
    }


@pytest.fixture
def webhook_harness(adapter_harness):
    """Harness configured for webhook mode with signing + token enabled.

    Anomaly state lives inside ``feishu.webhook_guard._AnomalyTracker``;
    the fixture injects a list-collector ``on_anomaly`` hook and spins up
    the webhook_guard handler so tests can:

      1) Call ``adapter._handle_webhook_request(req)`` — the thin shim
         delegates to ``self._webhook_handler``, which the fixture has
         populated.
      2) Inspect ``recorded_anomalies`` (list of WebhookAnomaly snapshots).
    """
    from gateway.platforms.feishu.webhook_guard import (
        WebhookAnomaly,
        RATE_WINDOW_SECONDS,
        RATE_LIMIT_MAX,
        _RateLimiter,
        _AnomalyTracker,
        _build_aiohttp_handler,
    )

    adapter_harness.adapter._settings = dataclasses.replace(
        adapter_harness.adapter._settings,
        encrypt_key="ek_test_secret",
        verification_token="vt_test",
    )
    adapter_harness.adapter._apply_settings(adapter_harness.adapter._settings)
    adapter_harness.adapter._bot_open_id = "ou_hermes_bot"
    adapter_harness.adapter._bot_user_id = "u_hermes_bot"
    adapter_harness.adapter._bot_name = "HermesBot"
    adapter_harness.adapter._group_policy = "open"
    adapter_harness.adapter._default_group_policy = "open"

    # List-collector for anomaly observations.
    recorded_anomalies: list = []

    def _collector(a: WebhookAnomaly) -> None:
        recorded_anomalies.append(a)

    rate_limiter = _RateLimiter(
        window_seconds=RATE_WINDOW_SECONDS,
        max_requests=RATE_LIMIT_MAX,
    )
    anomaly_tracker = _AnomalyTracker(on_anomaly=_collector)

    encrypt_key = adapter_harness.adapter._settings.encrypt_key
    verification_token = adapter_harness.adapter._settings.verification_token

    async def _stub_handle_request(headers, body_bytes):
        """Stand-in for ``channel.handle_webhook_request``.

        Reproduces the "valid signature → 200, invalid → 401" semantics
        using the historical sha256(ts+nonce+key+body) algorithm so the
        Hermes-owned guard assertions can run without bringing up a
        real SDK channel.
        """
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            return 400, b'{"code":400,"msg":"invalid json"}'
        if payload.get("type") == "url_verification":
            return 200, json.dumps(
                {"challenge": payload.get("challenge", "")}
            ).encode()
        # verification_token gate
        if verification_token:
            tok = (payload.get("header") or {}).get("token") or payload.get("token") or ""
            if not hmac.compare_digest(str(tok), verification_token):
                return 401, b"invalid token"
        # signature gate
        if encrypt_key:
            ts = str(headers.get("x-lark-request-timestamp", "") or "")
            nonce = str(headers.get("x-lark-request-nonce", "") or "")
            sig = str(headers.get("x-lark-signature", "") or "")
            if not (ts and nonce and sig):
                return 401, b"missing sig"
            body_str = body_bytes.decode("utf-8", errors="replace")
            content = f"{ts}{nonce}{encrypt_key}{body_str}"
            computed = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(computed, sig):
                return 401, b"bad sig"
        return 200, b'{"code":0,"msg":"ok"}'

    handler = _build_aiohttp_handler(
        app_id=adapter_harness.adapter._app_id,
        path=adapter_harness.adapter._webhook_path,
        handle_request=_stub_handle_request,
        rate_limiter=rate_limiter,
        anomaly_tracker=anomaly_tracker,
        encrypt_key=encrypt_key,
        verification_token=verification_token,
    )

    # Wire the thin shim to the handler.
    adapter_harness.adapter._webhook_handler = handler

    # Expose the collector so tests can assert on it.
    adapter_harness.recorded_anomalies = recorded_anomalies
    return adapter_harness


def _make_aiohttp_request(*, headers: dict, body: bytes, remote: str = "192.0.2.1"):
    """Build a minimal mock that satisfies the webhook_guard aiohttp handler."""
    req = Mock()
    req.headers = headers
    req.remote = remote
    req.content_length = len(body)

    async def _read():
        return body

    req.read = _read
    return req


def _make_event_body(token: str = "vt_test") -> bytes:
    """Build a body that will pass the verification-token gate.

    The header carries token=token; downstream the stub treats unknown
    event_types as ok (returns 200) so we land cleanly in the success
    path after token + signature checks pass.
    """
    return json.dumps({
        "schema": "2.0",
        "header": {"event_type": "im.message.receive_v1", "token": token},
        "event": {},
    }).encode("utf-8")


class TestWebhookSecurity:

    def test_valid_signature_returns_200(self, webhook_harness):
        body = _make_event_body()
        headers = _signed_headers(body=body, encrypt_key="ek_test_secret")
        req = _make_aiohttp_request(headers=headers, body=body)
        resp = asyncio.run(webhook_harness.adapter._handle_webhook_request(req))
        assert resp.status == 200

    def test_missing_signature_headers_return_401_without_calling_sdk(self):
        from gateway.platforms.feishu.webhook_guard import (
            RATE_LIMIT_MAX,
            RATE_WINDOW_SECONDS,
            _AnomalyTracker,
            _RateLimiter,
            _build_aiohttp_handler,
        )

        called = False

        async def _sdk_handler(_headers, _body):
            nonlocal called
            called = True
            return 200, b'{"code":0,"msg":"ok"}'

        handler = _build_aiohttp_handler(
            app_id="cli_test_app",
            path="/feishu/webhook",
            handle_request=_sdk_handler,
            rate_limiter=_RateLimiter(
                window_seconds=RATE_WINDOW_SECONDS,
                max_requests=RATE_LIMIT_MAX,
            ),
            anomaly_tracker=_AnomalyTracker(on_anomaly=lambda _a: None),
            encrypt_key="ek_test_secret",
            verification_token="vt_test",
        )

        body = _make_event_body(token="vt_test")
        req = _make_aiohttp_request(
            headers={"Content-Type": "application/json"},
            body=body,
        )

        resp = asyncio.run(handler(req))

        assert resp.status == 401
        assert called is False

    def test_encrypted_payload_is_rejected_before_sdk(self):
        from gateway.platforms.feishu.webhook_guard import (
            RATE_LIMIT_MAX,
            RATE_WINDOW_SECONDS,
            _AnomalyTracker,
            _RateLimiter,
            _build_aiohttp_handler,
        )

        called = False

        async def _sdk_handler(_headers, _body):
            nonlocal called
            called = True
            return 200, b'{"code":0,"msg":"ok"}'

        body = json.dumps({
            "encrypt": "ciphertext",
            "header": {"token": "vt_test"},
        }).encode("utf-8")
        headers = _signed_headers(body=body, encrypt_key="ek_test_secret")
        handler = _build_aiohttp_handler(
            app_id="cli_test_app",
            path="/feishu/webhook",
            handle_request=_sdk_handler,
            rate_limiter=_RateLimiter(
                window_seconds=RATE_WINDOW_SECONDS,
                max_requests=RATE_LIMIT_MAX,
            ),
            anomaly_tracker=_AnomalyTracker(on_anomaly=lambda _a: None),
            encrypt_key="ek_test_secret",
            verification_token="vt_test",
        )

        resp = asyncio.run(
            handler(_make_aiohttp_request(headers=headers, body=body))
        )

        assert resp.status == 400
        assert called is False

    def test_lowercase_lark_headers_are_forwarded_with_sdk_canonical_names(self):
        """aiohttp accepts case-insensitive headers, but the SDK dispatcher
        reads exact canonical keys from a plain dict.
        """
        from gateway.platforms.feishu.webhook_guard import (
            RATE_LIMIT_MAX,
            RATE_WINDOW_SECONDS,
            _AnomalyTracker,
            _RateLimiter,
            _build_aiohttp_handler,
        )

        seen_headers = {}

        async def _sdk_handler(headers, _body):
            seen_headers.update(headers)
            return 200, b'{"code":0,"msg":"ok"}'

        handler = _build_aiohttp_handler(
            app_id="cli_test_app",
            path="/feishu/webhook",
            handle_request=_sdk_handler,
            rate_limiter=_RateLimiter(
                window_seconds=RATE_WINDOW_SECONDS,
                max_requests=RATE_LIMIT_MAX,
            ),
            anomaly_tracker=_AnomalyTracker(on_anomaly=lambda _a: None),
        )

        body = _make_event_body()
        headers = _signed_headers(body=body, encrypt_key="ek_test_secret")
        lower_headers = {
            k.lower() if k.startswith("x-lark-") else k: v
            for k, v in headers.items()
        }
        req = _make_aiohttp_request(headers=lower_headers, body=body)

        resp = asyncio.run(handler(req))

        assert resp.status == 200
        assert seen_headers["X-Lark-Request-Timestamp"] == lower_headers[
            "x-lark-request-timestamp"
        ]
        assert seen_headers["X-Lark-Request-Nonce"] == lower_headers[
            "x-lark-request-nonce"
        ]
        assert seen_headers["X-Lark-Signature"] == lower_headers["x-lark-signature"]

    def test_invalid_signature_returns_401(self, webhook_harness):
        # Sign with the WRONG key but supply the correct verification token so
        # we land at the signature gate (not the earlier token gate).
        body = _make_event_body(token="vt_test")
        headers = _signed_headers(body=body, encrypt_key="WRONG_KEY")
        req = _make_aiohttp_request(headers=headers, body=body)
        resp = asyncio.run(webhook_harness.adapter._handle_webhook_request(req))
        assert resp.status == 401

    @pytest.mark.parametrize(
        "sdk_body",
        [
            b'{"msg":"invalid verification_token"}',
            b'{"msg":"signature verification failed"}',
        ],
    )
    def test_sdk_auth_failure_500_is_normalized_to_401(self, sdk_body):
        """SDK 1.6.x reports some auth failures as HTTP 500.

        The deployment guard should expose that as a client authentication
        failure, not as a gateway/server failure.
        """
        from gateway.platforms.feishu.webhook_guard import (
            RATE_LIMIT_MAX,
            RATE_WINDOW_SECONDS,
            _AnomalyTracker,
            _RateLimiter,
            _build_aiohttp_handler,
        )

        async def _sdk_handler(_headers, _body):
            return 500, sdk_body

        handler = _build_aiohttp_handler(
            app_id="cli_test_app",
            path="/feishu/webhook",
            handle_request=_sdk_handler,
            rate_limiter=_RateLimiter(
                window_seconds=RATE_WINDOW_SECONDS,
                max_requests=RATE_LIMIT_MAX,
            ),
            anomaly_tracker=_AnomalyTracker(on_anomaly=lambda _a: None),
        )

        body = _make_event_body(token="wrong_token")
        headers = _signed_headers(body=body, encrypt_key="ek_test_secret")
        req = _make_aiohttp_request(headers=headers, body=body)

        resp = asyncio.run(handler(req))

        assert resp.status == 401

    def test_rate_limit_blocks_after_threshold(self, webhook_harness):
        """Cap is 120 per 60s window. The 121st same-key request must
        return 429 before any other validation runs."""
        body = _make_event_body()

        async def _flood():
            statuses = []
            for _ in range(125):
                headers = _signed_headers(body=body, encrypt_key="ek_test_secret")
                req = _make_aiohttp_request(
                    headers=headers, body=body, remote="192.0.2.99"
                )
                resp = await webhook_harness.adapter._handle_webhook_request(req)
                statuses.append(resp.status)
            return statuses

        statuses = asyncio.run(_flood())
        assert 429 in statuses, (
            "Rate limiter should block after exceeding window cap "
            "(expected at request 121 of 125 sent)"
        )

    def test_anomaly_recorded_on_repeated_errors(self, webhook_harness):
        """Repeated malformed JSON requests (400 path) should fire the
        on_anomaly hook for the offending remote IP."""
        bad_body = b"not_json"

        async def _flood():
            for _ in range(30):
                headers = _signed_headers(body=bad_body, encrypt_key="ek_test_secret")
                req = _make_aiohttp_request(
                    headers=headers, body=bad_body, remote="192.0.2.55"
                )
                await webhook_harness.adapter._handle_webhook_request(req)

        asyncio.run(_flood())

        # Anomaly state lives in webhook_guard's _AnomalyTracker, not on
        # the adapter. Tests assert via the on_anomaly hook collector.
        recorded = webhook_harness.recorded_anomalies
        matching = [a for a in recorded if a.remote_ip == "192.0.2.55"]
        assert matching, (
            f"Expected at least one anomaly for 192.0.2.55; got remote_ips="
            f"{sorted({a.remote_ip for a in recorded})}"
        )
        last = matching[-1]
        assert last.count >= 1
        assert last.status_code == 400  # bad-json path returns 400


class TestStartupSecretGuard:
    """start_webhook_server must refuse to bind when both auth secrets are empty.

    Either FEISHU_ENCRYPT_KEY (signature check) or FEISHU_VERIFICATION_TOKEN
    (token check) must be configured. With neither, _handle() lets every POST
    through to the SDK -- equivalent to an unauthenticated public endpoint.
    """

    @pytest.mark.asyncio
    async def test_refuses_to_start_when_both_secrets_empty(self):
        from gateway.platforms.feishu.webhook_guard import (
            RateLimit,
            start_webhook_server,
        )

        async def _noop_handler(headers, body):
            return 200, b"{}"

        with pytest.raises(RuntimeError, match="at least one"):
            await start_webhook_server(
                host="127.0.0.1",
                port=0,
                path="/feishu/webhook",
                app_id="cli_test",
                handle_request=_noop_handler,
                rate_limit=RateLimit(window_seconds=60, max_requests=120),
                on_anomaly=lambda evt: None,
                encrypt_key="",
                verification_token="",
            )

    @pytest.mark.asyncio
    async def test_starts_when_only_verification_token_set(self, unused_tcp_port):
        from gateway.platforms.feishu.webhook_guard import (
            RateLimit,
            start_webhook_server,
        )

        async def _noop_handler(headers, body):
            return 200, b"{}"

        runner = await start_webhook_server(
            host="127.0.0.1",
            port=unused_tcp_port,
            path="/feishu/webhook",
            app_id="cli_test",
            handle_request=_noop_handler,
            rate_limit=RateLimit(window_seconds=60, max_requests=120),
            on_anomaly=lambda evt: None,
            encrypt_key="",
            verification_token="some-token",
        )
        try:
            assert runner.addresses, "site must be bound to an address after start"
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_starts_when_only_encrypt_key_set(self, unused_tcp_port):
        from gateway.platforms.feishu.webhook_guard import (
            RateLimit,
            start_webhook_server,
        )

        async def _noop_handler(headers, body):
            return 200, b"{}"

        runner = await start_webhook_server(
            host="127.0.0.1",
            port=unused_tcp_port,
            path="/feishu/webhook",
            app_id="cli_test",
            handle_request=_noop_handler,
            rate_limit=RateLimit(window_seconds=60, max_requests=120),
            on_anomaly=lambda evt: None,
            encrypt_key="some-key",
            verification_token="",
        )
        try:
            assert runner.addresses, "site must be bound to an address after start"
        finally:
            await runner.cleanup()


class TestUrlVerificationDoesNotResetAnomaly:
    """Sending url_verification must not reset the anomaly counter; otherwise an
    attacker can wipe their bad-request history at will between probes.
    """

    @pytest.mark.asyncio
    async def test_url_verification_does_not_clear_anomaly_counter(self):
        from gateway.platforms.feishu.webhook_guard import (
            _AnomalyTracker,
            _build_aiohttp_handler,
            _RateLimiter,
        )

        anomaly_events = []
        anomaly = _AnomalyTracker(on_anomaly=lambda evt: anomaly_events.append(evt))
        # Pre-populate: simulate an attacker who has accumulated anomaly hits.
        anomaly.record("198.51.100.7", 401, note="token")
        anomaly.record("198.51.100.7", 401, note="token")

        async def _never_called(headers, body):
            raise AssertionError("SDK handler must not run for url_verification")

        handler = _build_aiohttp_handler(
            app_id="cli_test",
            path="/feishu/webhook",
            handle_request=_never_called,
            rate_limiter=_RateLimiter(window_seconds=60, max_requests=120),
            anomaly_tracker=anomaly,
            encrypt_key="key",
            verification_token="tok",
        )

        # Use the existing _make_aiohttp_request shim (same pattern as other tests).
        request = _make_aiohttp_request(
            remote="198.51.100.7",
            headers={"Content-Type": "application/json"},
            body=b'{"type":"url_verification","challenge":"abc"}',
        )
        response = await handler(request)
        assert response.status == 200

        # Anomaly counter must still be 2 (not cleared by the url_verification path).
        assert anomaly._counts["198.51.100.7"][0] == 2, (
            "url_verification path must not clear anomaly counter"
        )


class TestRateLimiterFailClosed:
    """When the rate-limiter key table is full and pruning yields nothing, the
    next new key must be REJECTED (return False), not silently allowed."""

    def test_full_table_fails_closed(self):
        from gateway.platforms.feishu.webhook_guard import _RateLimiter

        limiter = _RateLimiter(
            window_seconds=60, max_requests=120, max_keys=3,
        )
        # Fill the table with fresh entries that can't be pruned.
        assert limiter.allow("k1")
        assert limiter.allow("k2")
        assert limiter.allow("k3")
        # New key arrives while table is full -- must be rejected.
        assert limiter.allow("k4") is False

    def test_new_key_admitted_when_table_not_full(self):
        from gateway.platforms.feishu.webhook_guard import _RateLimiter

        limiter = _RateLimiter(window_seconds=60, max_requests=120, max_keys=3)
        assert limiter.allow("k1")
        assert limiter.allow("k2")
        # Brand new key while there is still room — must be admitted.
        assert limiter.allow("k3")

    def test_full_table_admits_after_stale_entries_prune(self, monkeypatch):
        """Once entries time out of their window, new keys must be admitted again."""
        import time
        from gateway.platforms.feishu import webhook_guard

        limiter = webhook_guard._RateLimiter(
            window_seconds=60, max_requests=120, max_keys=3,
        )
        # Fill the table at t=0.
        fake_now = [1_000_000.0]
        monkeypatch.setattr(
            webhook_guard.time, "time", lambda: fake_now[0],
        )
        assert limiter.allow("k1")
        assert limiter.allow("k2")
        assert limiter.allow("k3")
        # New key at the same instant must be rejected (table full, nothing stale).
        assert limiter.allow("k4") is False
        # Advance past the window: k1/k2/k3 are now stale and can be pruned.
        fake_now[0] += 61
        assert limiter.allow("k5"), "new key must be admitted after stale entries prune"
