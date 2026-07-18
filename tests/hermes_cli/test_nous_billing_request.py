"""Tests for the hermes_cli.nous_billing HTTP client's response handling.

Focus: a 2xx response with a NON-JSON body (e.g. a reverse-proxy / SPA fallback
HTML page when a route isn't actually serving the billing API) must surface as a
typed BillingError, NOT a raw json.JSONDecodeError that escapes the typed-error
contract and reads downstream as "not logged in".
"""

from __future__ import annotations

import io
import json
import socket
from contextlib import contextmanager

import pytest

from hermes_cli import nous_billing as nb


class _FakeResp(io.BytesIO):
    """Minimal urlopen() context-manager stand-in with a .status attribute."""

    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _http_error(status: int, body: bytes | dict[str, object] = b"{}", headers=None):
    """Build the real urllib HTTPError object _request catches."""
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    return nb.urllib.error.HTTPError(
        "https://portal.example/api/billing/state",
        status,
        "HTTP Error",
        headers or {},
        fp=io.BytesIO(body),
    )


def _sequence(monkeypatch, *outcomes, resolver=None):
    """Stub urlopen with ordered outcomes and record each Request."""
    seen: list[dict[str, object]] = []
    monkeypatch.setattr(nb, "_token_cache", None, raising=False)
    monkeypatch.setattr(
        nb,
        "_resolve_token_and_base",
        resolver or (lambda **kw: ("tok", "https://portal.example")),
    )

    def _fake_urlopen(req, timeout=None):
        seen.append(
            {
                "method": req.get_method(),
                "url": req.full_url,
                "data": json.loads(req.data.decode()) if req.data else None,
                "headers": {k.lower(): v for k, v in req.header_items()},
            }
        )
        outcome = outcomes[len(seen) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(nb.urllib.request, "urlopen", _fake_urlopen)
    return seen


@contextmanager
def _stub(monkeypatch, body: bytes, status: int = 200):
    # Bypass auth/token resolution entirely — we only exercise response parsing.
    monkeypatch.setattr(nb, "_resolve_token_and_base", lambda **kw: ("tok", "https://portal.example"))
    monkeypatch.setattr(nb, "_token_cache", None, raising=False)
    monkeypatch.setattr(nb.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(body, status))
    yield


def test_non_json_2xx_body_raises_typed_billing_error(monkeypatch):
    # A 200 that returns an HTML page (route not actually mounted) must NOT crash
    # with json.JSONDecodeError — it becomes a typed, non-auth BillingError.
    html = b"<!DOCTYPE html><html><head><title>Not Found</title></head></html>"
    with _stub(monkeypatch, html, status=200):
        with pytest.raises(nb.BillingError) as ei:
            nb.get_subscription_state()
    exc = ei.value
    # Not the auth subclass — this is "endpoint unavailable", not "logged out".
    assert not isinstance(exc, nb.BillingAuthError)
    assert getattr(exc, "error", None) == "endpoint_unavailable"


def test_empty_2xx_body_returns_empty_dict(monkeypatch):
    with _stub(monkeypatch, b"", status=200):
        assert nb.get_billing_state() == {}


def test_valid_json_2xx_body_parses(monkeypatch):
    payload = {"org": {"name": "Acme"}, "balanceUsd": "10"}
    with _stub(monkeypatch, json.dumps(payload).encode(), status=200):
        assert nb.get_billing_state() == payload


def test_transient_siblings_not_parent_child():
    assert issubclass(nb.BillingRateLimited, nb.BillingTransient)
    assert issubclass(nb.BillingStripeUnavailable, nb.BillingTransient)
    assert issubclass(nb.BillingUpgradeCapExceeded, nb.BillingTransient)
    assert not issubclass(nb.BillingStripeUnavailable, nb.BillingRateLimited)
    assert not issubclass(nb.BillingUpgradeCapExceeded, nb.BillingRateLimited)
    assert not issubclass(nb.BillingRateLimited, nb.BillingStripeUnavailable)


# ---------------------------------------------------------------------------
# Subscription change (V3): the request the client actually puts on the wire.
# ---------------------------------------------------------------------------


@contextmanager
def _capture(monkeypatch, body: bytes = b"{}", status: int = 200):
    """Stub urlopen, recording the urllib.request.Request the client built."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        nb, "_resolve_token_and_base", lambda **kw: ("tok", "https://portal.example")
    )

    def _fake_urlopen(req, timeout=None):
        seen["method"] = req.get_method()
        seen["url"] = req.full_url
        seen["data"] = json.loads(req.data.decode()) if req.data else None
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _FakeResp(body, status)

    monkeypatch.setattr(nb.urllib.request, "urlopen", _fake_urlopen)
    yield seen


def test_post_subscription_preview_request(monkeypatch):
    with _capture(monkeypatch) as seen:
        nb.post_subscription_preview(subscription_type_id="nous-chat-plan-40")
    assert seen["method"] == "POST"
    assert seen["url"] == "https://portal.example/api/billing/subscription/preview"
    assert seen["data"] == {"subscriptionTypeId": "nous-chat-plan-40"}


def test_put_pending_change_tier_change_request(monkeypatch):
    with _capture(monkeypatch) as seen:
        nb.put_subscription_pending_change(subscription_type_id="nous-chat-plan-10")
    assert seen["method"] == "PUT"
    assert (
        seen["url"] == "https://portal.example/api/billing/subscription/pending-change"
    )
    assert seen["data"] == {
        "type": "tier_change",
        "subscriptionTypeId": "nous-chat-plan-10",
    }


def test_put_pending_change_cancellation_request(monkeypatch):
    with _capture(monkeypatch) as seen:
        nb.put_subscription_pending_change(cancel=True)
    assert seen["method"] == "PUT"
    assert seen["data"] == {"type": "cancellation"}


def test_put_pending_change_without_tier_or_cancel_raises():
    # No urlopen stub: a bad call must fail BEFORE any network I/O.
    with pytest.raises(nb.BillingError) as ei:
        nb.put_subscription_pending_change()
    assert getattr(ei.value, "error", None) == "invalid_subscription_type"


def test_delete_pending_change_request(monkeypatch):
    with _capture(monkeypatch) as seen:
        nb.delete_subscription_pending_change()
    assert seen["method"] == "DELETE"
    assert (
        seen["url"] == "https://portal.example/api/billing/subscription/pending-change"
    )
    assert seen["data"] is None


def test_post_subscription_upgrade_sends_idempotency_key(monkeypatch):
    with _capture(monkeypatch) as seen:
        nb.post_subscription_upgrade(
            subscription_type_id="nous-chat-plan-40", idempotency_key="abc-123"
        )
    assert seen["method"] == "POST"
    assert seen["url"] == "https://portal.example/api/billing/subscription/upgrade"
    assert seen["data"] == {"subscriptionTypeId": "nous-chat-plan-40"}
    assert seen["headers"].get("idempotency-key") == "abc-123"


def test_post_subscription_upgrade_blank_key_raises():
    with pytest.raises(nb.BillingError) as ei:
        nb.post_subscription_upgrade(
            subscription_type_id="nous-chat-plan-40", idempotency_key="  "
        )
    assert getattr(ei.value, "error", None) == "idempotency_key_required"


# ---------------------------------------------------------------------------
# Wire-layer HTTP error mapping through _request.
# ---------------------------------------------------------------------------


def test_401_refreshes_token_and_retries_successfully(monkeypatch):
    # A cached-token 401 is retried once with a freshly resolved token.
    def _resolver(*, use_cache=True):
        token = "tok-fresh" if not use_cache else "tok-stale"
        return token, "https://portal.example"

    seen = _sequence(
        monkeypatch,
        _http_error(401),
        _FakeResp(b'{"ok": true}'),
        resolver=_resolver,
    )

    assert nb.get_billing_state() == {"ok": True}
    assert len(seen) == 2
    assert seen[0]["headers"]["authorization"] == "Bearer tok-stale"
    assert seen[1]["headers"]["authorization"] == "Bearer tok-fresh"


def test_401_retries_once_then_plain_401_is_terminal(monkeypatch):
    # A second plain 401 maps to auth failure, not another recursive retry.
    seen = _sequence(monkeypatch, _http_error(401), _http_error(401))

    with pytest.raises(nb.BillingAuthError):
        nb.get_billing_state()

    assert len(seen) == 2


def test_401_retry_terminal_session_revoked_preserves_recovery(monkeypatch):
    # session_revoked is only surfaced after the one refresh attempt is exhausted.
    seen = _sequence(
        monkeypatch,
        _http_error(401),
        _http_error(401, {"error": "session_revoked", "recovery": "login"}),
    )

    with pytest.raises(nb.BillingSessionRevoked) as ei:
        nb.get_billing_state()

    assert len(seen) == 2
    assert ei.value.recovery == "login"


def test_post_charge_preserves_idempotency_key_across_401_retry(monkeypatch):
    # Money requests must retry with the exact same Idempotency-Key.
    seen = _sequence(
        monkeypatch,
        _http_error(401),
        _FakeResp(b'{"chargeId": "ch_1"}', status=202),
    )

    assert nb.post_charge(amount_usd="10", idempotency_key="k1") == {
        "chargeId": "ch_1"
    }
    assert len(seen) == 2
    assert seen[0]["headers"]["idempotency-key"] == "k1"
    assert seen[1]["headers"]["idempotency-key"] == "k1"


def test_403_insufficient_scope_maps_through_request(monkeypatch):
    # Wire 403 insufficient_scope drives the lazy billing:manage step-up path.
    _sequence(monkeypatch, _http_error(403, {"error": "insufficient_scope"}))

    with pytest.raises(nb.BillingScopeRequired):
        nb.get_billing_state()


def test_403_remote_spending_revoked_maps_through_request(monkeypatch):
    # The wire discriminator keeps spend-revocation distinct from missing scope.
    _sequence(
        monkeypatch,
        _http_error(
            403,
            {
                "error": "remote_spending_revoked",
                "actor": "self",
                "recovery": "reconnect",
            },
        ),
    )

    with pytest.raises(nb.BillingRemoteSpendingRevoked) as ei:
        nb.get_billing_state()

    assert ei.value.actor == "self"
    assert ei.value.recovery == "reconnect"


def test_403_cli_billing_disabled_stays_generic_with_portal_url(monkeypatch):
    # Business denials stay generic so surfaces can branch on code/recovery.
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    _sequence(
        monkeypatch,
        _http_error(
            403,
            {
                "error": "cli_billing_disabled",
                "code": "remote_spending_disabled",
                "recovery": "enable_account_toggle",
                "portalUrl": "/billing",
            },
        ),
    )

    with pytest.raises(nb.BillingError) as ei:
        nb.get_billing_state()

    assert type(ei.value) is nb.BillingError
    assert ei.value.code == "remote_spending_disabled"
    assert ei.value.recovery == "enable_account_toggle"
    assert ei.value.portal_url == "https://portal.nousresearch.com/billing"


def test_429_retry_after_header_maps_to_rate_limited(monkeypatch):
    # 429 is rate-limited and reads Retry-After from headers.
    _sequence(monkeypatch, _http_error(429, headers={"Retry-After": "15"}))

    with pytest.raises(nb.BillingRateLimited) as ei:
        nb.get_billing_state()

    assert ei.value.retry_after == 15


def test_429_body_retry_after_hint_is_ignored_without_header(monkeypatch):
    # Pins current behavior: the client reads only the Retry-After header.
    # Whether it should also honor retryAfter in JSON is an open product question.
    _sequence(
        monkeypatch,
        _http_error(429, {"error": "rate_limited", "retryAfter": 20}),
    )

    with pytest.raises(nb.BillingRateLimited) as ei:
        nb.get_billing_state()

    assert ei.value.retry_after is None


def test_503_without_headers_is_rate_limited_without_retry_after(monkeypatch):
    # Missing headers must not crash retry-after parsing.
    _sequence(monkeypatch, _http_error(503, headers=None))

    with pytest.raises(nb.BillingRateLimited) as ei:
        nb.get_billing_state()

    assert ei.value.retry_after is None


def test_non_json_403_body_maps_to_generic_billing_denied(monkeypatch):
    # An unparseable 403 has no discriminator, so no subtype is selected.
    _sequence(monkeypatch, _http_error(403, b"<html>Forbidden</html>"))

    with pytest.raises(nb.BillingError) as ei:
        nb.get_billing_state()

    assert type(ei.value) is nb.BillingError
    assert str(ei.value) == "Billing request denied."


def test_non_json_second_401_maps_to_auth_error_not_session_revoked(monkeypatch):
    # The terminal 401 must not become session_revoked without a JSON discriminator.
    _sequence(
        monkeypatch,
        _http_error(401),
        _http_error(401, b"<html>Unauthorized</html>"),
    )

    with pytest.raises(nb.BillingAuthError) as ei:
        nb.get_billing_state()

    assert not isinstance(ei.value, nb.BillingSessionRevoked)


def test_non_json_500_body_maps_to_generic_failure(monkeypatch):
    # Generic server failures keep the status and default failure message.
    _sequence(monkeypatch, _http_error(500, b"<html>Oops</html>"))

    with pytest.raises(nb.BillingError) as ei:
        nb.get_billing_state()

    assert ei.value.status == 500
    assert str(ei.value) == "Billing request failed (500)."


def test_404_get_charge_status_maps_to_generic_billing_error(monkeypatch):
    # get_charge_status should surface unexpected 404s as typed generic errors.
    _sequence(monkeypatch, _http_error(404))

    with pytest.raises(nb.BillingError) as ei:
        nb.get_charge_status("ch_404")

    assert type(ei.value) is nb.BillingError
    assert ei.value.status == 404


def test_502_empty_json_body_maps_to_generic_error_without_error_code(monkeypatch):
    # Empty JSON error bodies preserve the HTTP status and no error discriminator.
    _sequence(monkeypatch, _http_error(502, {}))

    with pytest.raises(nb.BillingError) as ei:
        nb.get_billing_state()

    assert type(ei.value) is nb.BillingError
    assert ei.value.status == 502
    assert ei.value.error is None


def test_urlerror_dns_maps_to_network_error(monkeypatch):
    # urllib transport failures are normalized to BillingError(network_error).
    _sequence(
        monkeypatch,
        nb.urllib.error.URLError("Name or service not known"),
    )

    with pytest.raises(nb.BillingError) as ei:
        nb.get_billing_state()

    assert ei.value.error == "network_error"
    assert "Could not reach Nous Portal" in str(ei.value)


def test_urlerror_wrapped_timeout_maps_to_network_error(monkeypatch):
    # Real urllib timeouts arrive wrapped in URLError at this layer.
    _sequence(monkeypatch, nb.urllib.error.URLError(TimeoutError("timed out")))

    with pytest.raises(nb.BillingError) as ei:
        nb.get_billing_state()

    assert ei.value.error == "network_error"
    assert "Could not reach Nous Portal" in str(ei.value)


def test_bare_socket_timeout_normalizes_to_network_error(monkeypatch):
    # urlopen wraps connect-phase timeouts in URLError, but a read-phase timeout
    # is a bare TimeoutError — it must still honor the typed-BillingError contract.
    _sequence(monkeypatch, socket.timeout())

    with pytest.raises(nb.BillingError) as ei:
        nb.get_billing_state()

    assert ei.value.error == "network_error"
    assert "timed out" in str(ei.value)


def test_401_retry_re_resolves_base_url(monkeypatch):
    # The auth retry re-runs base resolution too, so preview/prod flips are honored.
    def _resolver(*, use_cache=True):
        if use_cache:
            return "tok-stale", "https://old.example"
        return "tok-fresh", "https://new.example"

    seen = _sequence(
        monkeypatch,
        _http_error(401),
        _FakeResp(b'{"ok": true}'),
        resolver=_resolver,
    )

    assert nb.get_billing_state() == {"ok": True}
    assert len(seen) == 2
    assert seen[0]["url"] == "https://old.example/api/billing/state"
    assert seen[1]["url"] == "https://new.example/api/billing/state"
