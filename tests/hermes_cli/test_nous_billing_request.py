"""Tests for the hermes_cli.nous_billing HTTP client's response handling.

Focus: a 2xx response with a NON-JSON body (e.g. a reverse-proxy / SPA fallback
HTML page when a route isn't actually serving the billing API) must surface as a
typed BillingError, NOT a raw json.JSONDecodeError that escapes the typed-error
contract and reads downstream as "not logged in".
"""

from __future__ import annotations

import io
import json
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
