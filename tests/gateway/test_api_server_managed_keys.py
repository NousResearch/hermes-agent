"""Two-tier authentication tests for the Hermes OpenAI-compatible API.

These tests pin the new behaviour of ``_check_auth`` introduced when we
added Dashboard-managed API keys:

* Legacy ``API_SERVER_KEY`` keeps working unchanged.
* A wrong legacy key still returns 401.
* A correctly minted managed key authenticates.
* A revoked managed key returns 401.
* The legacy loopback pass-through closes when managed keys exist.
* Two different managed keys produce two distinct auth identities.
* ``/v1/capabilities`` reports ``auth.required`` true when managed keys exist.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import MethodType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_hermes_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def fresh_keys(temp_hermes_home):
    import hermes_cli.api_server_keys as k
    return k


@pytest.fixture
def adapter_cls(temp_hermes_home):
    # Importing the adapter pulls in a lot of gateway plumbing; we lazy-load
    # here so each test only pays that cost once per session.
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig
    return APIServerAdapter, PlatformConfig


def _make_request(headers: dict) -> MagicMock:
    """A MagicMock that mimics the small surface area of an aiohttp
    ``web.Request`` that ``_check_auth`` (and its audit log suffix helper)
    touch: ``headers``, ``method``, ``path_qs``, ``remote``, ``transport``.

    ``_check_auth`` stamps the authenticated identity via ``setattr`` so
    plain attribute access works on MagicMock out of the box. The original
    test fixture tried to fake ``__setitem__`` — but Python only honors
    ``__setitem__`` for ``request[k] = v`` when the type declares it at
    class level; an instance-level monkeypatch silently raises TypeError,
    which was being swallowed by the legacy ``except (UnicodeEncodeError,
    TypeError)`` clause and turning every request into a 401.
    """
    req = MagicMock()
    req.headers = headers
    req.method = "GET"
    req.path_qs = "/v1/chat/completions"
    req.remote = "127.0.0.1"
    req.transport = None
    return req


# ─── legacy behaviour is byte-for-byte unchanged ─────────────────────────────


def test_legacy_key_still_authenticates(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-legacy"}))
    assert adapter._api_key == "sk-legacy", adapter._api_key
    req = _make_request({"Authorization": "Bearer sk-legacy"})
    result = adapter._check_auth(req)
    assert result is None
    assert req.api_key_identity == {"kind": "legacy"}


def test_wrong_legacy_key_returns_401(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-legacy"}))
    req = _make_request({"Authorization": "Bearer wrong-key"})
    resp = adapter._check_auth(req)
    assert resp is not None
    assert resp.status == 401


def test_non_ascii_legacy_token_returns_401_not_500(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-legacy"}))
    req = _make_request({"Authorization": "Bearer ské-not-the-key"})
    resp = adapter._check_auth(req)  # must not raise
    assert resp is not None
    assert resp.status == 401


def test_no_legacy_no_managed_loopback_passthrough(adapter_cls, fresh_keys):
    """The historical test ``test_no_key_configured_allows_all``:
    no API_SERVER_KEY AND no managed keys ⇒ loopback request is allowed."""
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    req = _make_request({})
    assert adapter._check_auth(req) is None


# ─── tier-2 managed key behaviour ────────────────────────────────────────────


def test_managed_key_authenticates_alongside_legacy(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-legacy"}))
    row = fresh_keys.create_api_key(name="integration-test")
    req = _make_request({"Authorization": f"Bearer {row['plaintext']}"})
    resp = adapter._check_auth(req)
    assert resp is None
    assert req.api_key_identity["kind"] == "managed"
    assert req.api_key_identity["id"] == row["id"]


def test_managed_key_authenticates_without_legacy(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))  # no legacy
    row = fresh_keys.create_api_key(name="no-legacy")
    req = _make_request({"Authorization": f"Bearer {row['plaintext']}"})
    assert adapter._check_auth(req) is None


def test_wrong_managed_key_returns_401(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    fresh_keys.create_api_key(name="real")
    req = _make_request({"Authorization": "Bearer hm_live_wrongtoken"})
    resp = adapter._check_auth(req)
    assert resp is not None
    assert resp.status == 401


def test_revoked_managed_key_returns_401(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    row = fresh_keys.create_api_key(name="to-revoke")
    fresh_keys.revoke_api_key(row["id"])
    req = _make_request({"Authorization": f"Bearer {row['plaintext']}"})
    resp = adapter._check_auth(req)
    assert resp is not None
    assert resp.status == 401


def test_loopback_passthrough_closes_when_managed_keys_exist(adapter_cls, fresh_keys):
    """The historical loopback pass-through only applies when NO managed
    keys are configured. Once an operator creates one, every request
    without a valid bearer must 401."""
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    fresh_keys.create_api_key(name="any")  # presence is enough
    req = _make_request({})
    resp = adapter._check_auth(req)
    assert resp is not None
    assert resp.status == 401


# ─── two distinct managed keys = two distinct identities ──────────────────────


def test_two_managed_keys_have_distinct_identities(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    a = fresh_keys.create_api_key(name="a")
    b = fresh_keys.create_api_key(name="b")
    req_a = _make_request({"Authorization": f"Bearer {a['plaintext']}"})
    req_b = _make_request({"Authorization": f"Bearer {b['plaintext']}"})
    assert adapter._check_auth(req_a) is None
    assert adapter._check_auth(req_b) is None
    assert req_a.api_key_identity["id"] != req_b.api_key_identity["id"]
    assert req_a.api_key_identity["id"] == a["id"]
    assert req_b.api_key_identity["id"] == b["id"]


# ─── /v1/capabilities signal ────────────────────────────────────────────────


def test_capabilities_reports_required_when_managed_keys_exist(
    adapter_cls, fresh_keys
):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    row = fresh_keys.create_api_key(name="capability-test")

    # Drive the coroutine to completion via a small inline runner.
    import asyncio
    req = _make_request({"Authorization": f"Bearer {row['plaintext']}"})
    body = asyncio.run(adapter._handle_capabilities(req))
    payload = json.loads(body.body.decode("utf-8"))
    assert payload["auth"]["required"] is True
    assert "managed" in payload["auth"]["modes"]


def test_capabilities_required_false_when_neither_configured(adapter_cls, fresh_keys):
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    import asyncio
    req = _make_request({})
    body = asyncio.run(adapter._handle_capabilities(req))
    payload = json.loads(body.body.decode("utf-8"))
    assert payload["auth"]["required"] is False
    assert payload["auth"]["modes"] == ["none"]


# ─── idempotency principal uses managed-key id ──────────────────────────────


def test_run_idempotency_scope_uses_managed_key_id(adapter_cls, fresh_keys):
    from gateway.platforms.api_server_runs import _run_idempotency_scope
    APIServerAdapter, PlatformConfig = adapter_cls
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    a = fresh_keys.create_api_key(name="idem-a")
    b = fresh_keys.create_api_key(name="idem-b")

    req_a = _make_request({"Authorization": f"Bearer {a['plaintext']}"})
    req_b = _make_request({"Authorization": f"Bearer {b['plaintext']}"})
    # Run auth so each request gets its identity stamped.
    assert adapter._check_auth(req_a) is None
    assert adapter._check_auth(req_b) is None

    fake_api_server = MagicMock()
    fake_api_server._api_request_profile.get.return_value = None
    scope_a = _run_idempotency_scope(adapter, req_a, _api_server=fake_api_server)
    scope_b = _run_idempotency_scope(adapter, req_b, _api_server=fake_api_server)
    assert scope_a != scope_b  # independent principals → independent buckets