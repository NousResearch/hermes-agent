"""D1/D2 + E2E: auxiliary xAI recovery under shared mode."""

from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import auth


def _jwt(exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    )
    return f"{header}.{payload}.sig"


@pytest.fixture
def shared_env(tmp_path, monkeypatch):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    monkeypatch.setenv("HERMES_XAI_SHARED_AUTH", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return {
        "store": shared_dir / "xai_oauth.json",
        "profile_auth": hermes_home / "auth.json",
    }


def _write_shared(env, *, access="at-1", refresh="rt-1", generation=1):
    payload = {
        "_schema": 1,
        "generation": generation,
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "auth_mode": "oauth_device_code",
        "last_refresh": "2026-07-01T00:00:00Z",
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"},
    }
    env["store"].write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_auth_refresh_provider_detects_api_x_ai():
    """D2: auto-routed aux clients on api.x.ai map to xai-oauth."""
    from agent.auxiliary_client import _auth_refresh_provider_for_route

    assert (
        _auth_refresh_provider_for_route("auto", "https://api.x.ai/v1/")
        == "xai-oauth"
    )
    assert (
        _auth_refresh_provider_for_route("auto", "https://api.x.ai/v1/chat/completions")
        == "xai-oauth"
    )


def test_refresh_provider_credentials_passes_rejected_bearer(shared_env, monkeypatch):
    """D1: shared recovery force-refreshes with the rejected bearer (no double POST)."""
    from agent.auxiliary_client import _refresh_provider_credentials

    old = _jwt(int(time.time()) + 30)
    _write_shared(shared_env, access=old, refresh="rt-1", generation=1)
    posts = []

    def fake_pure(access, refresh, **kwargs):
        posts.append(refresh)
        return {
            "access_token": "should-not",
            "refresh_token": "should-not",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)
    # Winner already rotated under the lock.
    _write_shared(shared_env, access="winner-at", refresh="winner-rt", generation=2)

    with patch("agent.auxiliary_client._evict_cached_clients") as evict:
        ok = _refresh_provider_credentials(
            "xai-oauth",
            rejected_api_key=old,
            expected_generation=1,
        )
    assert ok is True
    assert posts == []  # adopted winner — no second POST
    evict.assert_called()


def test_e2e_stale_request_401_canonical_refresh_replaces_client(
    shared_env, monkeypatch
):
    """E2E: stale bearer → auth error → canonical refresh → client rebuilt."""
    from agent import auxiliary_client as aux

    stale = "stale-bearer"
    fresh = "fresh-bearer"
    _write_shared(shared_env, access=stale, refresh="rt-1", generation=1)

    posts = []

    def fake_pure(access, refresh, **kwargs):
        posts.append(refresh)
        return {
            "access_token": fresh,
            "refresh_token": "rt-2",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T03:00:00Z",
            "id_token": "id-after-refresh",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)
    monkeypatch.setattr(
        auth,
        "_xai_oauth_discovery",
        lambda *_a, **_k: {"token_endpoint": "https://auth.x.ai/oauth/token"},
    )

    # Simulate refresh path used by call_llm recovery.
    with patch.object(aux, "_evict_cached_clients") as evict:
        ok = aux._refresh_provider_credentials(
            "xai-oauth",
            rejected_api_key=stale,
        )
    assert ok is True
    assert posts == ["rt-1"]
    shared = json.loads(shared_env["store"].read_text(encoding="utf-8"))
    assert shared["access_token"] == fresh
    assert shared["refresh_token"] == "rt-2"
    assert shared.get("id_token") == "id-after-refresh"
    assert shared["generation"] == 2
    evict.assert_called_with("xai-oauth")

    # Auto-routed path also resolves to xai-oauth (D2).
    assert (
        aux._auth_refresh_provider_for_route("auto", "https://api.x.ai/v1/")
        == "xai-oauth"
    )


def test_proxy_adapter_uses_canonical_and_retries_403(shared_env, monkeypatch):
    """C1/C3: proxy adapter resolves shared grant and refreshes on 403."""
    from hermes_cli.proxy.adapters.xai import XAIGrokAdapter
    from hermes_cli.proxy.adapters.base import UpstreamCredential

    stale = "stale-at"
    _write_shared(shared_env, access=stale, refresh="rt-1", generation=1)

    posts = []

    def fake_pure(access, refresh, **kwargs):
        posts.append(refresh)
        return {
            "access_token": "fresh-at",
            "refresh_token": "rt-2",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T04:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)
    monkeypatch.setattr(
        auth,
        "_xai_oauth_discovery",
        lambda *_a, **_k: {"token_endpoint": "https://auth.x.ai/oauth/token"},
    )

    adapter = XAIGrokAdapter()
    assert adapter.is_authenticated() is True
    cred = adapter.get_credential()
    assert cred.bearer == stale

    retry = adapter.get_retry_credential(
        failed_credential=UpstreamCredential(
            bearer=stale,
            base_url="https://api.x.ai/v1",
            expires_at=None,
        ),
        status_code=403,
    )
    assert retry is not None
    assert retry.bearer == "fresh-at"
    assert posts == ["rt-1"]


def test_auth_refresh_provider_unwraps_fallback_chain_label():
    """R6: composite fallback_chain[N](xai-oauth) must resolve to xai-oauth."""
    from agent.auxiliary_client import _auth_refresh_provider_for_route

    assert (
        _auth_refresh_provider_for_route(
            "fallback_chain[0](xai-oauth)",
            "https://api.x.ai/v1/",
        )
        == "xai-oauth"
    )
    # Even without a usable base URL, the label unwrap must win.
    assert (
        _auth_refresh_provider_for_route(
            "fallback_chain[2](xai-oauth)",
            "",
        )
        == "xai-oauth"
    )


def test_f6_fallback_chain_passes_rejected_bearer(shared_env, monkeypatch):
    """F6/R6: real fallback-chain path refreshes xAI without route-helper patch.

    Critical: do NOT patch ``_auth_refresh_provider_for_route`` — the bug was
    that composite labels like ``fallback_chain[0](xai-oauth)`` never resolved
    to the xai-oauth refresh branch. This test exercises the real path.
    """
    from agent import auxiliary_client as aux

    old = _jwt(int(time.time()) + 30)
    _write_shared(shared_env, access=old, refresh="rt-1", generation=1)
    posts = []
    refresh_providers = []

    def fake_pure(access, refresh, **kwargs):
        posts.append(refresh)
        return {
            "access_token": "should-not",
            "refresh_token": "should-not",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)
    # Concurrent winner already rotated the grant.
    _write_shared(shared_env, access="winner-at", refresh="winner-rt", generation=2)

    class BoomCompletions:
        def create(self, **kwargs):
            err = Exception("401 unauthorized")
            err.status_code = 401
            raise err

    class BoomChat:
        completions = BoomCompletions()

    class BoomClient:
        api_key = old
        base_url = "https://api.x.ai/v1/"
        chat = BoomChat()

    retry_client = SimpleNamespace(
        api_key="winner-at",
        base_url="https://api.x.ai/v1/",
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )
            )
        ),
    )

    real_refresh = aux._refresh_provider_credentials

    def tracking_refresh(provider, **kwargs):
        refresh_providers.append(provider)
        return real_refresh(provider, **kwargs)

    with patch.object(aux, "_is_auth_error", return_value=True), patch.object(
        aux, "_refresh_provider_credentials", side_effect=tracking_refresh
    ), patch.object(
        aux, "_get_cached_client", return_value=(retry_client, "grok")
    ), patch.object(
        aux, "_validate_llm_response", side_effect=lambda r, t: r
    ), patch.object(
        aux, "_build_call_kwargs", return_value={"model": "grok", "messages": []}
    ), patch.object(
        aux, "_evict_cached_clients"
    ), patch.object(
        aux, "_mark_provider_unhealthy"
    ):
        result = aux._call_fallback_candidate_sync(
            BoomClient(),
            "grok",
            "fallback_chain[0](xai-oauth)",
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            temperature=None,
            max_tokens=16,
            tools=None,
            effective_timeout=30.0,
            effective_extra_body={},
            reasoning_config=None,
        )
    assert result is not None
    assert posts == []  # adopted winner — no second POST
    # R6 de-mask: real route resolution must hand "xai-oauth" to refresh.
    assert refresh_providers == ["xai-oauth"]
