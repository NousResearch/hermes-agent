"""Tests for MiniMax live-catalog discovery in the /model picker.

Covers the plumbing fixed by feat/minimax-live-catalog:

  1. The ``minimax`` and ``minimax-cn`` provider profiles point their
     ``models_url`` at the OpenAI-compat ``/v1/models`` endpoint, not at
     ``<base_url>/models`` (which would 404 on ``/anthropic/models``).

  2. ``provider_model_ids("minimax")`` and ``provider_model_ids("minimax-cn")``
     use the profile's ``fetch_models`` (live catalog) and fall back to the
     static ``_PROVIDER_MODELS`` table only when the live fetch fails or
     returns no models.

  3. ``list_authenticated_providers()`` shows the live catalog in the
     Telegram/Discord /model picker, so model ids added server-side (e.g.
     ``MiniMax-M3``) become visible without a Hermes release.

  4. ``minimax-oauth`` still works even though its ``auth_type`` skips the
     generic live-fetch path — it falls through to the static catalog.

All HTTP is mocked — no real MiniMax credentials or network required.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────

LIVE_CATALOG_INTL = {
    "object": "list",
    "data": [
        {"id": "MiniMax-M2",             "object": "model"},
        {"id": "MiniMax-M2.1",           "object": "model"},
        {"id": "MiniMax-M2.5",           "object": "model"},
        {"id": "MiniMax-M2.7",           "object": "model"},
        {"id": "MiniMax-M2.7-highspeed", "object": "model"},
        {"id": "MiniMax-M3",             "object": "model"},  # the user's case
    ],
}

LIVE_CATALOG_CN = {
    "object": "list",
    "data": [
        {"id": "abab6.5s-chat", "object": "model"},
        {"id": "abab7-chat",    "object": "model"},
        {"id": "MiniMax-M2",    "object": "model"},
        {"id": "MiniMax-M2.7",  "object": "model"},
    ],
}


class _CallRecorder(list):
    """List subclass that records every urlopen call for assertions."""


def _make_urlopen(
    catalog: dict,
    recorder: _CallRecorder | None = None,
    *,
    minimax_response: str = "catalog",
):
    """Return a callable suitable for patching urllib.request.urlopen.

    The returned callable acts as a context manager (urllib requires this
    when the response is fetched via ``with urlopen(req) as resp:``) AND
    accepts the bare-call shape used elsewhere in the codebase.

    If ``recorder`` is given, every call appends ``(url, auth_header)``.

    For non-MiniMax URLs (e.g. the Nous Portal remote manifest fetched
    by ``hermes_cli.model_catalog``) the helper always returns a valid
    empty-but-parseable OpenAI-compat response, so the surrounding
    fetch path doesn't blow up.

    ``minimax_response`` controls what the helper returns for MiniMax URLs:
      - "catalog"  — return ``catalog`` (live success path)
      - "empty"    — return ``{"data": []}`` (degenerate live response)
      - "401"      — raise ``HTTPError(401)``
      - "network"  — raise ``OSError`` (network down)
    """
    rec = recorder if recorder is not None else _CallRecorder()
    body = json.dumps(catalog).encode()
    empty_body = json.dumps({"object": "list", "data": []}).encode()
    minimax_hosts = ("api.minimax.io", "api.minimaxi.com")

    class _Resp:
        def __init__(self, body_bytes: bytes):
            self._body = body_bytes

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def _open(req, timeout=None):
        url = req.full_url
        auth = req.get_header("Authorization") or ""
        rec.append((url, auth))
        is_minimax = any(host in url for host in minimax_hosts)
        if is_minimax:
            if minimax_response == "catalog":
                return _Resp(body)
            if minimax_response == "empty":
                return _Resp(empty_body)
            if minimax_response == "401":
                err_body = json.dumps({
                    "type": "error",
                    "error": {"type": "authorized_error", "http_code": "401"},
                }).encode()
                raise _http_error(req, 401, err_body)
            if minimax_response == "network":
                raise OSError("network down")
        # Pass through for everything else (e.g. the Nous remote manifest)
        return _Resp(empty_body)

    return _open, rec


def _http_error(req, code: int = 401, body: bytes = b""):
    """Construct a urllib.error.HTTPError for use in side_effects."""
    from email.message import Message
    from urllib.error import HTTPError

    hdrs = Message()
    hdrs["Content-Type"] = "application/json"
    return HTTPError(req.full_url, code, "Unauthorized", hdrs, io.BytesIO(body))


# ── 1. Profile models_url wiring ────────────────────────────────────────


class TestMiniMaxProfileModelsUrl:
    """Each profile must set models_url to the OpenAI-compat /v1/models."""

    def test_minimax_profile_models_url(self):
        from providers import get_provider_profile

        p = get_provider_profile("minimax")
        assert p is not None, "minimax profile must be registered"
        assert p.models_url == "https://api.minimax.io/v1/models", (
            f"minimax models_url should point at /v1/models catalog, "
            f"got {p.models_url!r}"
        )

    def test_minimax_cn_profile_models_url(self):
        from providers import get_provider_profile

        p = get_provider_profile("minimax-cn")
        assert p is not None, "minimax-cn profile must be registered"
        assert p.models_url == "https://api.minimaxi.com/v1/models", (
            f"minimax-cn models_url should point at /v1/models catalog, "
            f"got {p.models_url!r}"
        )

    def test_minimax_oauth_profile_models_url(self):
        from providers import get_provider_profile

        p = get_provider_profile("minimax-oauth")
        assert p is not None, "minimax-oauth profile must be registered"
        # OAuth uses the international endpoint; auth header carries the
        # bearer token instead of the static API key.
        assert p.models_url == "https://api.minimax.io/v1/models"

    def test_models_url_differs_from_inference_base(self):
        """``models_url`` must NOT equal ``base_url + /models`` — that
        naive derivation would hit the 404 ``/anthropic/models`` path.
        """
        from providers import get_provider_profile

        for slug, expected_inference in (
            ("minimax",    "https://api.minimax.io/anthropic"),
            ("minimax-cn", "https://api.minimaxi.com/anthropic"),
        ):
            p = get_provider_profile(slug)
            assert p is not None, f"{slug} profile must be registered"
            naive = p.base_url.rstrip("/") + "/models"
            assert p.models_url != naive, (
                f"{slug}: models_url={p.models_url!r} collides with the "
                f"naive base_url/models={naive!r} (would 404)"
            )
            assert expected_inference in p.base_url, (
                f"{slug}: base_url unexpectedly changed to {p.base_url!r}"
            )


# ── 2. provider_model_ids() live catalog path ───────────────────────────


class TestProviderModelIdsMiniMax:
    """provider_model_ids('minimax') should hit the live /v1/models."""

    def test_minimax_returns_live_catalog_when_key_set(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        open_fn, rec = _make_urlopen(LIVE_CATALOG_INTL)

        with patch("urllib.request.urlopen", side_effect=open_fn):
            result = provider_model_ids("minimax")

        assert "MiniMax-M3" in result, (
            f"MiniMax-M3 (released server-side) should appear in live "
            f"catalog, got {result!r}"
        )
        assert "MiniMax-M2.7" in result
        # Live URL was hit, not the naive /anthropic/models
        assert rec[0][0] == "https://api.minimax.io/v1/models"
        assert rec[0][1] == "Bearer test-key"

    def test_minimax_cn_returns_live_catalog_when_key_set(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-key-cn")
        open_fn, rec = _make_urlopen(LIVE_CATALOG_CN)

        with patch("urllib.request.urlopen", side_effect=open_fn):
            result = provider_model_ids("minimax-cn")

        assert "abab7-chat" in result
        assert "MiniMax-M2.7" in result
        assert rec[0][0] == "https://api.minimaxi.com/v1/models"
        assert rec[0][1] == "Bearer test-key-cn"

    def test_minimax_falls_back_to_static_when_no_key(self, monkeypatch):
        """No API key → live fetch skipped → static+models.dev catalog."""
        from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids

        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        # Side-effect still installed so the Nous remote-manifest fetch
        # (a sibling of the minimax fetch) doesn't break with a
        # MagicMock body — but minimax_response="catalog" is irrelevant
        # because the test asserts minimax wasn't fetched at all.
        open_fn, rec = _make_urlopen(LIVE_CATALOG_INTL)

        with patch("urllib.request.urlopen", side_effect=open_fn):
            result = provider_model_ids("minimax")

        # No call to the minimax host happened
        assert not any("minimax" in url for url, _auth in rec), (
            f"minimax should not be fetched when MINIMAX_API_KEY is unset, "
            f"but got: {rec!r}"
        )
        # Static list entries are present (the fallback worked)
        assert set(_PROVIDER_MODELS["minimax"]).issubset(set(result))

    def test_minimax_falls_back_to_static_on_401(self, monkeypatch):
        """401 from /v1/models (stale key) → static+models.dev catalog."""
        from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids

        monkeypatch.setenv("MINIMAX_API_KEY", "bad-key")
        open_fn, rec = _make_urlopen(
            LIVE_CATALOG_INTL, minimax_response="401",
        )

        with patch("urllib.request.urlopen", side_effect=open_fn):
            result = provider_model_ids("minimax")

        # The minimax URL was attempted (then 401'd)
        assert any("minimax" in url for url, _auth in rec)
        # Static list entries are present
        assert set(_PROVIDER_MODELS["minimax"]).issubset(set(result))
        # A live-only id (not in static, not in models.dev) is absent —
        # confirms the live catalog was NOT used.
        assert "MiniMax-M99-future" not in result

    def test_minimax_falls_back_to_static_on_network_error(self, monkeypatch):
        """Network failure → static+models.dev catalog, no crash."""
        from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        open_fn, _rec = _make_urlopen(
            LIVE_CATALOG_INTL, minimax_response="network",
        )

        with patch("urllib.request.urlopen", side_effect=open_fn):
            result = provider_model_ids("minimax")

        assert set(_PROVIDER_MODELS["minimax"]).issubset(set(result))

    def test_minimax_falls_back_to_static_on_empty_live_response(self, monkeypatch):
        """If the live endpoint returns 0 models, fall back rather than
        expose an empty picker row (list_picker_providers() would then
        drop the row per test_empty_models_row_dropped).
        """
        from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        open_fn, _rec = _make_urlopen(
            LIVE_CATALOG_INTL, minimax_response="empty",
        )

        with patch("urllib.request.urlopen", side_effect=open_fn):
            result = provider_model_ids("minimax")

        assert set(_PROVIDER_MODELS["minimax"]).issubset(set(result))

    def test_minimax_passes_through_unknown_live_models(self, monkeypatch):
        """Live catalog should pass through unchanged (the caller merges
        with the static list only when live returns None).
        """
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        open_fn, _rec = _make_urlopen({
            "data": [{"id": "MiniMax-M99-future"}],
        })

        with patch("urllib.request.urlopen", side_effect=open_fn):
            result = provider_model_ids("minimax")

        assert result == ["MiniMax-M99-future"]


# ── 3. list_authenticated_providers() — picker integration ──────────────


class TestListAuthenticatedProvidersMiniMax:
    """End-to-end: live catalog should show up in the /model picker."""

    def test_minimax_row_uses_live_catalog_models(self, monkeypatch):
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        open_fn, _rec = _make_urlopen(LIVE_CATALOG_INTL)

        with patch("urllib.request.urlopen", side_effect=open_fn):
            providers = list_authenticated_providers(
                current_provider="minimax", max_models=50,
            )

        row = next((p for p in providers if p["slug"] == "minimax"), None)
        assert row is not None, (
            "minimax should appear in the picker when MINIMAX_API_KEY is set"
        )
        assert "MiniMax-M3" in row["models"], (
            f"MiniMax-M3 (the user's case) should be in picker models, "
            f"got {row['models']!r}"
        )
        assert row["is_current"] is True

    def test_minimax_total_models_matches_live_catalog(self, monkeypatch):
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        open_fn, _rec = _make_urlopen(LIVE_CATALOG_INTL)

        with patch("urllib.request.urlopen", side_effect=open_fn):
            providers = list_authenticated_providers(
                current_provider="minimax", max_models=2,  # cap picker view
            )

        row = next(p for p in providers if p["slug"] == "minimax")
        # Picker shows the first max_models; total reflects the full catalog
        assert len(row["models"]) == 2
        assert row["total_models"] == len(LIVE_CATALOG_INTL["data"])

    def test_minimax_cn_row_in_picker(self, monkeypatch):
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-key-cn")
        open_fn, _rec = _make_urlopen(LIVE_CATALOG_CN)

        with patch("urllib.request.urlopen", side_effect=open_fn):
            providers = list_authenticated_providers(
                current_provider="minimax-cn", max_models=50,
            )

        row = next((p for p in providers if p["slug"] == "minimax-cn"), None)
        assert row is not None
        assert "abab7-chat" in row["models"]
        assert row["is_current"] is True

    def test_minimax_row_falls_back_to_static_when_no_key(self, monkeypatch):
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        # Use a real side-effect so the Nous remote-manifest fetch
        # (sibling of the minimax fetch) doesn't blow up.
        open_fn, rec = _make_urlopen(LIVE_CATALOG_INTL)

        with patch("urllib.request.urlopen", side_effect=open_fn):
            providers = list_authenticated_providers(
                current_provider="openai", max_models=50,
            )

        # No call to the minimax host happened
        assert not any("minimax" in url for url, _auth in rec)
        row = next((p for p in providers if p["slug"] == "minimax"), None)
        # Without creds, the row may or may not appear (depends on the
        # auth-check path).  When it does, the model list must be the
        # static fallback.
        if row is not None:
            assert "MiniMax-M2.7" in row["models"]
            assert "MiniMax-M3" not in row["models"], (
                "Static fallback should not include M3 unless it was "
                "explicitly added to _PROVIDER_MODELS['minimax']"
            )

    def test_minimax_row_uses_static_fallback_on_401(self, monkeypatch):
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setenv("MINIMAX_API_KEY", "bad-key")
        open_fn, _rec = _make_urlopen(
            LIVE_CATALOG_INTL, minimax_response="401",
        )

        with patch("urllib.request.urlopen", side_effect=open_fn):
            providers = list_authenticated_providers(
                current_provider="minimax", max_models=50,
            )

        row = next(p for p in providers if p["slug"] == "minimax")
        # Static+models.dev list is the visible model set — legacy ids
        # at minimum must be there.
        assert "MiniMax-M2.7" in row["models"]
        # A live-only id (not in static, not in models.dev) is absent —
        # confirms the live catalog was NOT used.
        assert "MiniMax-M99-future" not in row["models"]


# ── 4. minimax-oauth is a special case ─────────────────────────────────


class TestMiniMaxOAuthPicker:
    """The OAuth profile uses auth_type='oauth_external', which the generic
    live-fetch path in list_authenticated_providers skips (Section 2 only
    triggers for auth_type='api_key').  The picker therefore shows the
    static catalog for the OAuth provider.  This test pins that behavior
    so a future refactor doesn't accidentally break the fallback.
    """

    def test_minimax_oauth_uses_static_catalog(self):
        """The OAuth profile uses auth_type='oauth_external', which the
        generic live-fetch path in list_authenticated_providers skips
        (Section 2 only triggers for auth_type='api_key').  The picker
        therefore shows the static catalog for the OAuth provider.
        This test pins that behavior so a future refactor doesn't
        accidentally break the fallback.
        """
        # Real side-effect so the Nous remote-manifest fetch (sibling
        # of any minimax fetch) returns parseable bytes; we just need to
        # assert that no minimax URL was hit.
        open_fn, rec = _make_urlopen(LIVE_CATALOG_INTL)

        with patch("urllib.request.urlopen", side_effect=open_fn):
            from hermes_cli.models import _PROVIDER_MODELS
            from hermes_cli.model_switch import list_authenticated_providers

            providers = list_authenticated_providers(
                current_provider="minimax-oauth", max_models=50,
            )

        # No call to the minimax host happened (OAuth skips live fetch)
        assert not any("minimax" in url for url, _auth in rec), (
            f"minimax-oauth should not trigger a live /v1/models fetch "
            f"(auth_type='oauth_external'), but got: {rec!r}"
        )
        row = next(
            (p for p in providers if p["slug"] == "minimax-oauth"), None
        )
        # Row may not appear (no real OAuth store in tests), but if it
        # does it must show the static curated list.
        if row is not None:
            assert row["models"] == list(_PROVIDER_MODELS["minimax-oauth"])
