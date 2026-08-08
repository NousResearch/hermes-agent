"""Unit tests for hermes_cli/session_endpoint.py (issue #77831)."""

import pytest

from hermes_cli.session_endpoint import (
    billing_provider_id,
    normalize_endpoint_url,
    resolve_endpoint_provider_id,
)


# ── normalize_endpoint_url ────────────────────────────────────────────────


class TestNormalizeEndpointUrl:
    def test_accepts_http_url_with_path(self):
        assert (
            normalize_endpoint_url("http://203.0.113.7:8355/v1")
            == "http://203.0.113.7:8355/v1"
        )

    def test_strips_single_trailing_slash(self):
        assert (
            normalize_endpoint_url("https://model-host.local:8355/v1/")
            == "https://model-host.local:8355/v1"
        )
        assert (
            normalize_endpoint_url("http://localhost:11434") == "http://localhost:11434"
        )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_endpoint_url("")
        with pytest.raises(ValueError, match="empty"):
            normalize_endpoint_url("   ")

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="scheme must be http or https"):
            normalize_endpoint_url("ftp://host/v1")
        with pytest.raises(ValueError, match="scheme must be http or https"):
            normalize_endpoint_url("model-host.local:8355")

    def test_rejects_missing_host(self):
        with pytest.raises(ValueError, match="missing host"):
            normalize_endpoint_url("http://")
        with pytest.raises(ValueError, match="missing host"):
            normalize_endpoint_url("http:///v1")

    def test_rejects_out_of_range_port(self):
        with pytest.raises(ValueError, match="invalid endpoint URL"):
            normalize_endpoint_url("http://host:99999/v1")


# ── resolve_endpoint_provider_id ──────────────────────────────────────────


def _no_entry(provider_id):
    return None


def _entry(provider_id):
    return {"name": provider_id.split(":", 1)[1], "base_url": "http://x/v1"}


@pytest.fixture
def patch_rt(monkeypatch):
    """Patch the lazy-imported runtime_provider lookups (module of origin)."""
    import hermes_cli.runtime_provider as rt_mod

    monkeypatch.setattr(rt_mod, "find_custom_provider_identity", lambda url: None)
    monkeypatch.setattr(rt_mod, "_get_named_custom_provider", _no_entry)
    return rt_mod


class TestRequestedProvider:
    def test_bare_custom(self, patch_rt):
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1", requested_provider="custom"
        ) == ("custom", "requested")

    def test_builtin(self, patch_rt):
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1", requested_provider="openrouter"
        ) == ("openrouter", "requested")

    def test_configured_custom_entry(self, patch_rt, monkeypatch):
        import hermes_cli.runtime_provider as rt_mod

        monkeypatch.setattr(rt_mod, "_get_named_custom_provider", _entry)
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1", requested_provider="custom:local"
        ) == ("custom:local", "requested")

    def test_unknown_custom_entry_rejected(self, patch_rt):
        with pytest.raises(ValueError, match="Unknown provider 'custom:nope'"):
            resolve_endpoint_provider_id(
                "http://203.0.113.7:8355/v1", requested_provider="custom:nope"
            )

    def test_auto_rejected(self, patch_rt):
        with pytest.raises(ValueError, match="not a routable provider id"):
            resolve_endpoint_provider_id(
                "http://203.0.113.7:8355/v1", requested_provider="auto"
            )

    def test_unknown_builtin_rejected(self, patch_rt, monkeypatch):
        import hermes_cli.auth as auth_mod

        def _boom(_):
            raise auth_mod.AuthError("Unknown provider 'nope'.")

        monkeypatch.setattr(auth_mod, "resolve_provider", _boom)
        with pytest.raises(ValueError, match="Unknown provider 'nope'"):
            resolve_endpoint_provider_id(
                "http://203.0.113.7:8355/v1", requested_provider="nope"
            )


class TestAutoDerive:
    def test_config_entry_owns_url(self, patch_rt, monkeypatch):
        import hermes_cli.runtime_provider as rt_mod

        monkeypatch.setattr(
            rt_mod, "find_custom_provider_identity", lambda url: "custom:local"
        )
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1",
            existing_provider="custom:model-host.local:8355",
        ) == ("custom:local", "config-entry")

    def test_keeps_bare_custom(self, patch_rt):
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1", existing_provider="custom"
        ) == ("custom", "existing")

    def test_keeps_builtin(self, patch_rt, monkeypatch):
        import hermes_cli.auth as auth_mod

        monkeypatch.setattr(auth_mod, "resolve_provider", lambda pid: "anthropic")
        assert resolve_endpoint_provider_id(
            "https://api.anthropic.com",
            existing_provider="anthropic",
        ) == ("anthropic", "existing")

    def test_keeps_configured_custom_slug(self, patch_rt, monkeypatch):
        import hermes_cli.runtime_provider as rt_mod

        monkeypatch.setattr(rt_mod, "_get_named_custom_provider", _entry)
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1",
            existing_provider="custom:local",
        ) == ("custom:local", "existing")

    def test_falls_back_to_bare_custom_for_orphan_slug(self, patch_rt):
        # The issue's footgun: a custom:<endpoint> slug with no config entry
        # must NOT be persisted (the runtime refuses it with
        # "Unknown provider 'custom:<endpoint>'").
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1",
            existing_provider="custom:model-host.local:8355",
        ) == ("custom", "custom-fallback")

    def test_falls_back_with_no_existing_provider(self, patch_rt):
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1", existing_provider=None
        ) == ("custom", "custom-fallback")

    def test_find_error_is_tolerated(self, patch_rt, monkeypatch):
        import hermes_cli.runtime_provider as rt_mod

        monkeypatch.setattr(
            rt_mod,
            "find_custom_provider_identity",
            lambda url: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert resolve_endpoint_provider_id(
            "http://203.0.113.7:8355/v1",
            existing_provider="custom:model-host.local:8355",
        ) == ("custom", "custom-fallback")


# ── billing_provider_id ───────────────────────────────────────────────────


class TestBillingProviderId:
    def test_bare_custom(self):
        assert billing_provider_id("custom") == "custom"

    def test_custom_slug_billed_under_bare_class(self):
        assert billing_provider_id("custom:local") == "custom"
        assert billing_provider_id("custom:model-host.local:8355") == "custom"

    def test_builtin_kept(self):
        assert billing_provider_id("anthropic") == "anthropic"
        assert billing_provider_id("openrouter") == "openrouter"
