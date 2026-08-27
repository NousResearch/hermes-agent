"""Every Nous model list is narrowed to the org's policy before it is shown.

Four surfaces build a Nous list from the curated manifest unioned with the
Portal's ``recommended-models`` endpoint. Neither source is authenticated, so
without this filter an org's hidden model is offered to the user and then
refused at request time with ``model_blocked_by_org_policy``.
"""

from __future__ import annotations

import argparse

import pytest

import hermes_cli.models as models_mod

CURATED = ["vendor/allowed", "vendor/blocked"]
ALLOWED = {"vendor/allowed"}


@pytest.fixture
def policy(monkeypatch):
    """An org whose policy admits only ``vendor/allowed``."""
    monkeypatch.setattr(models_mod, "nous_policy_allowed_ids", lambda **_k: ALLOWED)
    return ALLOWED


@pytest.fixture
def no_policy(monkeypatch):
    """An unrestricted org — lists must come through untouched."""
    monkeypatch.setattr(models_mod, "nous_policy_allowed_ids", lambda **_k: None)


class TestLoginNous:
    """``_login_nous`` — the model picked at login is the model then used."""

    def _run(self, monkeypatch, tmp_path):
        import hermes_cli.auth as auth_mod
        import hermes_cli.nous_subscription as ns

        seen: dict = {}
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            auth_mod,
            "_nous_device_code_login",
            lambda **_k: {
                "access_token": "tok",
                "agent_key": "key",
                "inference_base_url": "https://inference.example.com",
                "portal_base_url": "https://portal.example.com",
                "refresh_token": "r",
                "token_expires_at": 9999999999,
            },
        )
        monkeypatch.setattr(models_mod, "get_curated_nous_model_ids", lambda: list(CURATED))
        monkeypatch.setattr(models_mod, "get_pricing_for_provider", lambda _p: {})
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda **_k: None)
        monkeypatch.setattr(
            models_mod,
            "union_with_portal_paid_recommendations",
            lambda ids, pricing, _portal: (list(ids), pricing),
        )
        monkeypatch.setattr(ns, "prompt_enable_tool_gateway", lambda _c: None)

        def _capture(model_ids, **kwargs):
            seen["model_ids"] = list(model_ids)
            return None

        monkeypatch.setattr(auth_mod, "_prompt_model_selection", _capture)

        args = argparse.Namespace(
            portal_url=None, inference_url=None, client_id=None, scope=None,
            no_browser=True, timeout=15.0, ca_bundle=None, insecure=False,
        )
        auth_mod._login_nous(args, auth_mod.PROVIDER_REGISTRY["nous"])
        return seen

    def test_hidden_model_is_not_offered(self, monkeypatch, tmp_path, policy):
        assert self._run(monkeypatch, tmp_path).get("model_ids") == ["vendor/allowed"]

    def test_unrestricted_org_sees_the_full_curated_list(
        self, monkeypatch, tmp_path, no_policy
    ):
        assert self._run(monkeypatch, tmp_path).get("model_ids") == CURATED


class TestModelSwitchPicker:
    """The ``/model`` picker's nous branch (``list_authenticated_providers``)."""

    def _rows(self, monkeypatch):
        import hermes_cli.auth as auth_mod
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(
            auth_mod,
            "_load_auth_store",
            lambda *a, **k: {"providers": {"nous": {"access_token": "tok"}}},
        )
        monkeypatch.setattr(models_mod, "get_curated_nous_model_ids", lambda: list(CURATED))
        monkeypatch.setattr(models_mod, "get_pricing_for_provider", lambda _p: {})
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda **_k: None)
        monkeypatch.setattr(
            models_mod,
            "union_with_portal_paid_recommendations",
            lambda ids, pricing, _portal: (list(ids), pricing),
        )
        rows = ms.list_authenticated_providers(max_models=10)
        return next((r for r in rows if r["slug"] == "nous"), None)

    def test_hidden_model_is_filtered(self, monkeypatch, policy):
        row = self._rows(monkeypatch)
        assert row is not None, "nous row should be listed"
        assert "vendor/blocked" not in row["models"]
        assert "vendor/allowed" in row["models"]

    def test_unrestricted_org_keeps_both(self, monkeypatch, no_policy):
        row = self._rows(monkeypatch)
        assert row is not None
        assert set(CURATED) <= set(row["models"])

    def test_filter_survives_a_failed_recommendation_fetch(self, monkeypatch, policy):
        """The filter sits outside the try that wraps the Portal union, so a
        Portal outage still yields a policy-filtered curated list."""

        def _boom(_p):
            raise RuntimeError("portal down")

        monkeypatch.setattr(models_mod, "get_pricing_for_provider", _boom)
        row = self._rows(monkeypatch)
        assert row is not None
        assert "vendor/blocked" not in row["models"]


class TestRecommendedDefaultEndpoint:
    """``GET /api/model/recommended-default`` picks a model the user never sees
    chosen, so an unreachable one there is worse than in a picker."""

    def _call(self, monkeypatch):
        import hermes_cli.auth as auth_mod
        from hermes_cli.web_server import get_recommended_default_model

        # Blocked first, so an unfiltered list would make it the silent
        # default — otherwise this passes whether or not the filter runs.
        monkeypatch.setattr(
            models_mod, "get_curated_nous_model_ids",
            lambda: ["vendor/blocked", "vendor/allowed"],
        )
        monkeypatch.setattr(models_mod, "get_pricing_for_provider", lambda _p: {})
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda **_k: None)
        monkeypatch.setattr(
            models_mod,
            "union_with_portal_paid_recommendations",
            lambda ids, pricing, _portal: (list(ids), pricing),
        )
        monkeypatch.setattr(auth_mod, "get_provider_auth_state", lambda _p: {})
        return get_recommended_default_model(provider="nous")

    def test_hidden_model_is_never_the_silent_default(self, monkeypatch, policy):
        assert self._call(monkeypatch)["model"] == "vendor/allowed"

    def test_unrestricted_org_is_unaffected(self, monkeypatch, no_policy):
        assert self._call(monkeypatch)["model"] == "vendor/blocked"


class TestAuxiliaryFastModel:
    """``_fast_model_from_catalog`` treats the catalog's keys as a source of
    ids, so an anonymous read there can select a model the gateway refuses."""

    def _pick(self, monkeypatch, *, catalog):
        import agent.auxiliary_client as aux

        seen: dict = {}

        def _fake_fetch(*, api_key=None, base_url="", timeout=8.0, **_k):
            seen["api_key"] = api_key
            return {mid: {} for mid in catalog}

        monkeypatch.setattr(
            models_mod, "_resolve_nous_pricing_credentials",
            lambda: ("sk-nous", "https://inference.example.com"),
        )
        monkeypatch.setattr(models_mod, "fetch_models_with_pricing", _fake_fetch)
        picked = aux._fast_model_from_catalog("nous")
        return picked, seen

    def test_reads_the_catalog_with_nous_oauth_credentials(self, monkeypatch, no_policy):
        """The api-key resolver raises for OAuth providers; without a fallback
        the read goes out anonymous and returns the unfiltered catalog."""
        _, seen = self._pick(monkeypatch, catalog=["vendor/haiku-fast"])
        assert seen["api_key"] == "sk-nous"

    def test_hidden_model_is_not_selected(self, monkeypatch, policy):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(
            models_mod, "nous_policy_allowed_ids", lambda **_k: {"vendor/allowed"}
        )
        monkeypatch.setattr(aux, "_FAST_MODEL_FAMILIES", ("vendor/",))
        monkeypatch.setattr(aux, "_FAST_MODEL_EXCLUDE", ())
        picked, _ = self._pick(
            monkeypatch, catalog=["vendor/blocked", "vendor/allowed"]
        )
        assert picked == "vendor/allowed"
