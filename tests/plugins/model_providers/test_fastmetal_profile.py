"""Unit tests for the FastMetal provider profile.

FastMetal is an OpenAI-compatible gateway with two quirks the profile owns: thinking is a
request-body ``reasoning`` object that only routes with thinking controls accept (others 404 on
the field, so it must be omitted, not clamped), and the prepaid balance lives at ``/key/info``
on the host root. Everything else rides the generic chat_completions transport.
"""

from __future__ import annotations

import io
import json
import urllib.request

import pytest

import providers


@pytest.fixture
def fastmetal_profile():
    profile = providers.get_provider_profile("fastmetal")
    assert profile is not None, "fastmetal provider profile must be registered"
    return profile


class TestFastMetalProfile:
    def test_identity_and_endpoint(self, fastmetal_profile):
        assert fastmetal_profile.name == "fastmetal"
        assert fastmetal_profile.api_mode == "chat_completions"
        assert fastmetal_profile.auth_type == "api_key"
        assert fastmetal_profile.base_url == "https://api.fastmetal.ai/v1"
        assert fastmetal_profile.get_hostname() == "api.fastmetal.ai"

    @pytest.mark.parametrize("alias", ["fast-metal", "fastmetal-ai"])
    def test_aliases_resolve_to_the_same_profile(self, fastmetal_profile, alias):
        assert providers.get_provider_profile(alias) is fastmetal_profile

    def test_env_vars_key_then_base_url_override(self, fastmetal_profile):
        assert fastmetal_profile.env_vars == ("FASTMETAL_API_KEY", "FASTMETAL_BASE_URL")

    def test_curated_ids_are_bare_gateway_ids(self, fastmetal_profile):
        """FastMetal rejects vendor-prefixed ids; the curated list must be what /v1/models returns."""
        assert fastmetal_profile.fallback_models
        assert all("/" not in model for model in fastmetal_profile.fallback_models)
        assert fastmetal_profile.default_aux_model in fastmetal_profile.fallback_models

    def test_reasoning_declarations_agree_with_the_wire_gate(self, fastmetal_profile):
        """The picker badge (model_capabilities) and the request gate (route_exposes_reasoning) are two
        views of one FastMetal table; a model must not advertise reasoning it never receives."""
        from plugins.model_providers.fastmetal import route_exposes_reasoning

        for model in fastmetal_profile.fallback_models:
            declared = fastmetal_profile.model_capabilities.get(model, {}).get("supports_reasoning")
            assert declared is not None, f"{model} must declare supports_reasoning"
            assert declared == route_exposes_reasoning(model), model


class TestFastMetalReasoning:
    REASONING_ROUTE = "kimi-k3"
    PLAIN_ROUTE = "llama-3.1-8b-instruct"

    def test_unset_leaves_the_route_default(self, fastmetal_profile):
        assert fastmetal_profile.build_api_kwargs_extras(reasoning_config=None, model=self.REASONING_ROUTE) == ({}, {})

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
    def test_explicit_effort_goes_in_the_reasoning_object(self, fastmetal_profile, effort):
        extra_body, top_level = fastmetal_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model=self.REASONING_ROUTE,
        )
        assert extra_body == {"reasoning": {"effort": effort}}
        assert top_level == {}, "top-level reasoning_effort is ignored by FastMetal routes"

    @pytest.mark.parametrize("config", [{"enabled": False, "effort": "high"}, {"enabled": True, "effort": "none"}])
    def test_disable_is_reasoning_enabled_false(self, fastmetal_profile, config):
        extra_body, _ = fastmetal_profile.build_api_kwargs_extras(reasoning_config=config, model=self.REASONING_ROUTE)
        assert extra_body == {"reasoning": {"enabled": False}}

    @pytest.mark.parametrize("config", [{"enabled": True, "effort": "low"}, {"enabled": False, "effort": "high"}])
    def test_non_reasoning_route_never_receives_the_field(self, fastmetal_profile, config):
        """Live 2026-09-22: llama-3.1-8b-instruct answers any ``reasoning`` field — the disable form
        included — with HTTP 404 "No endpoints found that can handle the requested parameters"."""
        assert fastmetal_profile.build_api_kwargs_extras(reasoning_config=config, model=self.PLAIN_ROUTE) == ({}, {})

    def test_copied_aggregator_prefix_still_resolves_the_route(self, fastmetal_profile):
        extra_body, _ = fastmetal_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"}, model=f"fastmetal/{self.REASONING_ROUTE}",
        )
        assert extra_body == {"reasoning": {"effort": "low"}}

    def test_transport_emits_extra_body_reasoning_only(self, fastmetal_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model=self.REASONING_ROUTE, messages=[{"role": "user", "content": "ping"}], tools=None,
            provider_profile=fastmetal_profile, reasoning_config={"enabled": True, "effort": "low"},
            base_url="https://api.fastmetal.ai/v1", provider_name="fastmetal",
        )
        assert kwargs["extra_body"]["reasoning"] == {"effort": "low"}
        assert "reasoning_effort" not in kwargs


class TestFastMetalAccountUsage:
    @pytest.fixture
    def key_info(self, monkeypatch):
        """Route the hook's HTTP through a fake ``/key/info`` and record the URL it asked for."""
        calls: dict = {"payload": {"key": "sk-redacted", "info": {"spend": 250.5, "max_budget": 2000.0}}}

        def fake_open(request: urllib.request.Request, *, timeout: float):
            calls["url"] = request.full_url
            calls["auth"] = request.get_header("Authorization")
            return io.BytesIO(json.dumps(calls["payload"]).encode())

        monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", fake_open)
        return calls

    @pytest.fixture
    def runtime(self, monkeypatch):
        resolved: dict = {"api_key": "sk-test-1234567890", "base_url": "https://api.fastmetal.ai/v1"}
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_: resolved)
        return resolved

    def test_balance_window_from_spend_and_budget(self, fastmetal_profile, key_info, runtime):
        snapshot = fastmetal_profile.fetch_account_usage()
        assert snapshot is not None and snapshot.available
        assert key_info["url"] == "https://api.fastmetal.ai/key/info", "/key/info sits at the host root, not under /v1"
        assert key_info["auth"] == f"Bearer {runtime['api_key']}"
        (window,) = snapshot.windows
        assert window.used_percent == pytest.approx(250.5 / 2000.0 * 100)
        assert "1,749.50" in window.detail and "2,000.00" in window.detail

    def test_custom_base_url_keeps_the_same_origin(self, fastmetal_profile, key_info, runtime):
        runtime["base_url"] = "https://proxy.example.test/v1"
        fastmetal_profile.fetch_account_usage()
        assert key_info["url"] == "https://proxy.example.test/key/info"

    def test_uncapped_key_reports_spend_only(self, fastmetal_profile, key_info, runtime):
        key_info["payload"]["info"] = {"spend": 12.0, "max_budget": None}
        snapshot = fastmetal_profile.fetch_account_usage()
        assert snapshot.windows == ()
        assert any("12.00" in line for line in snapshot.details)

    def test_no_credential_means_no_request(self, fastmetal_profile, key_info, runtime):
        runtime["api_key"] = ""
        assert fastmetal_profile.fetch_account_usage() is None
        assert "url" not in key_info
