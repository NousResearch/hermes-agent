"""Bundled Token Plan contracts across discovery, auth, transport and auxiliaries."""

from contextlib import contextmanager
from pathlib import Path
import socket

import pytest

from agent.secret_scope import (
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from providers import get_provider_profile
from providers.base import ProviderProfile


PROFILE_IDS = (
    "alibaba-token-plan", "alibaba-token-plan-team",
    "alibaba-token-plan-cn", "alibaba-token-plan-cn-team",
)


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    def blocked(*args, **kwargs):
        raise AssertionError("Token Plan contract tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@contextmanager
def scoped(secrets):
    multiplex = set_multiplex_context(True)
    token = set_secret_scope(secrets)
    try:
        yield
    finally:
        reset_secret_scope(token)
        reset_multiplex_context(multiplex)


@pytest.mark.parametrize("provider_id", PROFILE_IDS)
def test_registry_auth_aliases_and_endpoint_override(provider_id):
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials

    profile = get_provider_profile(provider_id)
    assert profile is not None
    assert PROVIDER_REGISTRY[provider_id].inference_base_url == profile.base_url
    for alias in profile.aliases:
        assert get_provider_profile(alias) is profile
    key_names = [name for name in profile.env_vars if name.endswith("API_KEY")]
    url_name = next(name for name in profile.env_vars if name.endswith("BASE_URL"))
    for key_name in key_names:
        with scoped({key_name: "synthetic-key", url_name: "https://relay.invalid/v1/"}):
            credentials = resolve_api_key_provider_credentials(provider_id)
        assert credentials["api_key"] == "synthetic-key"
        assert credentials["source"] == key_name
        assert credentials["base_url"] == "https://relay.invalid/v1"
    with scoped({"DASHSCOPE_API_KEY": "unrelated-payg-key"}):
        assert resolve_api_key_provider_credentials(provider_id)["api_key"] == ""


@pytest.mark.parametrize("provider_id", PROFILE_IDS)
def test_tier_live_catalog_is_allowlisted_and_canonically_ordered(provider_id, monkeypatch):
    profile = get_provider_profile(provider_id)
    allowed = profile.fallback_models
    assert allowed
    monkeypatch.setattr(ProviderProfile, "fetch_models", lambda *a, **kw: [
        "unknown-preview", "wan2.7-image", *reversed(allowed), allowed[0].upper(),
    ])
    assert profile.fetch_models(api_key="synthetic") == list(allowed)
    monkeypatch.setattr(ProviderProfile, "fetch_models", lambda *a, **kw: [])
    assert profile.fetch_models(api_key="synthetic") == []
    monkeypatch.setattr(ProviderProfile, "fetch_models", lambda *a, **kw: None)
    assert profile.fetch_models(api_key="synthetic") is None


def wire_kwargs(provider_id, model, reasoning_config=None, **kwargs):
    from agent.transports.chat_completions import ChatCompletionsTransport

    return ChatCompletionsTransport().build_kwargs(
        model=model,
        messages=[{"role": "system", "content": "Stable instructions"},
                  {"role": "user", "content": "hello"}],
        provider_profile=get_provider_profile(provider_id),
        reasoning_config=reasoning_config,
        supports_reasoning=True,
        **kwargs,
    )


@pytest.mark.parametrize("effort,expected", [
    ("minimal", "low"), ("low", "low"), ("medium", "medium"),
    ("high", "xhigh"), ("xhigh", "xhigh"), ("max", "xhigh"),
])
def test_qwen_effort_mapping_preserves_replayed_thinking(effort, expected):
    body = wire_kwargs("alibaba-token-plan", "qwen3.8-max",
                       {"enabled": True, "effort": effort})["extra_body"]
    assert body["reasoning_effort"] == expected
    assert body["enable_thinking"] is True
    assert body["preserve_thinking"] is True


@pytest.mark.parametrize("model", ["MiniMax-M2.5", "glm-5.3"])
def test_always_thinking_models_never_receive_disable(model):
    body = wire_kwargs("alibaba-token-plan-team", model,
                       {"enabled": False}).get("extra_body", {})
    assert "enable_thinking" not in body
    if model.startswith("glm-"):
        assert body["clear_thinking"] is False


def test_hybrid_toggle_requires_boolean_and_retains_replay_controls():
    for config in (None, {"effort": "high"}, {"enabled": "false"}):
        body = wire_kwargs("alibaba-token-plan", "qwen3.7-plus", config)["extra_body"]
        assert "enable_thinking" not in body
        assert body["preserve_thinking"] is True
    body = wire_kwargs("alibaba-token-plan", "qwen3.7-plus",
                       {"enabled": False})["extra_body"]
    assert body["enable_thinking"] is False
    assert body["preserve_thinking"] is True
    unknown = wire_kwargs("alibaba-token-plan", "unknown-preview",
                          {"enabled": False, "effort": "high"})
    assert "enable_thinking" not in unknown.get("extra_body", {})


@pytest.mark.parametrize("provider_id", PROFILE_IDS)
def test_prompt_cache_key_opt_in_reaches_real_transport(provider_id):
    profile = get_provider_profile(provider_id)
    assert profile.supports_prompt_cache_key
    first = wire_kwargs(provider_id, profile.fallback_models[0], session_id="session-a")
    again = wire_kwargs(provider_id, profile.fallback_models[0], session_id="session-a")
    other = wire_kwargs(provider_id, profile.fallback_models[0], session_id="session-b")
    assert first["prompt_cache_key"] == again["prompt_cache_key"]
    assert first["prompt_cache_key"] != other["prompt_cache_key"]


def test_auxiliary_cache_separates_scoped_credentials_and_endpoints(monkeypatch):
    profile = get_provider_profile("alibaba-token-plan")
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_PERSONAL_API_KEY", "process-key")
    calls = []

    def fetch(self, *, api_key, base_url, **kwargs):
        assert api_key != "process-key"
        calls.append((api_key, base_url))
        if api_key == "scope-a" and base_url == profile.base_url:
            return ["qwen3.6-flash"]
        return ["deepseek-v4-flash-0731"]

    monkeypatch.setattr(type(profile), "fetch_models", fetch)
    key_name = "ALIBABA_TOKEN_PLAN_PERSONAL_API_KEY"
    for key, expected in (("scope-a", "qwen3.6-flash"),
                          ("scope-b", "deepseek-v4-flash-0731"),
                          ("scope-a", "qwen3.6-flash")):
        with scoped({key_name: key}):
            assert profile.resolve_aux_model() == expected
    assert len(calls) == 2
    with scoped({key_name: "scope-a", "ALIBABA_TOKEN_PLAN_BASE_URL": "https://relay.invalid/v1"}):
        assert profile.resolve_aux_model() == "deepseek-v4-flash-0731"
    assert len(calls) == 3
    with scoped({}):
        assert profile.resolve_aux_model() == ""
    multiplex = set_multiplex_context(True)
    try:
        assert profile.resolve_aux_model() == ""
    finally:
        reset_multiplex_context(multiplex)
    assert len(calls) == 3


@pytest.mark.parametrize("provider_id", PROFILE_IDS)
def test_auxiliary_core_default_and_dynamic_fallback(provider_id, monkeypatch):
    from agent import auxiliary_client

    profile = get_provider_profile(provider_id)
    assert profile.default_aux_model in profile.fallback_models
    assert auxiliary_client._get_aux_model_for_provider(provider_id) == profile.default_aux_model
    monkeypatch.setattr(auxiliary_client, "_fast_model_from_catalog", lambda _: "")
    monkeypatch.setattr(type(profile), "resolve_aux_model", lambda self: "synthetic-fast-model")
    assert auxiliary_client._get_aux_model_for_provider(provider_id, prefer_fast=True) == "synthetic-fast-model"


def test_auxiliary_vision_choice_does_not_reuse_text_cache(monkeypatch):
    profile = get_provider_profile("alibaba-token-plan-team")
    calls = []

    def fetch(self, **kwargs):
        calls.append(kwargs)
        return ["deepseek-v4-flash-0731", "qwen3.7-plus"]

    monkeypatch.setattr(type(profile), "fetch_models", fetch)
    with scoped({"ALIBABA_TOKEN_PLAN_TEAM_API_KEY": "vision-synthetic"}):
        assert profile.resolve_aux_model() == "deepseek-v4-flash-0731"
        assert profile.resolve_aux_model(vision=True) == "qwen3.7-plus"
        assert profile.resolve_aux_model(vision=True) == "qwen3.7-plus"
    assert len(calls) == 2


def test_auxiliary_discovery_failure_keeps_default_fallback(monkeypatch):
    profile = get_provider_profile("alibaba-token-plan-cn-team")

    def unavailable(self, **kwargs):
        raise TimeoutError("synthetic catalog timeout")

    monkeypatch.setattr(type(profile), "fetch_models", unavailable)
    with scoped({"ALIBABA_TOKEN_PLAN_CN_TEAM_API_KEY": "timeout-synthetic"}):
        assert profile.resolve_aux_model() == ""


def test_cn_personal_legacy_key_fallback_yields_to_dedicated_aliases():
    from hermes_cli.auth import resolve_api_key_provider_credentials

    provider_id = "alibaba-token-plan-cn"
    profile = get_provider_profile(provider_id)
    generic = {"ALIBABA_TOKEN_PLAN_API_KEY": "legacy-generic"}
    with scoped(generic):
        assert resolve_api_key_provider_credentials(provider_id)["api_key"] == "legacy-generic"
        assert profile._resolve_credentials()[0] == "legacy-generic"
    for dedicated in ("ALIBABA_TOKEN_PLAN_CN_API_KEY", "ALIBABA_TOKEN_PLAN_CN_PERSONAL_API_KEY"):
        with scoped({**generic, dedicated: "dedicated-cn"}):
            assert resolve_api_key_provider_credentials(provider_id)["api_key"] == "dedicated-cn"
            assert profile._resolve_credentials()[0] == "dedicated-cn"
    with scoped(generic):
        assert resolve_api_key_provider_credentials("alibaba-token-plan-cn-team")["api_key"] == ""


@pytest.mark.parametrize("provider_id,catalog_id", [
    ("alibaba-token-plan-team", "alibaba-token-plan"),
    ("alibaba-token-plan-cn-team", "alibaba-token-plan-cn"),
])
def test_team_context_metadata_uses_regional_token_plan_catalog(provider_id, catalog_id, monkeypatch):
    from agent import models_dev

    registry = {
        "alibaba-token-plan": {"models": {"synthetic-model": {"limit": {"context": 54321}}}},
        "alibaba-token-plan-cn": {"models": {"synthetic-model": {"limit": {"context": 65432}}}},
    }
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda **kw: registry)
    expected = registry[catalog_id]["models"]["synthetic-model"]["limit"]["context"]
    assert models_dev.lookup_models_dev_context(provider_id, "synthetic-model") == expected


@pytest.mark.parametrize("host,provider_id", [
    ("token-plan.ap-southeast-1.maas.aliyuncs.com", "alibaba-token-plan"),
    ("token-plan.cn-beijing.maas.aliyuncs.com", "alibaba-token-plan-cn"),
    ("token-plan.maas.qwencloudapi.com", "alibaba-token-plan"),
    ("token-plan.maas.qianwenaiapi.com", "alibaba-token-plan-cn"),
])
def test_old_and_current_endpoint_hosts_keep_region_metadata(host, provider_id):
    from agent.model_metadata import _infer_provider_from_url

    assert _infer_provider_from_url(f"https://{host}/compatible-mode/v1") == provider_id


def test_auxiliary_discovery_honours_selected_model_route_only(tmp_path, monkeypatch):
    profile = get_provider_profile("alibaba-token-plan")
    config_path = tmp_path / "hermes" / "config.yaml"
    calls = []

    def fetch(self, *, api_key, base_url, **kwargs):
        calls.append((api_key, base_url))
        return ["qwen3.6-flash"]

    monkeypatch.setattr(type(profile), "fetch_models", fetch)
    secrets = {
        "ALIBABA_TOKEN_PLAN_API_KEY": "canonical-route-key",
        "SYNTHETIC_RELAY_KEY": "configured-route-key",
    }
    for selected in ("alibaba-token-plan-cn", profile.name):
        config_path.write_text(
            f"model:\n  provider: {selected}\n  key_env: SYNTHETIC_RELAY_KEY\n"
            "  base_url: https://configured-relay.invalid/v1\n"
        )
        with scoped(secrets):
            assert profile.resolve_aux_model() == "qwen3.6-flash"
        expected = (
            ("configured-route-key", "https://configured-relay.invalid/v1")
            if selected == profile.name else ("canonical-route-key", profile.base_url)
        )
        assert calls[-1] == expected
    assert len(calls) == 2
