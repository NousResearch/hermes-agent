"""Auxiliary calls to OpenRouter must carry the operator's ``provider_routing`` preferences, exactly
like main turns. They used to ship no ``provider`` object at all, leaving endpoint choice to
OpenRouter's default load balancing - which served from providers the operator's
``provider_routing.only`` had excluded (DeepSeek served by wafer while only InferenceNet/DeepInfra/
Cloudflare were chosen), at a higher unit price than the chosen provider (#89384). The injection is
gated to OpenRouter routes (named provider or an openrouter.ai base URL - an OpenRouter wallet is a
wallet whatever the entry is named), reads config at call time (edits take effect without a restart),
and fails open to no routing when config has none. ``provider`` is OpenRouter's request contract, so
a custom-named entry at openrouter.ai still gets it even though CustomProfile ignores the kwarg."""
from unittest.mock import patch

import agent.auxiliary_client as ac

_ROUTING = {"provider_routing": {
    "only": ["InferenceNet", "DeepInfra", "Cloudflare"],
    "ignore": ["deepseek"],
    "data_collection": "deny",
}}
_MODEL = "deepseek/deepseek-v4.1-flash-1017"


def _project_body(provider_norm: str, base_url: str, cfg: dict | None = None) -> dict:
    cfg = _ROUTING if cfg is None else cfg
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        return ac._project_provider_profile(
            provider_norm, provider_norm, _MODEL, base_url, None, frozenset()).body


def test_openrouter_named_route_ships_the_operators_provider_routing():
    body = _project_body("openrouter", "https://openrouter.ai/api/v1")
    assert body.get("provider") == {
        "only": ["InferenceNet", "DeepInfra", "Cloudflare"],
        "ignore": ["deepseek"],
        "data_collection": "deny",
    }


def test_openrouter_host_under_a_custom_entry_name_is_still_routed():
    # An openrouter.ai base_url is an OpenRouter wallet whatever the provider key is called;
    # CustomProfile ignores provider_preferences, so the request-contract key must still land.
    body = _project_body("custom", "https://openrouter.ai/api/v1")
    assert (body.get("provider") or {}).get("only") == ["InferenceNet", "DeepInfra", "Cloudflare"]


def test_local_custom_endpoint_ships_no_provider_object():
    body = _project_body("custom", "http://127.0.0.1:8000/v1")
    assert "provider" not in body


def test_config_without_provider_routing_fails_open_to_old_behavior():
    body = _project_body("openrouter", "https://openrouter.ai/api/v1", cfg={})
    assert "provider" not in body


def test_per_model_overlay_wins_over_global_routing():
    cfg = {"provider_routing": {
        "only": ["InferenceNet"],
        "models": {_MODEL: {"only": ["DeepInfra"]}},
    }}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        prefs = ac._aux_provider_preferences(model=_MODEL)
    assert prefs["only"] == ["DeepInfra"]


def test_resolution_failure_degrades_to_no_routing():
    with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("config unreadable")):
        assert ac._aux_provider_preferences(model=_MODEL) == {}


def _build_kwargs(extra_body: dict | None) -> dict:
    with patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING):
        return ac._build_call_kwargs(
            "openrouter", _MODEL, [{"role": "user", "content": "hi"}],
            extra_body=extra_body, base_url="https://openrouter.ai/api/v1")


def test_configured_routing_survives_the_full_builder():
    eb = _build_kwargs(None)["extra_body"]
    assert eb.get("provider", {}).get("only") == ["InferenceNet", "DeepInfra", "Cloudflare"]


def test_explicit_caller_provider_wins_over_configured_routing():
    # auxiliary.<task>.extra_body.provider is a per-task override: the profile projection
    # supplies route defaults only (#89425's "provider" not in extra_body contract).
    override = {"only": ["ChosenOnly"]}
    eb = _build_kwargs({"provider": override})["extra_body"]
    assert eb["provider"] == override
