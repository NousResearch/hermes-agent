"""Runtime provider resolution: model.openai_runtime overlay, the ``openai`` alias, and launch keys."""

from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp


# ── model.openai_runtime: codex_app_server on every ladder rung (#115169) ─────────────────

_CODEX_STORE_CREDS = {"base_url": "https://chatgpt.com/backend-api/codex", "api_key": "tok",
                      "source": "hermes-auth-store", "last_refresh": 1}


def _codex_rung(monkeypatch, rung: str) -> dict:
    """Isolate one openai-codex ladder rung; returns the kwargs for resolve_runtime_provider."""
    monkeypatch.setattr(rp, "resolve_codex_runtime_credentials", lambda: dict(_CODEX_STORE_CREDS))
    if rung == "pool":
        entry = SimpleNamespace(api_key="tok", runtime_api_key="tok", base_url="", source="pool")
        monkeypatch.setattr(rp, "load_pool", lambda _p: SimpleNamespace(
            has_credentials=lambda: True, select=lambda model=None: entry))
        monkeypatch.setattr(rp, "credential_pool_matches_provider", lambda *a, **k: True)
        return {}
    monkeypatch.setattr(rp, "load_pool", lambda _p: SimpleNamespace(has_credentials=lambda: False))
    return {"explicit_api_key": "sk-explicit"} if rung == "explicit" else {}


@pytest.mark.parametrize("rung", ["pool", "oauth", "explicit"])
def test_openai_runtime_codex_app_server_applies_on_every_rung(monkeypatch, rung):
    """#115169: the opt-in was applied only inside the credential-pool rung, so the OAuth-store
    and explicit --api-key/--base-url rungs silently resolved codex_responses."""
    kwargs = _codex_rung(monkeypatch, rung)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "openai-codex", "default": "gpt-5.5", "openai_runtime": "codex_app_server"})

    resolved = rp.resolve_runtime_provider(requested="openai-codex", **kwargs)

    assert resolved["provider"] == "openai-codex"
    assert resolved["api_mode"] == "codex_app_server"


@pytest.mark.parametrize("rung", ["pool", "oauth", "explicit"])
@pytest.mark.parametrize("openai_runtime", [None, "auto"])
def test_openai_runtime_unset_keeps_wire_api_mode(monkeypatch, rung, openai_runtime):
    kwargs = _codex_rung(monkeypatch, rung)
    model_cfg = {"provider": "openai-codex", "default": "gpt-5.5"}
    if openai_runtime is not None:
        model_cfg["openai_runtime"] = openai_runtime
    monkeypatch.setattr(rp, "_get_model_config", lambda: model_cfg)

    assert rp.resolve_runtime_provider(requested="openai-codex", **kwargs)["api_mode"] == "codex_responses"


def test_openai_runtime_codex_app_server_survives_the_openai_to_custom_alias_expansion(monkeypatch):
    """``provider: openai`` expands to the anonymous ``custom`` runtime (#116055) before the overlay runs;
    the overlay must judge the name the user configured, or the documented ``openai`` opt-in is a silent no-op."""
    monkeypatch.setattr(rp, "load_pool", lambda _p: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "openai", "default": "gpt-5.5-codex", "openai_runtime": "codex_app_server"})

    resolved = rp.resolve_runtime_provider(requested="openai", explicit_api_key="sk-explicit")

    assert resolved["provider"] == "custom"  # the alias expansion itself is unchanged
    assert resolved["api_mode"] == "codex_app_server"


# ── #116055: ``provider: openai`` means the same thing on both auxiliary paths ──────────────────

def test_openai_alias_resolves_identically_on_runtime_and_aux_client_paths(monkeypatch):
    """background_review/curator/MoA (resolve_runtime_provider) and compression/vision/title
    (_resolve_task_provider_model) must land on the same endpoint for the same aux block."""
    from agent import auxiliary_client as aux
    block = {"provider": "openai", "model": "review-model", "base_url": "https://gateway.example/v1", "api_key": "gw-key"}
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: block if task == "background_review" else {})
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "custom:mylocal", "default": "local-main"})

    aux_provider, _aux_model, aux_base, aux_key, _ = aux._resolve_task_provider_model("background_review")
    runtime = rp.resolve_runtime_provider(requested=block["provider"], target_model=block["model"],
                                          explicit_api_key=block["api_key"], explicit_base_url=block["base_url"])

    assert (aux_provider, aux_base, aux_key) == ("custom", "https://gateway.example/v1", "gw-key")
    assert (runtime["provider"], runtime["base_url"], runtime["api_key"]) == (aux_provider, aux_base, aux_key)


def test_openai_alias_without_base_url_pairs_openai_key_with_openai_base_url(monkeypatch):
    """No aux base_url: the alias lands on OPENAI_BASE_URL (the proxy the key was issued for) and the
    runtime path pairs OPENAI_API_KEY with it instead of sending a placeholder key to the proxy."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm-proxy.corp.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proxy-issued")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "custom:mylocal", "default": "local-main"})

    runtime = rp.resolve_runtime_provider(requested="openai", target_model="gpt-x")

    assert (runtime["provider"], runtime["base_url"], runtime["api_key"]) == ("custom", "https://llm-proxy.corp.example/v1", "sk-proxy-issued")


def test_bare_custom_explicit_launch_key_wins_over_custom_pool(monkeypatch):
    """R2-M1: on the bare-custom terminal rung a launch ``--api-key`` must not be replaced by a
    pooled credential for the same URL (named-custom and direct-alias already honour it)."""
    from hermes_cli.runtime_provider_backends import _resolve_openrouter_runtime

    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "custom"})
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: {
        "provider": "custom", "base_url": "http://127.0.0.1:9/v1", "api_key": "sk-pooled", "source": "pool"})
    resolved = _resolve_openrouter_runtime(requested_provider="custom", explicit_api_key="sk-launch-explicit",
                                           explicit_base_url="http://127.0.0.1:9/v1")
    assert resolved["api_key"] == "sk-launch-explicit"
