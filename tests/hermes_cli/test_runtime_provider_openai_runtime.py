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


# ── #125005: ``model.provider: openai`` honours model.base_url and model.key_env ──────────────────

def _write_openai_proxy_config(tmp_path, monkeypatch, extra: str = ""):
    from hermes_cli import config as _cfg

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        "model:\n  provider: openai\n  default: gpt-x\n"
        "  base_url: https://proxy.example/v1\n  key_env: PROXY_KEY\n" + extra)
    _cfg._LOAD_CONFIG_CACHE.clear()
    _cfg._RAW_CONFIG_CACHE.clear()


@pytest.mark.parametrize("stale_env_base", [None, "https://stale.example/v1"])
def test_openai_main_model_runtime_and_discovery_use_the_configured_endpoint(tmp_path, monkeypatch, stale_env_base):
    """Inference and model discovery hit the SAME endpoint, the configured one, with the
    ``model.key_env`` key, whether or not a stale OPENAI_BASE_URL is exported."""
    from hermes_cli import models

    _write_openai_proxy_config(tmp_path, monkeypatch)
    monkeypatch.setenv("PROXY_KEY", "sk-proxy")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    if stale_env_base:
        monkeypatch.setenv("OPENAI_BASE_URL", stale_env_base)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    runtime = rp.resolve_runtime_provider()

    assert models._openai_discovery_base_url("openai") == runtime["base_url"] == "https://proxy.example/v1"
    assert runtime["api_key"] == "sk-proxy"


def test_openai_discovery_lists_model_base_url_with_model_key_env(tmp_path, monkeypatch):
    from hermes_cli import models

    _write_openai_proxy_config(tmp_path, monkeypatch)
    monkeypatch.setenv("PROXY_KEY", "sk-proxy")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    seen = {}

    def _fetch(api_key, base, **kwargs):
        seen.update(api_key=api_key, base=base)
        return ["gpt-x"]

    monkeypatch.setattr(models, "fetch_api_models", _fetch)
    assert models._openai_catalog("openai", False) == ["gpt-x"]
    assert seen == {"api_key": "sk-proxy", "base": "https://proxy.example/v1"}


def test_openai_discovery_does_not_post_openai_api_key_to_model_base_url(tmp_path, monkeypatch):
    """key_env declared but empty: discovery returns no catalog instead of posting OPENAI_API_KEY."""
    from hermes_cli import models

    _write_openai_proxy_config(tmp_path, monkeypatch)
    monkeypatch.delenv("PROXY_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    def _fetch(api_key, base, **kwargs):
        raise AssertionError(f"discovery posted {api_key} to {base}")

    monkeypatch.setattr(models, "fetch_api_models", _fetch)
    assert models._openai_catalog("openai", False) is None


_AUX_VISION_OPENAI_BLOCK = "auxiliary:\n  vision:\n    provider: openai\n    model: gpt-x\n    api_key: sk-openai-block\n"


@pytest.mark.parametrize(("proxy_key", "expected_key"), [("sk-proxy", "sk-proxy"), (None, "no-key-required")])
def test_aux_openai_block_on_model_base_url_uses_model_key_env(tmp_path, monkeypatch, proxy_key, expected_key):
    """An ``auxiliary.vision`` ``provider: openai`` block without base_url lands on model.base_url with
    model.key_env. Neither its own api_key nor OPENAI_API_KEY is sent there, even when key_env is empty."""
    from agent import auxiliary_client as aux

    _write_openai_proxy_config(tmp_path, monkeypatch, _AUX_VISION_OPENAI_BLOCK)
    if proxy_key:
        monkeypatch.setenv("PROXY_KEY", proxy_key)
    else:
        monkeypatch.delenv("PROXY_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
    aux._client_cache.clear()

    _provider, client, _model = aux.resolve_vision_provider_client()

    assert client is not None
    assert str(client.api_key) == expected_key
    assert "proxy.example" in str(client.base_url)


def _background_review_route(task_cfg):
    from agent import background_review

    parent = SimpleNamespace(provider="openai", model="gpt-x", _credential_pool=None, request_overrides={},
                             max_tokens=None, _current_main_runtime=dict)
    return background_review._resolve_review_runtime(parent, task_cfg)


def _curator_route(_task_cfg):
    from agent import curator

    return curator._resolve_review_provider()[0]


@pytest.mark.parametrize("route", [_background_review_route, _curator_route], ids=["background_review", "curator"])
@pytest.mark.parametrize(("proxy_key", "expected_key"), [("sk-proxy", "sk-proxy"), (None, "no-key-required")])
def test_runtime_path_aux_openai_block_on_model_base_url_uses_model_key_env(tmp_path, monkeypatch, route,
                                                                          proxy_key, expected_key):
    """background_review and curator resolve their block through resolve_runtime_provider. The block's
    key stays off model.base_url there too, so both auxiliary paths send the same credential."""
    block = {"provider": "openai", "model": "gpt-review", "api_key": "sk-openai-block"}
    _write_openai_proxy_config(tmp_path, monkeypatch, (
        "auxiliary:\n  background_review:\n    provider: openai\n    model: gpt-review\n    api_key: sk-openai-block\n"
        "  curator:\n    provider: openai\n    model: gpt-review\n    api_key: sk-openai-block\n"))
    if proxy_key:
        monkeypatch.setenv("PROXY_KEY", proxy_key)
    else:
        monkeypatch.delenv("PROXY_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    runtime = route(block)

    assert (runtime["base_url"], runtime["api_key"]) == ("https://proxy.example/v1", expected_key)
