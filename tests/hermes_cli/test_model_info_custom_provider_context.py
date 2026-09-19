"""Regression tests for ``/api/model/info`` custom-provider context-length resolution.

Issue #86097: the route read ``model.base_url`` straight from the model config and
never resolved the runtime provider nor forwarded ``custom_providers``. For installs
whose endpoint lives in a ``custom_providers`` entry, ``model.base_url`` is empty
(the normal shape for that setup), so every resolver route gated on a non-empty
``base_url`` was skipped and the settings UI showed the 256K fallback as
"Auto-detected" instead of the entry's real window.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.web_routers.models import get_model_info

ENTRY_URL = "http://127.0.0.1:8301/v1"


def _make_cfg(entry=None):
    model_cfg = {"default": "profile-main", "provider": "omlx-local", "base_url": ""}
    custom_providers = [entry] if entry is not None else []
    return {"model": model_cfg, "custom_providers": custom_providers}


def _run(cfg, resolved=None, resolve_raises=False):
    """Call ``get_model_info`` with config load, profile scope, and runtime
    provider resolution patched; return (info, calls-seen)."""
    seen = {}

    def fake_resolve(*, requested=None, explicit_api_key=None, explicit_base_url=None, target_model=None):
        if resolve_raises:
            raise RuntimeError("resolution unavailable")
        seen["requested"] = requested
        seen["target_model"] = target_model
        return dict(resolved or {})

    with (
        patch("hermes_cli.config.load_config", side_effect=lambda: cfg),
        patch("hermes_cli.web_server_profiles._profile_scope"),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_resolve),
        patch("agent.models_dev.get_model_capabilities", return_value=None),
    ):
        info = get_model_info(profile=None)

    return info, seen


def _capture_resolver_call(monkeypatch):
    """Record what the route hands to ``get_model_context_length``."""
    from agent import model_metadata

    captured = {}

    def fake_ctx(model, base_url="", api_key="", config_context_length=None, provider="", custom_providers=None):
        captured.update(model=model, base_url=base_url, api_key=api_key,
                        provider=provider, custom_providers=custom_providers)
        # Stand-in for the entry's real window (oMLX publishes 262144 via /v1/models).
        if base_url == ENTRY_URL and custom_providers:
            return 262144
        return 256000

    monkeypatch.setattr(model_metadata, "get_model_context_length", fake_ctx)
    return captured


def test_custom_provider_endpoint_reaches_the_resolver(monkeypatch):
    """The resolver must receive the custom entry's base_url and the config's
    custom_providers — not the empty model.base_url (#86097)."""
    captured = _capture_resolver_call(monkeypatch)
    entry = {"name": "omlx-local", "base_url": ENTRY_URL}

    info, seen = _run(_make_cfg(entry), resolved={"base_url": ENTRY_URL, "api_key": ""})

    assert seen["requested"] == "omlx-local"
    assert seen["target_model"] == "profile-main"
    assert captured["base_url"] == ENTRY_URL, "resolver got the empty model.base_url"
    assert captured["custom_providers"] == [entry], "custom_providers were not forwarded"
    assert info["auto_context_length"] == 262144
    assert info["effective_context_length"] == 262144


def test_resolver_key_forwarded_to_the_probe(monkeypatch):
    """The resolved entry credential rides along, so a keyed local server does not
    answer the context probe with 401s."""
    captured = _capture_resolver_call(monkeypatch)

    _run(_make_cfg({"name": "omlx-local", "base_url": ENTRY_URL}),
         resolved={"base_url": ENTRY_URL, "api_key": "route" + "-entry-" + "credential"})

    assert captured["api_key"] == "route" + "-entry-" + "credential"


def test_resolution_failure_degrades_to_config_base_url(monkeypatch):
    """A resolver error must not break the route: fall back to the (possibly
    empty) config base_url — the pre-fix behaviour, not a 500."""
    captured = _capture_resolver_call(monkeypatch)

    info, _ = _run(_make_cfg(), resolve_raises=True)

    assert captured["base_url"] == ""
    assert info["auto_context_length"] == 256000
    assert info["model"] == "profile-main"


def test_cloud_runtime_without_base_url_keeps_configured_one(monkeypatch):
    """A resolved cloud provider (no base_url of its own) leaves a configured
    model.base_url untouched."""
    captured = _capture_resolver_call(monkeypatch)
    cfg = _make_cfg()
    cfg["model"]["base_url"] = "https://proxy.example.internal/v1"

    _run(cfg, resolved={"provider": "openai"})

    assert captured["base_url"] == "https://proxy.example.internal/v1"
