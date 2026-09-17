"""Plugin-distributed model metadata (#102115), without user config edits."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import providers
from providers.base import ProviderProfile
from agent import models_dev


def test_plugin_metadata_resolves_without_catalog_and_user_override_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_discovered", True)
    monkeypatch.setattr(models_dev, "_registry_models", lambda *a, **k: None)
    overrides = {}
    monkeypatch.setattr(models_dev, "_load_model_overrides", lambda: overrides)
    declaration = {"tier-high": {"supports_reasoning": False, "supports_vision": True,
                                  "supports_tools": True, "context_window": 64000}}
    original = deepcopy(declaration)
    providers.register_provider(ProviderProfile(
        name="fixture-provider", aliases=("fixture-alias",), model_capabilities=declaration))

    for name in ("fixture-provider", "fixture-alias"):
        caps = models_dev.get_model_capabilities(name, "tier-high")
        info = models_dev.get_model_info(name, "tier-high")
        assert caps.supports_reasoning is False
        assert caps.supports_vision is True
        assert caps.supports_tools is True
        assert caps.context_window == info.context_window == 64000
        assert info.reasoning is False
        assert models_dev.lookup_models_dev_context(name, "tier-high") == 64000
        assert models_dev.get_model_capabilities(name, "undeclared") is None

    overrides["fixture-provider"] = {"tier-high": {"supports_reasoning": True}}
    caps = models_dev.get_model_capabilities("fixture-provider", "tier-high")
    assert caps.supports_reasoning is True
    assert caps.supports_vision is True
    assert caps.context_window == 64000
    overrides["_default"] = {"context_window": 8000}
    assert models_dev.lookup_models_dev_context("fixture-provider", "tier-high") == 64000
    overrides["fixture-provider"]["tier-high"]["context_window"] = 96000
    assert models_dev.lookup_models_dev_context("fixture-provider", "tier-high") == 96000
    assert declaration == original

    # Fresh interpreters exercise the real filesystem loader/config reader;
    # monkeypatched registries above cannot leak into these consumers.
    home = tmp_path / "profile"
    plugin = home / "plugins" / "model-providers" / "fixture-provider"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='fixture-provider', "
        "aliases=('fixture-alias',), model_capabilities=" + repr(original) + "))\n"
    )
    probe = (
        "import json\n"
        "from agent.models_dev import get_model_capabilities, lookup_models_dev_context\n"
        "from hermes_cli.config import load_config_readonly\n"
        "cfg = load_config_readonly()['model']\n"
        "p, m = cfg['provider'], cfg['default']\n"
        "caps = get_model_capabilities(p, m)\n"
        "print(json.dumps([caps.supports_reasoning, caps.supports_vision, "
        "caps.context_window, lookup_models_dev_context(p, m)]))\n"
    )
    env = {"HOME": str(tmp_path), "HERMES_HOME": str(home), "PATH": os.defpath,
           "PYTHONIOENCODING": "utf-8"}
    for provider, patch, expected in (
        ("fixture-provider", {"_default": {"context_window": 8000}}, [False, True, 64000, 64000]),
        ("fixture-alias", {}, [False, True, 64000, 64000]),
        ("fixture-alias", {"fixture-alias": {"tier-high": {
            "supports_reasoning": True, "context_window": 96000}}}, [True, True, 96000, 96000]),
    ):
        # JSON is valid YAML; no third-party serializer needed for the fixture.
        (home / "config.yaml").write_text(json.dumps({
            "model": {"provider": provider, "default": "tier-high"},
            "model_overrides": patch,
        }))
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[2],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout.splitlines()[-1]) == expected


def test_partial_plugin_metadata_preserves_unknowns_and_catalog_fields(monkeypatch):
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_discovered", True)
    catalog = {"known": {"reasoning": True, "tool_call": True,
                         "limit": {"context": 32000, "output": 4000},
                         "modalities": {"input": ["text", "image"]}}}
    original = deepcopy(catalog)
    monkeypatch.setitem(models_dev.PROVIDER_TO_MODELS_DEV, "fixture-provider", "fixture-provider")
    monkeypatch.setattr(models_dev, "_registry_models", lambda *a, **k: catalog)
    monkeypatch.setattr(models_dev, "_load_model_overrides", lambda: {})
    providers.register_provider(ProviderProfile(name="fixture-provider", model_capabilities={
        "known": {"context_window": 64000},
        "unknown": {"context_window": 48000},
    }))
    known = models_dev.get_model_capabilities("fixture-provider", "known")
    assert known.context_window == 64000
    assert known.max_output_tokens == 4000
    assert known.supports_reasoning is True
    assert known.supports_vision is True
    unknown = models_dev.get_model_capabilities("fixture-provider", "unknown")
    assert unknown.context_window == 48000
    assert unknown.supports_reasoning is None
    assert unknown.supports_vision is None
    assert catalog == original
