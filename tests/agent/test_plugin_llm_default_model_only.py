"""Fail-closed routing for a plugin's bounded default-model call (#49)."""
import asyncio
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import plugin_llm
from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope, UnscopedSecretError
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager


def _ctx():
    return PluginContext(PluginManifest(name="sample", source="test", key="sample"), PluginManager())


def _response(model):
    return SimpleNamespace(model=model, choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'), finish_reason="stop")], usage=None)


def test_default_model_only_profile_and_recovery(tmp_path, monkeypatch):
    homes = [tmp_path / name for name in ("a", "b")]
    for home, model in zip(homes, ("primary-a", "primary-b")):
        home.mkdir()
        (home / "config.yaml").write_text(
            f"model:\n  provider: openrouter\n  default: {model}\n"
            "fallback_providers:\n  - provider: anthropic\n    model: other-model\n"
            "auxiliary:\n  fallback_providers: [anthropic]\n",
            encoding="utf-8",
        )
    calls = []
    unavailable = False
    failing = False

    class Sync:
        def __init__(self, model):
            self.model = model
            self.base_url = "https://api.openrouter.ai/v1"
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            calls.append(("wire", kwargs["model"]))
            if failing:
                raise ConnectionError("primary unreachable")
            return _response(self.model)

    class Async:
        def __init__(self, model):
            self.model = model
            self.base_url = "https://api.openrouter.ai/v1"
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(("async-wire", kwargs["model"]))
            if failing:
                raise ConnectionError("primary unreachable")
            return _response(self.model)

    def resolver(provider, model=None, async_mode=False, **kwargs):
        calls.append((provider, model))
        if provider != "openrouter":
            pytest.fail("discovered or fallback provider was contacted")
        if unavailable:
            return None, None
        return (Async(model) if async_mode else Sync(model)), model

    monkeypatch.setattr(aux, "resolve_provider_client", resolver)
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 0)
    ctx = _ctx()
    for home in (homes[0], homes[1], homes[0]):
        monkeypatch.setenv("HERMES_HOME", str(home))
        model = "primary-a" if home == homes[0] else "primary-b"
        result = ctx.llm.complete_structured(instructions="Extract", input=[{"type": "text", "text": "bounded excerpt"}],
                                             json_mode=True, default_model_only=True)
        assert (result.provider, result.model, result.parsed) == ("openrouter", model, {"ok": True})
        async_result = asyncio.run(ctx.llm.acomplete([{"role": "user", "content": "bounded excerpt"}], default_model_only=True))
        assert (async_result.provider, async_result.model) == ("openrouter", model)
        assert calls[-2:] == [("openrouter", model), ("async-wire", model)]

    unavailable = True
    aux._client_cache.clear()
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    calls.clear()
    with pytest.raises(aux.AuxiliaryClientUnavailable):
        ctx.llm.complete([{"role": "user", "content": "private"}], default_model_only=True)
    assert all(provider == "openrouter" for provider, _ in calls)

    unavailable = False
    failing = True
    aux._client_cache.clear()
    calls.clear()
    with pytest.raises(ConnectionError):
        ctx.llm.complete([{"role": "user", "content": "private"}], default_model_only=True)
    assert all(provider in {"openrouter", "wire"} for provider, _ in calls)
    calls.clear()
    with pytest.raises(ConnectionError):
        asyncio.run(ctx.llm.acomplete_structured(instructions="Extract", input=[{"type": "text", "text": "private"}],
                                                   default_model_only=True))
    assert all(provider in {"openrouter", "async-wire"} for provider, _ in calls)


@pytest.mark.parametrize("override", [{"provider": "openrouter"}, {"model": "elsewhere"}, {"task": "vision"}, {"profile": "other"}])
def test_default_model_only_rejects_route_overrides(override):
    with pytest.raises(ValueError, match="default_model_only"):
        _ctx().llm.complete([{"role": "user", "content": "private"}], default_model_only=True, **override)


def test_default_model_only_requires_concrete_config(tmp_path, monkeypatch):
    home = tmp_path / "empty"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  provider: auto\n  default: auto\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    with pytest.raises(aux.AuxiliaryClientUnavailable, match="concrete default"):
        _ctx().llm.complete([{"role": "user", "content": "private"}], default_model_only=True)


def test_default_model_only_context_scoped_config_secrets_and_attribution(tmp_path, monkeypatch):
    """A→B→A within one process: process env is deliberately always A."""
    homes = [tmp_path / name for name in ("a", "b")]
    for home, model in zip(homes, ("model-a", "model-b")):
        home.mkdir()
        (home / "config.yaml").write_text(
            f"model:\n  provider: openrouter\n  default: {model}\n  key_env: TEST_DEFAULT_KEY\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setenv("TEST_DEFAULT_KEY", "process-a-key")
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 0)
    observed = []

    class Wire:
        base_url = "https://api.openrouter.ai/v1"

        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            # A model-less reply must use the published route, not a fresh config read.
            observed.append(("wire", kwargs["model"]))
            return _response(None)

    def resolver(provider, model=None, async_mode=False, *, explicit_api_key=None, **kwargs):
        observed.append((provider, model, explicit_api_key))
        return Wire(), model

    monkeypatch.setattr(aux, "resolve_provider_client", resolver)
    monkeypatch.setattr(plugin_llm, "_main_config_value", lambda *args: "changed-after-route")
    set_multiplex_active(True)
    try:
        for home, model, key in ((homes[0], "model-a", "key-a"), (homes[1], "model-b", "key-b"),
                                 (homes[0], "model-a", "key-a")):
            home_token = set_hermes_home_override(home)
            secret_token = set_secret_scope({"TEST_DEFAULT_KEY": key})
            try:
                result = _ctx().llm.complete([{"role": "user", "content": "private"}], default_model_only=True)
                assert (result.provider, result.model) == ("openrouter", model)
                assert observed[-1] == ("wire", model)
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)

        assert [entry for entry in observed if entry[0] != "wire"] == [
            ("openrouter", "model-a", "key-a"), ("openrouter", "model-b", "key-b")]

        home_token = set_hermes_home_override(homes[1])
        try:
            secret_token = set_secret_scope({})
            try:
                with pytest.raises(aux.AuxiliaryClientUnavailable, match="credential"):
                    _ctx().llm.complete([{"role": "user", "content": "private"}], default_model_only=True)
            finally:
                reset_secret_scope(secret_token)
            with pytest.raises(UnscopedSecretError):
                _ctx().llm.complete([{"role": "user", "content": "private"}], default_model_only=True)
        finally:
            reset_hermes_home_override(home_token)
    finally:
        set_multiplex_active(False)
