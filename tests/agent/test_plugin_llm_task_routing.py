"""Regression coverage for plugin-owned auxiliary task routing."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from agent.plugin_llm import (
    PluginLlmTextInput,
    PluginLlmTrustError,
    _TrustPolicy,
    _check_task,
    _resolve_attribution,
    make_plugin_llm_for_test,
)


def _response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _policy() -> _TrustPolicy:
    return _TrustPolicy(plugin_id="plugin-key")


def _set_tasks(monkeypatch, entries: list[dict[str, str]], builtins: list[str] = []) -> None:
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_auxiliary_tasks", lambda: entries)
    monkeypatch.setattr(
        "hermes_cli.main._AUX_TASKS", [(key, key, "") for key in builtins]
    )


def test_task_gate_allows_owned_and_rejects_foreign(monkeypatch, caplog):
    _set_tasks(monkeypatch, [{"key": "classifier", "plugin": "plugin-key"}])
    assert _check_task(_policy(), plugin_id="plugin-key", requested_task=" classifier ") == "classifier"
    assert _check_task(_policy(), plugin_id="plugin-key", requested_task="auto") is None

    _set_tasks(monkeypatch, [{"key": "classifier", "plugin": "other"}])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(PluginLlmTrustError, match="classifier"):
            _check_task(_policy(), plugin_id="plugin-key", requested_task="classifier")
    assert any("plugin-key" in record.getMessage() for record in caplog.records)


def test_builtin_task_requires_explicit_opt_in(monkeypatch):
    _set_tasks(monkeypatch, [], ["vision"])
    with pytest.raises(PluginLlmTrustError, match="allow_task_override"):
        _check_task(_policy(), plugin_id="plugin-key", requested_task="vision")
    allowed = _TrustPolicy(plugin_id="plugin-key", allow_task_override=True)
    assert _check_task(allowed, plugin_id="plugin-key", requested_task="vision") == "vision"


def test_all_public_variants_preserve_owned_task_and_audit(monkeypatch):
    _set_tasks(monkeypatch, [{"key": "classifier", "plugin": "plugin-key"}])
    calls: list[dict[str, Any]] = []

    def sync_caller(**kwargs: Any):
        calls.append(kwargs)
        return "aux-provider", "aux-model", _response()

    async def async_caller(**kwargs: Any):
        calls.append(kwargs)
        return "aux-provider", "aux-model", _response('{"kind":"ok"}')

    llm = make_plugin_llm_for_test(
        plugin_id="plugin-key", policy=_policy(), sync_caller=sync_caller, async_caller=async_caller
    )
    complete = llm.complete([{"role": "user", "content": "x"}], task="classifier")
    structured = llm.complete_structured(
        instructions="classify", input=[PluginLlmTextInput(text="x")], task="classifier"
    )
    async_complete = asyncio.run(
        llm.acomplete([{"role": "user", "content": "x"}], task="classifier")
    )
    async_structured = asyncio.run(
        llm.acomplete_structured(
            instructions="classify", input=[PluginLlmTextInput(text="x")], task="classifier"
        )
    )
    for result in (complete, structured, async_complete, async_structured):
        assert (result.provider, result.model) == ("aux-provider", "aux-model")
        assert result.audit["task"] == "classifier"
    assert [call["task"] for call in calls] == ["classifier"] * 4


def test_async_variants_write_resolved_route_audit_logs(monkeypatch, caplog):
    _set_tasks(monkeypatch, [{"key": "classifier", "plugin": "plugin-key"}])

    async def async_caller(**_kwargs: Any):
        return "aux-provider", "aux-model", _response('{"kind":"ok"}')

    llm = make_plugin_llm_for_test(
        plugin_id="plugin-key", policy=_policy(), async_caller=async_caller
    )
    with caplog.at_level(logging.INFO, logger="agent.plugin_llm"):
        asyncio.run(
            llm.acomplete(
                [{"role": "user", "content": "x"}],
                purpose="async-chat",
                task="classifier",
            )
        )
        asyncio.run(
            llm.acomplete_structured(
                instructions="classify",
                input=[PluginLlmTextInput(text="x")],
                purpose="async-structured",
                task="classifier",
            )
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "plugin_llm.acomplete plugin=plugin-key provider=aux-provider "
        "model=aux-model task=classifier purpose=async-chat tokens=2" in message
        for message in messages
    )
    assert any(
        "plugin_llm.acomplete_structured plugin=plugin-key provider=aux-provider "
        "model=aux-model task=classifier purpose=async-structured "
        "content_type=text tokens=2" in message
        for message in messages
    )


def test_task_route_uses_real_resolver_and_reports_the_selected_route(tmp_path, monkeypatch, caplog):
    from agent import auxiliary_client
    from hermes_cli import config as config_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """model:\n  provider: main-provider\n  model: main-model\nauxiliary:\n  classifier:\n    provider: aux-provider\n    model: aux-model\n""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(config_mod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})

    manager = PluginManager()
    manager._discovered = True
    ctx = PluginContext(PluginManifest(name="display", key="plugin-key"), manager)
    ctx.register_auxiliary_task("classifier", display_name="Classifier", description="test")
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda: manager)

    selected: dict[str, str] = {}

    def fake_cached(provider, model, **_kwargs):
        selected.update(provider=provider, model=model)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: _response()))
        ), model

    monkeypatch.setattr(auxiliary_client, "_get_cached_client", fake_cached)
    with caplog.at_level(logging.INFO, logger="agent.plugin_llm"):
        result = ctx.llm.complete([{"role": "user", "content": "x"}], task="classifier")

    assert selected == {"provider": "aux-provider", "model": "aux-model"}
    assert (result.provider, result.model) == ("aux-provider", "aux-model")
    assert result.audit["task"] == "classifier"
    assert any("provider=aux-provider model=aux-model task=classifier" in record.getMessage() for record in caplog.records)


def test_task_route_fallback_reports_successful_provider_and_model(tmp_path, monkeypatch):
    from agent import auxiliary_client
    from hermes_cli import config as config_mod

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """auxiliary:\n  classifier:\n    provider: primary-provider\n    model: primary-model\n    fallback_chain:\n      - provider: fallback-provider\n        model: fallback-model\n""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(config_mod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})
    _set_tasks(monkeypatch, [{"key": "classifier", "plugin": "plugin-key"}])
    monkeypatch.setattr(auxiliary_client, "_transient_retry_count", lambda: 0)
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("down"))))),
            "primary-model",
        ),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "resolve_provider_client",
        lambda _provider, model, **_kwargs: (
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: _response()))),
            model,
        ),
    )

    result = make_plugin_llm_for_test(plugin_id="plugin-key", policy=_policy()).complete(
        [{"role": "user", "content": "x"}], task="classifier"
    )
    assert (result.provider, result.model) == ("fallback-provider", "fallback-model")


def test_async_task_route_fallback_reports_successful_provider_and_model(tmp_path, monkeypatch):
    from agent import auxiliary_client
    from hermes_cli import config as config_mod

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """auxiliary:\n  classifier:\n    provider: primary-provider\n    model: primary-model\n    fallback_chain:\n      - provider: fallback-provider\n        model: fallback-model\n""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(config_mod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})
    _set_tasks(monkeypatch, [{"key": "classifier", "plugin": "plugin-key"}])

    async def fail(**_kwargs):
        raise ConnectionError("down")

    async def succeed(**_kwargs):
        return _response()

    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))),
            "primary-model",
        ),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "resolve_provider_client",
        lambda _provider, model, **_kwargs: (
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=succeed))),
            model,
        ),
    )
    monkeypatch.setattr(auxiliary_client, "_to_async_client", lambda client, model, **_kwargs: (client, model))

    result = asyncio.run(
        make_plugin_llm_for_test(plugin_id="plugin-key", policy=_policy()).acomplete(
            [{"role": "user", "content": "x"}], task="classifier"
        )
    )
    assert (result.provider, result.model) == ("fallback-provider", "fallback-model")


@pytest.mark.parametrize("async_mode", [False, True])
def test_unavailable_task_provider_reports_configured_fallback(tmp_path, monkeypatch, async_mode):
    from agent import auxiliary_client
    from hermes_cli import config as config_mod

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """auxiliary:\n  classifier:\n    provider: unavailable-provider\n    model: primary-model\n    fallback_chain:\n      - provider: fallback-provider\n        model: fallback-model\n""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(config_mod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})
    _set_tasks(monkeypatch, [{"key": "classifier", "plugin": "plugin-key"}])
    monkeypatch.setattr(auxiliary_client, "_get_cached_client", lambda *_args, **_kwargs: (None, None))

    if async_mode:
        async def create(**_kwargs):
            return _response()
    else:
        def create(**_kwargs):
            return _response()

    fallback_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(
        auxiliary_client,
        "resolve_provider_client",
        lambda _provider, model, **_kwargs: (fallback_client, model),
    )
    if async_mode:
        monkeypatch.setattr(
            auxiliary_client, "_to_async_client", lambda client, model, **_kwargs: (client, model)
        )
        result = asyncio.run(
            make_plugin_llm_for_test(plugin_id="plugin-key", policy=_policy()).acomplete(
                [{"role": "user", "content": "x"}], task="classifier"
            )
        )
    else:
        result = make_plugin_llm_for_test(plugin_id="plugin-key", policy=_policy()).complete(
            [{"role": "user", "content": "x"}], task="classifier"
        )
    assert (result.provider, result.model) == ("fallback-provider", "fallback-model")


def test_default_call_retains_main_route_path(monkeypatch):
    _set_tasks(monkeypatch, [])
    seen: dict[str, Any] = {}

    def caller(**kwargs: Any):
        seen.update(kwargs)
        return "main-provider", "main-model", _response()

    result = make_plugin_llm_for_test(plugin_id="plugin-key", policy=_policy(), sync_caller=caller).complete(
        [{"role": "user", "content": "x"}]
    )
    assert seen["task"] is None
    assert result.audit["task"] == ""
    assert (result.provider, result.model) == ("main-provider", "main-model")


def test_successful_fallback_route_beats_requested_override_for_attribution():
    provider, model = _resolve_attribution(
        provider_override="primary-provider",
        model_override="primary-model",
        response=_response(),
        route_info={"provider": "fallback-provider", "model": "fallback-model"},
    )
    assert (provider, model) == ("fallback-provider", "fallback-model")
