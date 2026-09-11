"""Tests for plugin auxiliary-task routing via ``ctx.llm.complete(task=...)``.

Covers issue #64174 (sub-issue 08/14 of #64182): a plugin can route an
LLM call through an auxiliary model slot it registered, the default path
is unchanged, and a foreign/unknown task key is rejected loudly rather
than silently downgraded to the main model (round-2 design correction).

The auxiliary client is stubbed via ``make_plugin_llm_for_test`` so the
injected caller both captures the ``task`` that would reach
``call_llm`` and stands in for a distinguishable slot model.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from agent.plugin_llm import (
    PluginLlmTrustError,
    PluginLlmTextInput,
    _check_task,
    _resolve_attribution,
    _resolve_task_ownership,
    _TrustPolicy,
    make_plugin_llm_for_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response(text: str = "ok", *, prompt: int = 3, completion: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def _capturing_caller(captured: Dict[str, Any]):
    """Sync caller that records kwargs and reports a slot-derived model.

    When a ``task`` is routed it reports ``("aux-provider", "aux-model")``
    so a test can prove the call provably landed on the slot's model
    (acceptance criterion: distinguishable model).
    """

    def caller(**kwargs: Any):
        captured.update(kwargs)
        if kwargs.get("task"):
            return "aux-provider", "aux-model", _fake_response()
        return "main-provider", "main-model", _fake_response()

    return caller


def _async_capturing_caller(captured: Dict[str, Any]):
    async def caller(**kwargs: Any):
        captured.update(kwargs)
        if kwargs.get("task"):
            return "aux-provider", "aux-model", _fake_response()
        return "main-provider", "main-model", _fake_response()

    return caller


def _set_registry(monkeypatch, entries: List[Dict[str, Any]]) -> None:
    """Point ``_resolve_task_ownership`` at a controlled plugin registry."""
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_auxiliary_tasks", lambda: list(entries)
    )


def _set_builtins(monkeypatch, keys: List[str]) -> None:
    monkeypatch.setattr(
        "hermes_cli.main_provider_setup._AUX_TASKS", [(k, k.title(), "") for k in keys]
    )


def _policy(plugin_id: str = "my-plugin", *, allow_task_override: bool = False) -> _TrustPolicy:
    return _TrustPolicy(plugin_id=plugin_id, allow_task_override=allow_task_override)


# ---------------------------------------------------------------------------
# _check_task unit behavior
# ---------------------------------------------------------------------------


class TestCheckTask:
    def test_none_returns_none(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, [])
        assert _check_task(_policy(), plugin_id="my-plugin", requested_task=None) is None

    @pytest.mark.parametrize("raw", ["auto", "AUTO", "  auto  ", "", "   "])
    def test_auto_and_blank_return_none(self, monkeypatch, raw):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        assert _check_task(_policy(), plugin_id="my-plugin", requested_task=raw) is None

    def test_own_registered_key_allowed(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, ["vision"])
        assert (
            _check_task(_policy(), plugin_id="my-plugin", requested_task="classifier")
            == "classifier"
        )

    def test_own_key_stripped(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        assert (
            _check_task(_policy(), plugin_id="my-plugin", requested_task="  classifier ")
            == "classifier"
        )

    def test_foreign_key_rejected_and_named(self, monkeypatch, caplog):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "other-plugin"}])
        _set_builtins(monkeypatch, ["vision"])
        with caplog.at_level(logging.WARNING):
            with pytest.raises(PluginLlmTrustError) as exc:
                _check_task(_policy(), plugin_id="my-plugin", requested_task="classifier")
        # Error names both offending plugin and key; no silent fallback.
        assert "my-plugin" in str(exc.value)
        assert "classifier" in str(exc.value)
        assert any(
            "my-plugin" in r.getMessage() and "classifier" in r.getMessage()
            for r in caplog.records
        )

    def test_unknown_key_rejected(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, ["vision"])
        with pytest.raises(PluginLlmTrustError):
            _check_task(_policy(), plugin_id="my-plugin", requested_task="nope")

    def test_builtin_key_denied_without_flag(self, monkeypatch, caplog):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, ["vision", "compression"])
        with caplog.at_level(logging.WARNING):
            with pytest.raises(PluginLlmTrustError) as exc:
                _check_task(
                    _policy(allow_task_override=False),
                    plugin_id="my-plugin",
                    requested_task="vision",
                )
        assert "allow_task_override" in str(exc.value)
        assert any("vision" in r.getMessage() for r in caplog.records)

    def test_builtin_key_allowed_with_flag(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, ["vision", "compression"])
        assert (
            _check_task(
                _policy(allow_task_override=True),
                plugin_id="my-plugin",
                requested_task="vision",
            )
            == "vision"
        )

    def test_own_key_wins_over_builtin_flag_requirement(self, monkeypatch):
        # A plugin's own slot never needs allow_task_override, even if a
        # built-in of the same name somehow existed — own ownership is checked
        # first.
        _set_registry(monkeypatch, [{"key": "shared", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, ["shared"])
        assert (
            _check_task(
                _policy(allow_task_override=False),
                plugin_id="my-plugin",
                requested_task="shared",
            )
            == "shared"
        )


# ---------------------------------------------------------------------------
# End-to-end routing through PluginLlm (sync + async, plain + structured)
# ---------------------------------------------------------------------------


class TestRouting:
    def test_default_call_passes_task_none(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, [])
        captured: Dict[str, Any] = {}
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_policy(),
            sync_caller=_capturing_caller(captured),
        )
        result = llm.complete([{"role": "user", "content": "hi"}])
        assert captured["task"] is None
        assert result.provider == "main-provider"
        assert result.model == "main-model"
        assert result.audit["task"] == ""

    def test_registered_task_routes_and_reports_slot_model(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, ["vision"])
        captured: Dict[str, Any] = {}
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_policy(),
            sync_caller=_capturing_caller(captured),
        )
        result = llm.complete([{"role": "user", "content": "hi"}], task="classifier")
        # Provably routed: the task reached call_llm and the slot model won.
        assert captured["task"] == "classifier"
        assert result.provider == "aux-provider"
        assert result.model == "aux-model"
        assert result.audit["task"] == "classifier"

    def test_foreign_task_raises_before_invoking_caller(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "other-plugin"}])
        _set_builtins(monkeypatch, [])
        captured: Dict[str, Any] = {}
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_policy(),
            sync_caller=_capturing_caller(captured),
        )
        with pytest.raises(PluginLlmTrustError):
            llm.complete([{"role": "user", "content": "hi"}], task="classifier")
        # The caller must never run for a rejected task — no wrong-model call.
        assert captured == {}

    def test_unknown_task_raises_before_invoking_caller(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, [])
        captured: Dict[str, Any] = {}
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_policy(),
            sync_caller=_capturing_caller(captured),
        )
        with pytest.raises(PluginLlmTrustError):
            llm.complete([{"role": "user", "content": "hi"}], task="unknown")
        assert captured == {}

    def test_structured_routes_task(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        captured: Dict[str, Any] = {}
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_policy(),
            sync_caller=_capturing_caller(captured),
        )
        result = llm.complete_structured(
            instructions="classify this",
            input=[PluginLlmTextInput(text="payload")],
            task="classifier",
        )
        assert captured["task"] == "classifier"
        assert result.audit["task"] == "classifier"

    def test_async_routes_task(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        captured: Dict[str, Any] = {}
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_policy(),
            async_caller=_async_capturing_caller(captured),
        )
        result = asyncio.run(
            llm.acomplete([{"role": "user", "content": "hi"}], task="classifier")
        )
        assert captured["task"] == "classifier"
        assert result.audit["task"] == "classifier"

    def test_async_structured_routes_task(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        captured: Dict[str, Any] = {}
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_policy(),
            async_caller=_async_capturing_caller(captured),
        )
        result = asyncio.run(
            llm.acomplete_structured(
                instructions="classify this",
                input=[PluginLlmTextInput(text="payload")],
                task="classifier",
            )
        )
        assert captured["task"] == "classifier"
        assert result.audit["task"] == "classifier"

    def test_async_variants_log_exact_route(self, monkeypatch, caplog):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_policy(),
            async_caller=_async_capturing_caller({}),
        )

        with caplog.at_level(logging.INFO, logger="agent.plugin_llm"):
            asyncio.run(
                llm.acomplete(
                    [{"role": "user", "content": "hi"}],
                    task="classifier",
                    purpose="plain",
                )
            )
            asyncio.run(
                llm.acomplete_structured(
                    instructions="classify",
                    input=[PluginLlmTextInput(text="payload")],
                    task="classifier",
                    purpose="structured",
                )
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "plugin_llm.acomplete plugin=my-plugin provider=aux-provider "
            "model=aux-model task=classifier purpose=plain tokens=8" in message
            for message in messages
        )
        assert any(
            "plugin_llm.acomplete_structured plugin=my-plugin provider=aux-provider "
            "model=aux-model task=classifier purpose=structured "
            "content_type=text tokens=8" in message
            for message in messages
        )


def test_successful_fallback_route_beats_requested_route_for_attribution():
    provider, model = _resolve_attribution(
        provider_override="primary-provider",
        model_override="primary-model",
        response=_fake_response(),
        route_info={"provider": "fallback-provider", "model": "fallback-model"},
    )
    assert (provider, model) == ("fallback-provider", "fallback-model")


class TestForwardsToCallLlm:
    """Cover the production ``_invoke_*`` path (no injected caller), which is
    where the previously-hardcoded ``task=None`` is replaced by the routed
    key. The injected-caller tests above bypass this line."""

    def test_sync_task_uses_auxiliary_attribution_and_log(self, monkeypatch, caplog):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        seen: Dict[str, Any] = {}

        def fake_call_llm(**kwargs: Any):
            seen.update(kwargs)
            kwargs["route_info"].update(provider="aux-provider", model="aux-model")
            return _fake_response()

        monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
        llm = make_plugin_llm_for_test(plugin_id="my-plugin", policy=_policy())
        with caplog.at_level(logging.INFO, logger="agent.plugin_llm"):
            result = llm.complete([{"role": "user", "content": "hi"}], task="classifier")
        assert seen["task"] == "classifier"
        assert (result.provider, result.model) == ("aux-provider", "aux-model")
        assert any(
            "provider=aux-provider model=aux-model task=classifier" in record.getMessage()
            for record in caplog.records
        )

    def test_sync_default_forwards_task_none(self, monkeypatch):
        _set_registry(monkeypatch, [])
        _set_builtins(monkeypatch, [])
        seen: Dict[str, Any] = {}

        def fake_call_llm(**kwargs: Any):
            seen.update(kwargs)
            return _fake_response()

        monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
        llm = make_plugin_llm_for_test(plugin_id="my-plugin", policy=_policy())
        llm.complete([{"role": "user", "content": "hi"}])
        assert seen["task"] is None

    def test_async_task_uses_auxiliary_attribution(self, monkeypatch):
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])
        seen: Dict[str, Any] = {}

        async def fake_async_call_llm(**kwargs: Any):
            seen.update(kwargs)
            kwargs["route_info"].update(provider="aux-provider", model="aux-model")
            return _fake_response()

        monkeypatch.setattr("agent.auxiliary_client.async_call_llm", fake_async_call_llm)
        llm = make_plugin_llm_for_test(plugin_id="my-plugin", policy=_policy())
        result = asyncio.run(
            llm.acomplete([{"role": "user", "content": "hi"}], task="classifier")
        )
        assert seen["task"] == "classifier"
        assert (result.provider, result.model) == ("aux-provider", "aux-model")


# ---------------------------------------------------------------------------
# Ownership resolution against the real plugin registry
# ---------------------------------------------------------------------------


class TestOwnershipIntegration:
    def _make_manager(self):
        from hermes_cli.plugins import PluginManager

        manager = PluginManager()
        manager._discovered = True
        return manager

    def _register(self, manager, *, name: str, key: str, task_key: str):
        from hermes_cli.plugins import PluginContext, PluginManifest

        manifest = PluginManifest(name=name, key=key)
        ctx = PluginContext(manifest, manager)
        ctx.register_auxiliary_task(
            task_key, display_name=task_key.title(), description="x"
        )
        return ctx

    def test_owner_stored_as_canonical_id(self, monkeypatch):
        # A manifest with a distinct key stores the canonical id (key), which
        # is exactly what ctx.llm is bound to — so the trust gate matches.
        manager = self._make_manager()
        self._register(manager, name="Display Name", key="my_key", task_key="classifier")
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda: manager
        )
        _set_builtins(monkeypatch, ["vision"])

        owned, builtin = _resolve_task_ownership("my_key")
        assert "classifier" in owned
        assert "vision" in builtin
        # The name (not the canonical id) does not own it.
        owned_by_name, _ = _resolve_task_ownership("Display Name")
        assert "classifier" not in owned_by_name

    def test_check_task_end_to_end_with_real_registry(self, monkeypatch):
        manager = self._make_manager()
        self._register(manager, name="p", key="", task_key="classifier")
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda: manager
        )
        _set_builtins(monkeypatch, ["vision"])

        assert (
            _check_task(_policy(plugin_id="p"), plugin_id="p", requested_task="classifier")
            == "classifier"
        )
        with pytest.raises(PluginLlmTrustError):
            _check_task(
                _policy(plugin_id="other"), plugin_id="other", requested_task="classifier"
            )

    def test_auto_task_reports_configured_fallback_provider_and_model(self, tmp_path, monkeypatch):
        from agent import auxiliary_client as auxiliary_mod
        from hermes_cli import config as config_mod

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            """
auxiliary:
  classifier:
    provider: auto
    fallback_chain:
      - provider: fallback-provider
        model: fallback-model
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(config_mod, "_LOAD_CONFIG_CACHE", {})
        monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})

        manager = self._make_manager()
        ctx = self._register(manager, name="my-plugin", key="my-plugin", task_key="classifier")
        monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda: manager)
        _set_builtins(monkeypatch, [])
        monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: "")
        monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: "")

        captured: Dict[str, Any] = {}
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: _fake_response())
            )
        )

        real_provider_client = auxiliary_mod.resolve_provider_client

        def fake_provider_client(provider, model, _async_mode=False, **kwargs):
            if provider == "auto":
                return real_provider_client(provider, model, _async_mode, **kwargs)
            captured.update(provider=provider, model=model, **kwargs)
            return client, model

        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", fake_provider_client
        )

        result = ctx.llm.complete(
            [{"role": "user", "content": "hi"}], task="classifier"
        )

        assert (captured["provider"], captured["model"]) == (
            "fallback-provider", "fallback-model"
        )
        assert (result.provider, result.model) == (
            "fallback-provider", "fallback-model"
        )

    def test_async_auto_resolution_preserves_route_provider(self, monkeypatch):
        from agent import auxiliary_client

        sync_client = SimpleNamespace()
        async_client = SimpleNamespace()
        monkeypatch.setattr(
            auxiliary_client,
            "_resolve_auto_route",
            lambda **_kwargs: (sync_client, "fallback-model", "fallback-provider"),
        )
        monkeypatch.setattr(
            auxiliary_client,
            "_to_async_client",
            lambda _client, model, **_kwargs: (async_client, model),
        )

        resolved_client, model = auxiliary_client.resolve_provider_client(
            "auto", async_mode=True, task="classifier"
        )

        assert resolved_client is async_client
        assert auxiliary_client._effective_provider_for_client(
            resolved_client, "auto"
        ) == "fallback-provider"
        assert model == "fallback-model"

    def test_sync_fallback_reports_the_successful_route(self, tmp_path, monkeypatch):
        from hermes_cli import config as config_mod

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            """
auxiliary:
  classifier:
    provider: primary-provider
    model: primary-model
    fallback_chain:
      - provider: fallback-provider
        model: fallback-model
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(config_mod, "_LOAD_CONFIG_CACHE", {})
        monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])

        def fail(**_kwargs):
            raise ConnectionError("connection refused")

        failed_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
        )
        fallback_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: _fake_response()))
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._get_cached_client",
            lambda provider, model, **_kwargs: (failed_client, model),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client",
            lambda provider, model, **_kwargs: (fallback_client, model),
        )
        monkeypatch.setattr("agent.auxiliary_client._transient_retry_count", lambda: 0)

        result = make_plugin_llm_for_test(
            plugin_id="my-plugin", policy=_policy()
        ).complete([{"role": "user", "content": "hi"}], task="classifier")

        assert (result.provider, result.model) == ("fallback-provider", "fallback-model")

    def test_async_fallback_reports_the_successful_route(self, tmp_path, monkeypatch):
        from hermes_cli import config as config_mod

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            """
auxiliary:
  classifier:
    provider: primary-provider
    model: primary-model
    fallback_chain:
      - provider: fallback-provider
        model: fallback-model
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(config_mod, "_LOAD_CONFIG_CACHE", {})
        monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})
        _set_registry(monkeypatch, [{"key": "classifier", "plugin": "my-plugin"}])
        _set_builtins(monkeypatch, [])

        async def fail(**_kwargs):
            raise ConnectionError("connection refused")

        async def succeed(**_kwargs):
            return _fake_response()

        failed_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
        )
        fallback_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=succeed))
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._get_cached_client",
            lambda provider, model, **_kwargs: (failed_client, model),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client",
            lambda provider, model, **_kwargs: (fallback_client, model),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._to_async_client",
            lambda client, model, **_kwargs: (client, model),
        )

        result = asyncio.run(
            make_plugin_llm_for_test(plugin_id="my-plugin", policy=_policy()).acomplete(
                [{"role": "user", "content": "hi"}], task="classifier"
            )
        )

        assert (result.provider, result.model) == ("fallback-provider", "fallback-model")
