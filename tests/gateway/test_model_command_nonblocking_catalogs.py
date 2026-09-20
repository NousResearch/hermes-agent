"""#74003: the chat ``/model`` listing must never join live catalog probes.

``list_authenticated_providers(non_blocking_catalogs=True)`` is the cache-only read path (rows read
the disk cache, stale ones warm in the background). The gateway listing has to opt in, and
``list_picker_providers`` has to forward the flag — otherwise a cold cache with several credentialed
providers force-refreshes every catalog in the request path."""

import json

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

_PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY",
                  "MISTRAL_API_KEY", "XAI_API_KEY")


def test_picker_listing_with_non_blocking_catalogs_never_forces_a_refresh(tmp_path, monkeypatch):
    import hermes_cli.models as models
    from hermes_cli.model_switch_providers import list_picker_providers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in _PROVIDER_KEYS:
        monkeypatch.setenv(key, "dummy")
    (tmp_path / "provider_models_cache.json").write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr(models, "fetch_api_models", lambda *a, **k: [])

    forced: list[str] = []
    real_cached = models.cached_provider_model_ids

    def counting_cached(slug, *args, **kwargs):
        if kwargs.get("force_refresh"):
            forced.append(slug)
        return real_cached(slug, *args, **kwargs)

    monkeypatch.setattr(models, "cached_provider_model_ids", counting_cached)

    list_picker_providers(max_models=50, include_moa=True, current_provider="deepseek",
                          user_providers={}, custom_providers=[], excluded_providers=[],
                          non_blocking_catalogs=True)

    assert forced == [], f"cache-only listing force-refreshed catalogs in the request path: {forced}"


class _PickerResult:
    success = True


class _PickerAdapter:
    async def send_model_picker(self, **kwargs):
        return _PickerResult()


@pytest.mark.asyncio
async def test_gateway_model_listing_passes_non_blocking_catalogs(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: gpt-x\n  provider: openrouter\nproviders: {}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    seen: list[dict] = []

    def fake_list_picker_providers(**kwargs):
        seen.append(kwargs)
        return [{"slug": "openrouter", "name": "OpenRouter", "is_current": True,
                 "models": ["gpt-x"], "total_models": 1}]

    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers",
                        fake_list_picker_providers)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: _PickerAdapter()}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)

    event = MessageEvent(text="/model", message_type=MessageType.TEXT,
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"))
    assert await runner._handle_model_command(event) is None
    assert seen and seen[0].get("non_blocking_catalogs") is True, (
        f"gateway /model listing must pass non_blocking_catalogs=True, got {seen}")

