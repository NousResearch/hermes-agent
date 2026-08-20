"""OpenRouter live-free picker group.

The curated OpenRouter row stays intact. A second picker row uses a unique
display slug (``openrouter-free``) and a separate ``runtime_slug`` of
``openrouter`` so Telegram/Discord can index both rows without switching
onto a fake provider.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from hermes_cli import model_switch
from hermes_cli.models import (
    OPENROUTER_FREE_PICKER_SLUG,
    OPENROUTER_FREE_RUNTIME_SLUG,
    _openrouter_model_is_free,
    _openrouter_model_supports_tools,
    fetch_openrouter_free_models,
    picker_runtime_slug,
)
import hermes_cli.models as models_mod


class _Resp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


LIVE_FREE_PAYLOAD = (
    b'{"data":['
    b'{"id":"nvidia/nemotron-3-ultra-550b-a55b:free",'
    b'"pricing":{"prompt":"0","completion":"0"},'
    b'"supported_parameters":["tools","temperature"]},'
    b'{"id":"openrouter/free",'
    b'"pricing":{"prompt":"0","completion":"0"},'
    b'"supported_parameters":["tools"]},'
    b'{"id":"paid/model",'
    b'"pricing":{"prompt":"0.001","completion":"0.002"},'
    b'"supported_parameters":["tools"]},'
    b'{"id":"free/no-tools",'
    b'"pricing":{"prompt":"0","completion":"0"},'
    b'"supported_parameters":["temperature"]},'
    b'{"id":"free/unknown-tools",'
    b'"pricing":{"prompt":"0","completion":"0"}}'
    b"]}"
)


def test_openrouter_free_pricing_requires_both_sides_zero():
    assert _openrouter_model_is_free({"prompt": "0", "completion": "0"}) is True
    assert _openrouter_model_is_free({"prompt": "0", "completion": "0.1"}) is False
    assert _openrouter_model_is_free(None) is False


def test_fetch_openrouter_free_models_keeps_nvidia_and_unknown_tools(monkeypatch):
    monkeypatch.setattr(models_mod, "_openrouter_free_cache", None)
    monkeypatch.setattr(models_mod, "_openrouter_free_cache_time", 0.0)
    monkeypatch.setattr(
        models_mod,
        "_urlopen_model_catalog_request",
        lambda *a, **k: _Resp(LIVE_FREE_PAYLOAD),
    )

    rows = fetch_openrouter_free_models(force_refresh=True)
    ids = [mid for mid, _ in rows]

    assert "nvidia/nemotron-3-ultra-550b-a55b:free" in ids
    assert "openrouter/free" in ids
    assert "free/unknown-tools" in ids
    assert "paid/model" not in ids
    assert "free/no-tools" not in ids
    assert ids == sorted(ids)
    assert all(desc == "free" for _, desc in rows)


def test_picker_runtime_slug_separates_display_from_runtime():
    assert picker_runtime_slug(OPENROUTER_FREE_PICKER_SLUG) == OPENROUTER_FREE_RUNTIME_SLUG
    assert picker_runtime_slug({
        "slug": OPENROUTER_FREE_PICKER_SLUG,
        "runtime_slug": OPENROUTER_FREE_RUNTIME_SLUG,
    }) == "openrouter"
    assert picker_runtime_slug({"slug": "openrouter"}) == "openrouter"
    assert picker_runtime_slug("anthropic") == "anthropic"


def test_list_picker_providers_emits_unique_free_row(monkeypatch):
    monkeypatch.setattr(
        model_switch,
        "list_authenticated_providers",
        lambda **kw: [
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "is_current": True,
                "is_user_defined": False,
                "models": ["anthropic/claude-sonnet-5"],
                "total_models": 1,
                "source": "built-in",
            },
            {
                "slug": OPENROUTER_FREE_PICKER_SLUG,
                "runtime_slug": OPENROUTER_FREE_RUNTIME_SLUG,
                "name": "OpenRouter Free Models",
                "is_current": False,
                "is_user_defined": False,
                "models": ["stale/id:free"],
                "total_models": 1,
                "source": "hermes",
            },
        ],
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_openrouter_models",
        lambda *a, **k: [("anthropic/claude-sonnet-5", "")],
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_openrouter_free_models",
        lambda *a, **k: [
            ("nvidia/nemotron-3-ultra-550b-a55b:free", "free"),
            ("openrouter/free", "free"),
        ],
    )

    rows = model_switch.list_picker_providers(current_provider="openrouter", max_models=1)
    slugs = [r["slug"] for r in rows]
    assert slugs.count("openrouter") == 1
    assert slugs.count(OPENROUTER_FREE_PICKER_SLUG) == 1

    free = next(r for r in rows if r["slug"] == OPENROUTER_FREE_PICKER_SLUG)
    assert free["runtime_slug"] == "openrouter"
    assert free["models"] == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/free",
    ]
    assert free["total_models"] == 2


def test_resolve_provider_full_maps_picker_slug_to_openrouter():
    from hermes_cli.providers import resolve_provider_full

    pdef = resolve_provider_full("openrouter-free")
    assert pdef is not None
    assert pdef.id == "openrouter"


def test_openrouter_free_in_curated_snapshot():
    from hermes_cli.models import OPENROUTER_MODELS

    ids = [mid for mid, _ in OPENROUTER_MODELS]
    assert "openrouter/free" in ids


def test_telegram_switch_uses_runtime_slug_not_display_slug(monkeypatch):
    import asyncio
    import sys

    if "telegram" not in sys.modules or not hasattr(sys.modules["telegram"], "__file__"):
        mod = MagicMock()
        mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
        mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
        sys.modules.setdefault("telegram", mod)
        sys.modules.setdefault("telegram.ext", mod)
        sys.modules.setdefault("telegram.constants", mod)
        sys.modules.setdefault("telegram.error", SimpleNamespace(
            NetworkError=type("NetworkError", (OSError,), {}),
            TimedOut=type("TimedOut", (OSError,), {}),
            BadRequest=type("BadRequest", (Exception,), {}),
        ))

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    seen = {}

    async def _cb(_chat_id, model_id, provider_slug):
        seen["model"] = model_id
        seen["provider"] = provider_slug
        return "ok"

    adapter._model_picker_state["1"] = {
        "providers": [{
            "slug": OPENROUTER_FREE_PICKER_SLUG,
            "runtime_slug": OPENROUTER_FREE_RUNTIME_SLUG,
            "name": "OpenRouter Free Models",
            "models": ["nvidia/nemotron-3-ultra-550b-a55b:free"],
        }],
        "selected_provider": OPENROUTER_FREE_PICKER_SLUG,
        "selected_runtime_slug": "openrouter",
        "model_list": ["nvidia/nemotron-3-ultra-550b-a55b:free"],
        "on_model_selected": _cb,
        "current_model": "x",
        "current_provider": "openrouter",
    }

    query = SimpleNamespace(
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        from_user=SimpleNamespace(id=1, first_name="t"),
        message=SimpleNamespace(chat_id=1, chat=SimpleNamespace(type="private"), message_thread_id=None),
    )
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **k: True)
    asyncio.run(adapter._handle_model_picker_callback(query, "mm:0", "1"))

    assert seen["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert seen["provider"] == "openrouter"


def test_tool_support_helper_still_drops_explicit_empty_list():
    assert _openrouter_model_supports_tools({"supported_parameters": []}) is False
    assert _openrouter_model_supports_tools({"supported_parameters": ["tools"]}) is True
    assert _openrouter_model_supports_tools({}) is True
