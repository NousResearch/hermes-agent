"""Unit tests for per-platform model overrides (PR #7297).

Covers the helpers added in ``gateway/run.py``:

* ``_resolve_gateway_model(config, platform=...)`` — per-platform model lookup
  with fallback to ``model.default``.
* ``_resolve_platform_provider_overrides(config, platform=...)`` — per-platform
  provider bundle (``base_url``, ``api_key``).
* ``GatewayRunner._resolve_session_agent_runtime`` — central helper the PR
  pushes platform-awareness into.  Verifies:
  - ``source=None`` (memory flush / hygiene-less paths) → default model.
  - ``source=<platform>`` → platform override applied.
  - Session ``/model`` override takes precedence over platform override.
"""

import threading
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


# ── _resolve_gateway_model ──────────────────────────────────────────


def test_resolve_gateway_model_no_platform_returns_default():
    cfg = {"model": {"default": "slate-1", "platforms": {"telegram": "slate-2"}}}
    assert gateway_run._resolve_gateway_model(cfg) == "slate-1"
    assert gateway_run._resolve_gateway_model(cfg, platform=None) == "slate-1"


def test_resolve_gateway_model_string_shorthand_override():
    cfg = {"model": {"default": "slate-1", "platforms": {"telegram": "slate-2"}}}
    assert gateway_run._resolve_gateway_model(cfg, platform="telegram") == "slate-2"


def test_resolve_gateway_model_dict_override():
    cfg = {
        "model": {
            "default": "slate-1",
            "platforms": {
                "telegram": {
                    "model": "slate-2",
                    "base_url": "https://other/v1",
                    "api_key": "sk-zzz",
                }
            },
        }
    }
    assert gateway_run._resolve_gateway_model(cfg, platform="telegram") == "slate-2"


def test_resolve_gateway_model_platform_missing_falls_back_to_default():
    cfg = {"model": {"default": "slate-1", "platforms": {"telegram": "slate-2"}}}
    assert gateway_run._resolve_gateway_model(cfg, platform="discord") == "slate-1"


def test_resolve_gateway_model_platforms_key_absent():
    cfg = {"model": {"default": "slate-1"}}
    assert gateway_run._resolve_gateway_model(cfg, platform="telegram") == "slate-1"


def test_resolve_gateway_model_string_model_config():
    # model: "slate-1" (bare string, not a dict) — no platform overrides possible.
    cfg = {"model": "slate-1"}
    assert gateway_run._resolve_gateway_model(cfg) == "slate-1"
    assert gateway_run._resolve_gateway_model(cfg, platform="telegram") == "slate-1"


# ── _resolve_platform_provider_overrides ────────────────────────────


def test_resolve_platform_provider_overrides_none_without_platform():
    cfg = {"model": {"platforms": {"telegram": {"base_url": "x"}}}}
    assert gateway_run._resolve_platform_provider_overrides(cfg) is None
    assert gateway_run._resolve_platform_provider_overrides(cfg, platform=None) is None


def test_resolve_platform_provider_overrides_string_shorthand_returns_none():
    # String shorthand means "same provider, different model" — no provider bundle.
    cfg = {"model": {"platforms": {"telegram": "slate-2"}}}
    assert gateway_run._resolve_platform_provider_overrides(cfg, platform="telegram") is None


def test_resolve_platform_provider_overrides_dict_full_bundle():
    cfg = {
        "model": {
            "platforms": {
                "telegram": {
                    "model": "slate-2",
                    "base_url": "https://other/v1",
                    "api_key": "sk-zzz",
                }
            }
        }
    }
    result = gateway_run._resolve_platform_provider_overrides(cfg, platform="telegram")
    assert result == {"base_url": "https://other/v1", "api_key": "sk-zzz"}


def test_resolve_platform_provider_overrides_dict_model_only_returns_none():
    cfg = {"model": {"platforms": {"telegram": {"model": "slate-2"}}}}
    assert gateway_run._resolve_platform_provider_overrides(cfg, platform="telegram") is None


def test_resolve_platform_provider_overrides_missing_platform():
    cfg = {"model": {"platforms": {"telegram": {"base_url": "x"}}}}
    assert gateway_run._resolve_platform_provider_overrides(cfg, platform="discord") is None


# ── _resolve_session_agent_runtime ──────────────────────────────────


def _make_runner(monkeypatch, platform_base="https://default/v1", platform_api_key="sk-default"):
    """Build a minimal GatewayRunner with fake runtime resolution."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()

    def _fake_runtime_kwargs():
        return {
            "provider": "custom",
            "api_key": platform_api_key,
            "base_url": platform_base,
            "api_mode": "chat",
        }

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _fake_runtime_kwargs)
    return runner


def test_session_runtime_no_source_uses_default_model(monkeypatch):
    """Memory flush (source=None) must ignore platform overrides."""
    runner = _make_runner(monkeypatch)
    user_config = {
        "model": {
            "default": "slate-1",
            "platforms": {
                "telegram": {"model": "slate-2", "base_url": "https://hub/v1"},
            },
        }
    }
    model, runtime = runner._resolve_session_agent_runtime(user_config=user_config)
    assert model == "slate-1"
    assert runtime["base_url"] == "https://default/v1"


def test_session_runtime_platform_override_applied(monkeypatch):
    runner = _make_runner(monkeypatch)
    user_config = {
        "model": {
            "default": "slate-1",
            "platforms": {
                "telegram": {
                    "model": "slate-2",
                    "base_url": "https://hub/v1",
                    "api_key": "sk-hub",
                }
            },
        }
    }
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="tg-chat-1",
        chat_type="dm",
        user_id="other",
    )
    model, runtime = runner._resolve_session_agent_runtime(
        source=source, user_config=user_config
    )
    assert model == "slate-2"
    assert runtime["base_url"] == "https://hub/v1"
    assert runtime["api_key"] == "sk-hub"


def test_session_runtime_string_shorthand_only_swaps_model(monkeypatch):
    """String shorthand swaps the model but preserves the default provider bundle."""
    runner = _make_runner(monkeypatch)
    user_config = {
        "model": {
            "default": "slate-1",
            "platforms": {"telegram": "slate-2"},
        }
    }
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="tg-chat-1",
        chat_type="dm",
        user_id="other",
    )
    model, runtime = runner._resolve_session_agent_runtime(
        source=source, user_config=user_config
    )
    assert model == "slate-2"
    # base_url/api_key stay on the default runtime, not swapped.
    assert runtime["base_url"] == "https://default/v1"
    assert runtime["api_key"] == "sk-default"


def test_session_runtime_session_override_beats_platform_override(monkeypatch):
    """A complete session /model override must fully replace the platform override."""
    runner = _make_runner(monkeypatch)
    user_config = {
        "model": {
            "default": "slate-1",
            "platforms": {
                "telegram": {
                    "model": "slate-2",
                    "base_url": "https://hub/v1",
                    "api_key": "sk-hub",
                }
            },
        }
    }
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="tg-chat-1",
        chat_type="dm",
        user_id="other",
    )
    session_key = "agent:main:telegram:dm:other"
    runner._session_model_overrides[session_key] = {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "api_key": "sk-session",
        "base_url": "https://chatgpt/codex",
        "api_mode": "codex_responses",
    }
    model, runtime = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=user_config
    )
    assert model == "gpt-5.4"
    assert runtime["provider"] == "openai-codex"
    assert runtime["base_url"] == "https://chatgpt/codex"
    assert runtime["api_key"] == "sk-session"
    assert runtime["api_mode"] == "codex_responses"


def test_session_runtime_partial_session_override_keeps_platform_bundle(monkeypatch):
    """A session override that only sets *model* should leave the platform
    override's base_url / api_key intact."""
    runner = _make_runner(monkeypatch)
    user_config = {
        "model": {
            "default": "slate-1",
            "platforms": {
                "telegram": {
                    "model": "slate-2",
                    "base_url": "https://hub/v1",
                    "api_key": "sk-hub",
                }
            },
        }
    }
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="tg-chat-1",
        chat_type="dm",
        user_id="other",
    )
    session_key = "agent:main:telegram:dm:other"
    # Model-only override — no api_key means the "complete bundle" short-circuit
    # at line 936 does NOT fire, and we fall through to platform-aware merge.
    runner._session_model_overrides[session_key] = {"model": "slate-turbo"}
    model, runtime = runner._resolve_session_agent_runtime(
        source=source, session_key=session_key, user_config=user_config
    )
    assert model == "slate-turbo"
    assert runtime["base_url"] == "https://hub/v1"
    assert runtime["api_key"] == "sk-hub"
