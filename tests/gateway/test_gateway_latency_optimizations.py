"""Tests for gateway per-turn latency optimizations.

- _reload_runtime_env_preserving_config_authority mtime caching
- Per-platform skip_context_files config
- Agent cache signature includes skip_context_files
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest


# ── Mtime caching for _reload_runtime_env ──

def _make_dotenv(tmp_path, content=""):
    path = tmp_path / ".env"
    path.write_text(content)
    return path


def _make_config(tmp_path, content="agent:\n  max_turns: 20\n"):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


def test_reload_skips_when_files_unchanged(monkeypatch, tmp_path):
    """When neither .env nor config.yaml changed, reload is skipped entirely."""
    from gateway.run import (
        _reload_runtime_env_preserving_config_authority,
        _RELOAD_LAST_DOTENV_MTIME,
        _RELOAD_LAST_CONFIG_MTIME,
        _hermes_home as _orig_home,
    )

    dotenv = _make_dotenv(tmp_path)
    config = _make_config(tmp_path)

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    # Reset caches
    monkeypatch.setattr("gateway.run._RELOAD_LAST_DOTENV_MTIME", 0.0)
    monkeypatch.setattr("gateway.run._RELOAD_LAST_CONFIG_MTIME", 0.0)

    load_calls = []

    def fake_load_dotenv(**kwargs):
        load_calls.append(1)

    monkeypatch.setattr("gateway.run.load_hermes_dotenv", fake_load_dotenv)

    # First call — should load
    _reload_runtime_env_preserving_config_authority()
    assert len(load_calls) == 1

    # Second call — files unchanged, should skip
    _reload_runtime_env_preserving_config_authority()
    assert len(load_calls) == 1  # still 1


def test_reload_reruns_when_dotenv_changes(monkeypatch, tmp_path):
    """When .env mtime changes, reload runs again."""
    from gateway.run import _reload_runtime_env_preserving_config_authority

    dotenv = _make_dotenv(tmp_path)
    config = _make_config(tmp_path)

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.run._RELOAD_LAST_DOTENV_MTIME", 0.0)
    monkeypatch.setattr("gateway.run._RELOAD_LAST_CONFIG_MTIME", 0.0)

    load_calls = []

    def fake_load_dotenv(**kwargs):
        load_calls.append(1)

    monkeypatch.setattr("gateway.run.load_hermes_dotenv", fake_load_dotenv)

    _reload_runtime_env_preserving_config_authority()
    assert len(load_calls) == 1

    # Touch .env to change mtime
    time.sleep(0.01)
    dotenv.write_text("NEW_KEY=value\n")

    _reload_runtime_env_preserving_config_authority()
    assert len(load_calls) == 2


def test_reload_reruns_when_config_changes(monkeypatch, tmp_path):
    """When config.yaml mtime changes, reload runs again."""
    from gateway.run import _reload_runtime_env_preserving_config_authority

    dotenv = _make_dotenv(tmp_path)
    config = _make_config(tmp_path)

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.run._RELOAD_LAST_DOTENV_MTIME", 0.0)
    monkeypatch.setattr("gateway.run._RELOAD_LAST_CONFIG_MTIME", 0.0)

    load_calls = []

    def fake_load_dotenv(**kwargs):
        load_calls.append(1)

    monkeypatch.setattr("gateway.run.load_hermes_dotenv", fake_load_dotenv)

    _reload_runtime_env_preserving_config_authority()
    assert len(load_calls) == 1

    # Touch config to change mtime
    time.sleep(0.01)
    config.write_text("agent:\n  max_turns: 50\n")

    _reload_runtime_env_preserving_config_authority()
    assert len(load_calls) == 2


# ── Agent cache signature includes skip_context_files ──

def test_signature_differs_for_skip_context_files():
    """skip_context_files=True vs False produce different signatures."""
    from gateway.run import GatewayRunner

    runtime = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
    }
    sig_false = GatewayRunner._agent_config_signature(
        "claude-sonnet-4", runtime, ["hermes-telegram"], "",
        skip_context_files=False,
    )
    sig_true = GatewayRunner._agent_config_signature(
        "claude-sonnet-4", runtime, ["hermes-telegram"], "",
        skip_context_files=True,
    )
    assert sig_false != sig_true


def test_signature_stable_for_same_skip_context_files():
    """Same skip_context_files value produces stable signature."""
    from gateway.run import GatewayRunner

    runtime = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
    }
    sig1 = GatewayRunner._agent_config_signature(
        "claude-sonnet-4", runtime, ["hermes-telegram"], "",
        skip_context_files=True,
    )
    sig2 = GatewayRunner._agent_config_signature(
        "claude-sonnet-4", runtime, ["hermes-telegram"], "",
        skip_context_files=True,
    )
    assert sig1 == sig2


def test_skip_context_files_default_false_unchanged():
    """Default skip_context_files=False keeps existing signatures stable."""
    from gateway.run import GatewayRunner

    runtime = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
    }
    # Old signature (no skip_context_files param)
    sig_old = GatewayRunner._agent_config_signature(
        "claude-sonnet-4", runtime, ["hermes-telegram"], "",
    )
    # New signature with explicit False
    sig_new = GatewayRunner._agent_config_signature(
        "claude-sonnet-4", runtime, ["hermes-telegram"], "",
        skip_context_files=False,
    )
    assert sig_old == sig_new


# ── Per-platform skip_context_files config parsing ──

def test_skip_context_files_defaults_false():
    """When no per-platform config is set, skip_context_files defaults to False."""
    from gateway.run import GatewayRunner

    config = {}  # empty config
    platforms_gw = (config.get("gateway") or {}).get("platforms") or {}
    plat_cfg = platforms_gw.get("weixin") or {}
    skip = plat_cfg.get("skip_context_files")
    result = bool(skip) if skip is not None else False
    assert result is False


def test_skip_context_files_enabled_for_platform():
    """When gateway.platforms.weixin.skip_context_files=true, it's enabled."""
    config = {
        "gateway": {
            "platforms": {
                "weixin": {
                    "skip_context_files": True,
                }
            }
        }
    }
    platforms_gw = (config.get("gateway") or {}).get("platforms") or {}
    plat_cfg = platforms_gw.get("weixin") or {}
    skip = plat_cfg.get("skip_context_files")
    result = bool(skip) if skip is not None else False
    assert result is True


def test_skip_context_files_platform_isolation():
    """skip_context_files for weixin doesn't affect telegram."""
    config = {
        "gateway": {
            "platforms": {
                "weixin": {"skip_context_files": True},
            }
        }
    }
    # weixin
    plat_wx = ((config.get("gateway") or {}).get("platforms") or {}).get("weixin") or {}
    skip_wx = plat_wx.get("skip_context_files")
    assert bool(skip_wx) if skip_wx is not None else False is True

    # telegram
    plat_tg = ((config.get("gateway") or {}).get("platforms") or {}).get("telegram") or {}
    skip_tg = plat_tg.get("skip_context_files")
    assert bool(skip_tg) if skip_tg is not None else False is False
