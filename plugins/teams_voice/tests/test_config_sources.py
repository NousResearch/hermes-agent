"""Both config sources work: config.yaml block precedence + env fallback."""

from __future__ import annotations

import pytest

from plugins.teams_voice import config as cfgmod
from plugins.teams_voice.realtime import openai_client as rc


def test_resolve_config_prefers_block_then_env(monkeypatch):
    monkeypatch.setenv("TEAMS_VOICE_SHARED_SECRET", "from-env")
    monkeypatch.setenv("TEAMS_VOICE_PORT", "9999")
    # config.yaml block wins for the keys it sets...
    block = {"shared_secret": "from-config", "host": "0.0.0.0"}
    cfg = cfgmod.resolve_config(extra=block)
    assert cfg.shared_secret == "from-config"  # block beats env
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9999  # ...and env fills the rest


def test_resolve_config_env_only_when_block_empty(monkeypatch):
    monkeypatch.setenv("TEAMS_VOICE_SHARED_SECRET", "env-secret")
    cfg = cfgmod.resolve_config(extra={})
    assert cfg.shared_secret == "env-secret"


def test_realtime_block_overrides_env(monkeypatch):
    monkeypatch.setenv("TEAMS_VOICE_REALTIME_VOICE", "cedar")  # env
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "envkey")
    block = {
        "backend": "azure",
        "azure_endpoint": "https://res.cognitiveservices.azure.com",
        "azure_deployment": "gpt-realtime",
        "azure_api_version": "2025-04-01-preview",
        "voice": "verse",  # config.yaml beats the env voice
        "api_key": "blockkey",
    }
    cfg = rc.realtime_config_from_env(block=block)
    assert cfg.voice == "verse"
    assert cfg.api_key == "blockkey"
    assert cfg.api_key_header == "api-key"
    assert cfg.base_url.endswith("deployment=gpt-realtime")
