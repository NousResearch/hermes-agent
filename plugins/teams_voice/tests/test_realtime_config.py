"""Tests for realtime backend config resolution (OpenAI vs Azure)."""

from __future__ import annotations

import pytest

from plugins.teams_voice.realtime import openai_client as rc

_ENV_KEYS = [
    "TEAMS_VOICE_REALTIME_BACKEND",
    "TEAMS_VOICE_REALTIME_URL",
    "TEAMS_VOICE_AZURE_ENDPOINT",
    "TEAMS_VOICE_AZURE_DEPLOYMENT",
    "TEAMS_VOICE_AZURE_API_VERSION",
    "TEAMS_VOICE_REALTIME_VOICE",
    "TEAMS_VOICE_REALTIME_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_FOUNDRY_API_KEY",
    "OPENAI_API_KEY",
    "TEAMS_VOICE_REALTIME_MODEL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_azure_builds_proven_url(monkeypatch):
    monkeypatch.setenv("TEAMS_VOICE_REALTIME_BACKEND", "azure")
    monkeypatch.setenv("TEAMS_VOICE_AZURE_ENDPOINT", "https://pcfcaoai2.cognitiveservices.azure.com")
    monkeypatch.setenv("TEAMS_VOICE_AZURE_DEPLOYMENT", "gpt-realtime")
    monkeypatch.setenv("TEAMS_VOICE_AZURE_API_VERSION", "2025-04-01-preview")
    monkeypatch.setenv("TEAMS_VOICE_REALTIME_VOICE", "cedar")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k123")

    cfg = rc.realtime_config_from_env(block={})
    assert cfg.configured
    assert cfg.api_key_header == "api-key"
    assert cfg.voice == "cedar"
    assert cfg.api_key == "k123"  # reused from AZURE_FOUNDRY_API_KEY
    assert cfg.base_url == (
        "wss://pcfcaoai2.cognitiveservices.azure.com/openai/realtime"
        "?api-version=2025-04-01-preview&deployment=gpt-realtime"
    )


def test_azure_explicit_url_passthrough(monkeypatch):
    monkeypatch.setenv("TEAMS_VOICE_REALTIME_URL", "wss://x.openai.azure.com/openai/realtime?foo=1")
    monkeypatch.setenv("TEAMS_VOICE_REALTIME_API_KEY", "k")
    cfg = rc.realtime_config_from_env(block={})
    assert cfg.api_key_header == "api-key"  # azure.com detected
    assert cfg.base_url == "wss://x.openai.azure.com/openai/realtime?foo=1"


def test_openai_default_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = rc.realtime_config_from_env(block={})
    assert cfg.api_key_header == "Authorization"
    assert cfg.base_url == rc.DEFAULT_BASE_URL
    assert cfg.api_key == "sk-test"


def test_unconfigured_when_no_key(monkeypatch):
    cfg = rc.realtime_config_from_env(block={})
    assert not cfg.configured
