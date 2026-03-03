"""Tests for setup gating based on enabled optional integrations."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from hermes_cli.config import DEFAULT_CONFIG, get_missing_env_vars
from hermes_cli.setup import _has_any_provider_configured


def _write_config(home: Path, provider: str) -> dict:
    """Write a minimal config with the requested TTS provider."""
    home.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.setdefault("tts", {})["provider"] = provider
    (home / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    (home / ".env").write_text("", encoding="utf-8")
    return cfg


def test_enabled_only_excludes_elevenlabs_when_tts_provider_is_edge(tmp_path: Path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    cfg = _write_config(hermes_home, provider="edge")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    missing = get_missing_env_vars(required_only=False, enabled_only=True, config=cfg)
    names = {item["name"] for item in missing}

    assert "ELEVENLABS_API_KEY" not in names


def test_enabled_only_includes_elevenlabs_when_tts_provider_is_elevenlabs(
    tmp_path: Path, monkeypatch
) -> None:
    hermes_home = tmp_path / ".hermes"
    cfg = _write_config(hermes_home, provider="elevenlabs")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    missing = get_missing_env_vars(required_only=False, enabled_only=True, config=cfg)
    names = {item["name"] for item in missing}

    assert "ELEVENLABS_API_KEY" in names


def test_enabled_only_includes_openai_voice_key_when_tts_provider_is_openai(
    tmp_path: Path, monkeypatch
) -> None:
    hermes_home = tmp_path / ".hermes"
    cfg = _write_config(hermes_home, provider="openai")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)

    missing = get_missing_env_vars(required_only=False, enabled_only=True, config=cfg)
    names = {item["name"] for item in missing}

    assert "VOICE_TOOLS_OPENAI_KEY" in names


def test_provider_detection_ignores_blank_keys(tmp_path: Path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / ".env").write_text(
        "OPENROUTER_API_KEY=\nOPENAI_API_KEY=\nANTHROPIC_API_KEY=\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert _has_any_provider_configured() is False


def test_provider_detection_accepts_nonempty_key(tmp_path: Path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert _has_any_provider_configured() is True
