"""Irodori hakua.ogg voice discovery and config resolution."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def irodori_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    repo = tmp_path / "irodori-tts-server"
    voices = repo / "voices"
    voices.mkdir(parents=True)
    (voices / "hakua.ogg").write_bytes(b"OggS")
    config_path = home / "config.yaml"
    config_path.write_text(
        f"""
tts:
  provider: irodori
  irodori:
    repo_dir: {repo.as_posix()}
    voice: hakua
    base_url: http://127.0.0.1:8088
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return repo


def test_settings_default_voice_is_hakua_when_ref_exists(irodori_env):
    core = importlib.import_module("plugins.irodori_tts.core")
    cfg = core.settings()
    assert cfg.voice == "hakua"
    assert cfg.repo_dir == irodori_env


def test_list_local_voices_includes_hakua_ogg(irodori_env):
    core = importlib.import_module("plugins.irodori_tts.core")
    voices = core.list_local_voices()
    ids = {entry["id"] for entry in voices}
    assert "hakua" in ids


def test_status_payload_includes_hakua_ref_path(irodori_env, monkeypatch):
    core = importlib.import_module("plugins.irodori_tts.core")
    invoke_script = irodori_env.parent / "invoke.ps1"
    start_script = irodori_env.parent / "start.ps1"
    invoke_script.write_text("# invoke", encoding="utf-8")
    start_script.write_text("# start", encoding="utf-8")
    monkeypatch.setenv("IRODORI_TTS_INVOKE_SCRIPT", str(invoke_script))
    monkeypatch.setenv("IRODORI_TTS_START_SCRIPT", str(start_script))
    monkeypatch.setattr(core, "powershell_path", lambda: "powershell")
    monkeypatch.setattr(
        core,
        "server_health",
        lambda base_url=None, timeout=3.0: {"ok": True, "endpoint": "http://test/health"},
    )

    payload = core.status_payload()

    assert payload["defaults"]["voice"] == "hakua"
    assert payload["hakua_ref"]["present"] is True
    assert Path(payload["hakua_ref"]["path"]).name == "hakua.ogg"


def test_settings_reads_tts_irodori_config(monkeypatch, tmp_path: Path) -> None:
    core = importlib.import_module("plugins.irodori_tts.core")
    repo_dir = tmp_path / "irodori-repo"
    repo_dir.mkdir()

    tts_config = {
        "irodori": {
            "repo_dir": str(repo_dir),
            "base_url": "http://127.0.0.1:9099",
            "model": "custom-model",
            "voice": "custom-voice",
            "speed": 1.5,
            "timeout": 120,
        }
    }

    cfg = core.settings(tts_config)

    assert cfg.repo_dir == repo_dir
    assert cfg.base_url == "http://127.0.0.1:9099"
    assert cfg.model == "custom-model"
    assert cfg.voice == "custom-voice"
    assert cfg.speed == 1.5
    assert cfg.timeout == 120


def test_auto_selects_hakua_when_reference_ogg_present(tmp_path: Path) -> None:
    core = importlib.import_module("plugins.irodori_tts.core")
    repo_dir = tmp_path / "irodori"
    voices_dir = repo_dir / "voices"
    voices_dir.mkdir(parents=True)
    (voices_dir / "hakua.ogg").write_bytes(b"OggS")

    cfg = core.settings({"irodori": {"repo_dir": str(repo_dir)}})

    assert cfg.voice == core.HAKUA_VOICE_ID
