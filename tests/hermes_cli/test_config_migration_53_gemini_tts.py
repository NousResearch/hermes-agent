"""Migration 52→53: Google retires gemini-2.5-flash-preview-tts on 2026-11-17.

Contract: the seeded old default is dropped so the config follows the 3.8 default; a model the user
chose survives, and so does the old default when a persona_prompt_file is set (3.8 reads that file
as style, so the user migrates it) — that config gets a warning instead. Driven through
``run_migrations`` against a temp home.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml


def _run(home, gemini):
    from hermes_cli.config_migrations import run_migrations

    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"_config_version": 52, "tts": {"provider": "gemini", "gemini": gemini}}), encoding="utf-8")
    results = {"env_added": [], "config_added": [], "warnings": []}
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}, clear=False):
        run_migrations(52, results, quiet=True)
        from hermes_cli.config import load_config
        merged = load_config()["tts"]["gemini"]
    saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["tts"]["gemini"]
    return saved, merged, results["warnings"]


def test_seeded_default_follows_38_but_user_choices_survive(tmp_path):
    from hermes_cli.config_migrations import LEGACY_GEMINI_TTS_MODEL
    from tools.tts_tool_providers import DEFAULT_GEMINI_TTS_MODEL

    saved, merged, warnings = _run(tmp_path / "seeded", {"model": LEGACY_GEMINI_TTS_MODEL, "voice": "Kore",
                                                       "persona_prompt_file": ""})
    assert "model" not in saved and saved["voice"] == "Kore"
    assert merged["model"] == DEFAULT_GEMINI_TTS_MODEL == "gemini-3.8-flash-tts" and not warnings

    saved, _, warnings = _run(tmp_path / "pinned", {"model": "gemini-2.5-pro-preview-tts"})
    assert saved["model"] == "gemini-2.5-pro-preview-tts" and not warnings

    saved, _, warnings = _run(tmp_path / "persona", {"model": LEGACY_GEMINI_TTS_MODEL,
                                                    "persona_prompt_file": "~/.hermes/tts/host.md"})
    assert saved["model"] == LEGACY_GEMINI_TTS_MODEL
    assert len(warnings) == 1 and "2026-11-17" in warnings[0]
