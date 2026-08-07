#!/usr/bin/env python3
"""Minimal repro for voice-mode TTS playback truncation (upstream issue draft).

Demonstrates two vanilla Hermes bugs without custom command providers:

  1. speak_text and cli._voice_speak_response pre-truncate at 4000 chars on main.
  2. play_audio_file uses proc.wait(timeout=300) for ffplay on MP3/OGG on main.

This script inspects speak_text and play_audio_file source. The CLI path is
also covered by tests/tools/test_voice_cli_integration.py.

Run from repo root with the project venv:

    python scripts/repro_voice_tts_playback.py

On main you should see FAIL for both checks. After the fix branch, both PASS.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_speak_text_cap() -> tuple[bool, str]:
    from hermes_cli import voice as vm

    src = inspect.getsource(vm.speak_text)
    if "text[:4000]" in src:
        return False, "speak_text still contains text[:4000]"
    if "prepare_voice_tts_text" not in src:
        return False, "speak_text does not delegate to prepare_voice_tts_text"
    return True, "speak_text defers length cap to prepare_voice_tts_text / provider"


def check_cli_voice_speak_cap() -> tuple[bool, str]:
    import cli

    src = inspect.getsource(cli.HermesCLI._voice_speak_response)
    if "text[:4000]" in src:
        return False, "_voice_speak_response still contains text[:4000]"
    if "prepare_voice_tts_text" not in src:
        return False, "_voice_speak_response does not call prepare_voice_tts_text"
    return True, "_voice_speak_response defers length cap to prepare_voice_tts_text"


def check_playback_wait() -> tuple[bool, str]:
    from tools import voice_mode as vm

    src = inspect.getsource(vm.play_audio_file)
    if "proc.wait(timeout=300)" in src:
        return False, "play_audio_file still uses flat proc.wait(timeout=300)"
    if "_audio_file_duration_seconds" not in src:
        return False, "play_audio_file does not probe duration before ffplay wait"
    return True, "play_audio_file scales ffplay wait from probed duration"


def main() -> int:
    checks = [
        ("speak_text input cap", check_speak_text_cap),
        ("cli _voice_speak_response cap", check_cli_voice_speak_cap),
        ("ffplay wait timeout", check_playback_wait),
    ]
    failed = 0
    for name, fn in checks:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
