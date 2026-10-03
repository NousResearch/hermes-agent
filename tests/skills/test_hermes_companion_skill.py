"""Hermes Companion reads iPhone files and refuses to invent them."""

import subprocess
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[2] / "optional-skills/health/hermes-companion"


def test_skill_states_the_iphone_app_is_required():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "Hermes Companion iOS app" in text
    assert "https://hermescompanion.funktional.dev" in text
    assert "no file exists" in text
    assert "Do not invent a place" in text


def test_companion_reader_passes_offline():
    script = SKILL / "scripts/test_companion.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok script omits battery" in result.stdout
