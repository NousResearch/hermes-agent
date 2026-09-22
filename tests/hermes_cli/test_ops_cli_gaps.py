"""Behavior contracts for profile-scoped ``soul`` and ``skills`` CLI writes.

These tests intentionally invoke a fresh CLI process.  Profile selection happens
before the CLI modules are imported, so an in-process call would not exercise
the same ``-p``/``HERMES_HOME`` resolution path as a user command.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_NAME = "ops"


def _run_hermes(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the real CLI against *home*, retaining the subprocess boundary."""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.fixture
def profile_homes(tmp_path: Path) -> tuple[Path, Path]:
    """Create independent default and named profiles under a temporary root."""
    default = tmp_path / "hermes"
    named = default / "profiles" / PROFILE_NAME
    named.mkdir(parents=True)

    # Pin the active profile to the default one.  The command under test must
    # still operate on the explicitly selected named profile.
    (default / "active_profile").write_text("default\n", encoding="utf-8")
    for home, label in ((default, "default"), (named, PROFILE_NAME)):
        (home / ".env").write_text("# test profile\n", encoding="utf-8")
        (home / "SOUL.md").write_text(
            f"# {label} identity\nDo not replace this profile.\n",
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(
            "model:\n"
            "  provider: test\n"
            "  default: test/model\n"
            "skills:\n"
            "  disabled: []\n"
            "  platform_disabled:\n"
            "    cli: [profile-only-cli-skill]\n",
            encoding="utf-8",
        )

    return default, named


def test_soul_set_is_scoped_to_explicit_profile(profile_homes):
    default, named = profile_homes
    default_before = (default / "SOUL.md").read_bytes()
    named_before = (named / "SOUL.md").read_bytes()

    result = _run_hermes(default, "-p", PROFILE_NAME, "soul", "set", "named identity")

    assert result.returncode == 0, result.stderr
    assert (default / "SOUL.md").read_bytes() == default_before
    named_after = (named / "SOUL.md").read_bytes()
    assert named_after != named_before
    assert b"named identity" in named_after


def test_skills_enable_disable_are_scoped_to_explicit_profile(profile_homes):
    default, named = profile_homes
    default_config_before = (default / "config.yaml").read_bytes()
    named_config_before = (named / "config.yaml").read_bytes()

    disabled = _run_hermes(
        default, "-p", PROFILE_NAME, "skills", "disable", "profile-only-skill"
    )
    assert disabled.returncode == 0, disabled.stderr
    assert (default / "config.yaml").read_bytes() == default_config_before

    before = yaml.safe_load(named_config_before)
    after_disable = yaml.safe_load((named / "config.yaml").read_text(encoding="utf-8"))
    assert "profile-only-skill" in after_disable["skills"]["disabled"]
    assert after_disable["model"] == before["model"]
    assert after_disable["skills"]["platform_disabled"] == {
        "cli": ["profile-only-cli-skill"]
    }
    assert (named / "config.yaml").read_bytes() != named_config_before

    enabled = _run_hermes(
        default, "-p", PROFILE_NAME, "skills", "enable", "profile-only-skill"
    )
    assert enabled.returncode == 0, enabled.stderr
    assert (default / "config.yaml").read_bytes() == default_config_before

    after_enable = yaml.safe_load((named / "config.yaml").read_text(encoding="utf-8"))
    assert "profile-only-skill" not in after_enable["skills"]["disabled"]
    assert after_enable["model"] == before["model"]
    assert after_enable["skills"]["platform_disabled"] == {
        "cli": ["profile-only-cli-skill"]
    }


def test_soul_set_file_writes_file_content_literally(profile_homes, tmp_path: Path):
    default, named = profile_homes
    source = tmp_path / "soul-input.md"
    literal = "# Exact input\n\n  preserve indentation\n${not_an_env_var}\n終端\n"
    source.write_text(literal, encoding="utf-8")

    result = _run_hermes(
        default, "-p", PROFILE_NAME, "soul", "set", "--file", str(source)
    )

    assert result.returncode == 0, result.stderr
    assert (named / "SOUL.md").read_bytes() == literal.encode("utf-8")


def test_soul_set_without_text_or_file_has_clear_error(profile_homes):
    default, named = profile_homes
    before = (named / "SOUL.md").read_bytes()

    result = _run_hermes(default, "-p", PROFILE_NAME, "soul", "set")

    assert result.returncode != 0
    message = f"{result.stdout}\n{result.stderr}".lower()
    assert "traceback" not in message
    assert "text" in message or "--file" in message
    assert (named / "SOUL.md").read_bytes() == before
