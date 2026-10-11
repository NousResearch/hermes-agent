"""The PM runtime environment names the launch home without the sticky-profile fallback warning.

``hermes_bootstrap`` asks PM whether the venv is current (``prepare_launch`` ->
``pm.runtime_environment()``) before ``main._apply_profile_override`` re-homes the CLI to
``-p <name>`` / the sticky ``active_profile``. That PM work is install-scoped (``installs/``,
``tools/``, ``cache/uv`` all hang off ``get_default_hermes_root()``), yet it read the home through
``get_hermes_home()``, whose fallback check printed "[HERMES_HOME fallback] ... wrong profile" for
a process that was about to be re-homed and wrote nothing profile-scoped. The bootstrap runs once
per process image and the launcher ``execv``s into the store interpreter, so a plain
``hermes -p <name>`` printed it twice with one PID. Same class as ``_parser._cfg_path`` and
``apply_scratch_tmp_env`` (tests/test_hermes_home_profile_warning.py).
"""

from pathlib import Path

import pytest

import hermes_constants
from pm.runtime import runtime_environment

PROFILE = "homelab-delegator"


@pytest.fixture
def sticky_profile_home(tmp_path, monkeypatch):
    """A temp ``~/.hermes`` whose sticky ``active_profile`` names a live non-default profile."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / PROFILE
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text(f"{PROFILE}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", False)
    return root


@pytest.mark.platforms("posix")
def test_unset_home_names_the_install_root_without_warning(sticky_profile_home, capsys):
    env = runtime_environment()

    assert env["HERMES_HOME"] == str(sticky_profile_home)
    assert "HERMES_HOME fallback" not in capsys.readouterr().err


@pytest.mark.platforms("posix")
def test_each_process_image_stays_silent(sticky_profile_home, capsys):
    """The launcher re-execs into the store interpreter: two bootstraps, each with a fresh latch."""
    runtime_environment()
    hermes_constants._profile_fallback_warned = False  # what execv does to the one-shot latch
    runtime_environment()

    assert capsys.readouterr().err.count("HERMES_HOME fallback") == 0


def test_profile_home_is_forwarded_unchanged(sticky_profile_home, monkeypatch):
    profile = sticky_profile_home / "profiles" / PROFILE
    monkeypatch.setenv("HERMES_HOME", str(profile))

    assert runtime_environment()["HERMES_HOME"] == str(profile)


def test_task_override_still_wins(sticky_profile_home, tmp_path):
    other = tmp_path / "task-home"
    token = hermes_constants.set_hermes_home_override(other)
    try:
        assert runtime_environment()["HERMES_HOME"] == str(other)
    finally:
        hermes_constants.reset_hermes_home_override(token)
