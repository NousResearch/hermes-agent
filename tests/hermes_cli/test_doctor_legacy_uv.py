"""A pre-PM ``$HERMES_HOME/bin/uv*`` shadows the user's own uv and must be reported and cleared.

PM keeps its uv in the store and deliberately off PATH, so the only ``uv`` a
Hermes install should ever contribute to a shell is none. On Windows the
launcher prepends ``$HERMES_HOME/bin`` to the User PATH and the agent terminal
appends it everywhere else (#101269), so a leftover ``uv`` there is the one
every invocation resolves — and it is a version nothing maintains.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest

from hermes_cli import doctor, doctor_state


def _uv(path: Path, body: str = "hermes") -> None:
    path.write_text(f"#!/usr/bin/env sh\n# {body}\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with the ``bin/`` dir a launcher install publishes."""
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    return hermes_home


def test_doctor_reports_a_pre_pm_uv_shadow(home, capsys):
    _uv(home / "bin" / "uvx")
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert any("hermes doctor --fix" in issue for issue in finding.issues)
    assert "shadow your own uv" in capsys.readouterr().out
    assert finding.fixed == 0
    # A report must not delete: the user gets to choose how to clear it.
    assert (home / "bin" / "uvx").is_file()


def test_doctor_fix_removes_only_the_uv_family(home, monkeypatch):
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    for name in ("uv", "uvx", "uv.exe", "uvx.exe"):
        _uv(home / "bin" / name)
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")
    (home / "bin" / "mytool").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 4
    assert not finding.issues
    for name in ("uv", "uvx", "uv.exe", "uvx.exe"):
        assert not (home / "bin" / name).exists()
    # ``bin/`` holds the launchers and the user's own scripts — never those.
    assert (home / "bin" / "hermes").is_file()
    assert (home / "bin" / "mytool").is_file()


def test_doctor_fix_keeps_the_family_until_the_store_has_uv(home, monkeypatch, capsys):
    """``--fix`` must not trade the install's only uv for none: while PM's store has no uv the
    legacy binary is still what this install runs on (same guard as
    ``update_cmd_maint._purge_legacy_managed_uv``: remove only once a private
    target exists). The shadowing is still reported, so the run exits non-zero with a next step."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    _uv(home / "bin" / "uv")
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.issues
    assert (home / "bin" / "uv").is_file()
    out = capsys.readouterr().out
    assert "shadow your own uv" in out
    assert "hermes update" in finding.issues[0]


def test_doctor_is_quiet_once_the_family_is_gone(home):
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert not finding.issues
    assert finding.fixed == 0


def test_legacy_bin_uv_shadows_the_users_uv_until_the_cleanup(home, tmp_path):
    """End-to-end: on a launcher layout the leftover wins, after cleanup the user's does."""
    from hermes_cli.uninstall import remove_legacy_managed_uv
    from tools.environments.local import _append_missing_sane_path_entries, _managed_runtime_path_entries

    user_bin = tmp_path / "user-bin"
    user_bin.mkdir()
    _uv(home / "bin" / "uv", body="stale-hermes-uv")
    _uv(user_bin / "uv", body="users-own-uv")

    # The agent terminal's PATH carries $HERMES_HOME/bin for the launchers.
    assert str(home / "bin") in _managed_runtime_path_entries()

    # Windows-style launcher layout: $HERMES_HOME/bin is ahead of the user's PATH.
    def composed() -> str:
        return _append_missing_sane_path_entries(os.pathsep.join([str(home / "bin"), str(user_bin)]))

    assert shutil.which("uv", path=composed()) == str(home / "bin" / "uv")

    assert [p.name for p in remove_legacy_managed_uv(home)] == ["uv"]

    assert shutil.which("uv", path=composed()) == str(user_bin / "uv")
    assert str(home / "bin") in composed().split(os.pathsep)
