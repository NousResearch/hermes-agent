"""``hermes update`` tells the user about a pre-PM ``uv``/``uvx`` — on every platform.

The notice is the only surface that reaches a user who is not already running
``doctor``, so its two contracts are that it appears at all, and that the command
it prints actually names the home the file is in. Both are asserted by running the
notice and reading what a user would read.

The platform split matters (#101269): POSIX can be repaired automatically, Windows
cannot, and :func:`legacy_uv.automatic_cleanup` is POSIX-only. That must decide the
WANTED REMEDY, never whether the user hears about the shadow at all — a silent
Windows update is how the residue stayed undiscoverable in the first place.
"""
from __future__ import annotations

import pytest

from hermes_cli import legacy_uv
from hermes_constants import profile_name_for_home
from hermes_cli.update_cmd_maint import _notice_legacy_managed_uv
from tests.hermes_cli.legacy_uv_support import _pre_pm_binary, _uv_binary_with_ctrl_z

pytest_plugins = ["tests.hermes_cli.legacy_uv_support"]


@pytest.fixture
def store_has_uv(monkeypatch):
    """PM's store already carries a uv, so the legacy copy is removable."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)


def test_the_notice_stays_silent_while_the_legacy_copy_is_the_only_uv(home, monkeypatch, capsys, store_has_uv):
    """Naming a file as removable while it is the install's only uv invites a broken tree."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    _pre_pm_binary(home / "bin" / "uv")

    _notice_legacy_managed_uv()

    assert "shadow your own uv" not in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_the_notice_names_a_command_that_targets_the_reported_home(
        home, monkeypatch, capsys, store_has_uv):
    """On a platform that can fix it, the printed command must be runnable and scoped.

    Doctor resolves ONE home, so the advice has to name the profile it was found
    in. ``-p`` is the only scope spelling every shell accepts: an ``HERMES_HOME=``
    prefix is POSIX-only, and the default profile — the case the notice hits first
    on Windows — is exactly where that prefix cannot run.
    """
    # Make this temp home the platform default, so ``-p default`` is the truthful scope.
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_constants._default_hermes_root_memo", None)
    _pre_pm_binary(home / "bin" / "uv")

    _notice_legacy_managed_uv()

    out = capsys.readouterr().out
    assert "shadow your own uv" in out, "the leftover must be reported"
    assert "doctor --fix" in out, "an automatic platform must offer the automatic repair"
    assert "hermes -p default doctor --fix" in out, (
        f"the printed command does not name the profile the file is in: {out}"
    )
    assert "HERMES_HOME=" not in out, "a POSIX-only env prefix is not runnable everywhere"


@pytest.mark.platforms("posix")
def test_the_notice_scopes_advice_to_a_named_profile(home, monkeypatch, capsys, store_has_uv):
    """A leftover in another profile needs ``-p``: an unscoped ``--fix`` never sees it."""
    named = home / "profiles" / "work"
    (named / "bin").mkdir(parents=True)
    _pre_pm_binary(named / "bin" / "uv")
    _pre_pm_binary(home / "bin" / "uv")
    # Discovered profiles are HOME-anchored, not HERMES_HOME-anchored, so the walk
    # has to be handed this one; the path shape is what names it.
    import hermes_cli.profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "list_profiles",
                        lambda **_: [type("P", (), {"path": named, "name": "work"})()])

    _notice_legacy_managed_uv()

    out = capsys.readouterr().out
    assert "hermes -p work doctor --fix" in out, f"the named profile is not scoped: {out}"


@pytest.mark.platforms("posix")
def test_the_notice_leaves_a_custom_root_unscoped(home, monkeypatch, capsys, store_has_uv):
    """``profile_name_for_home`` calls any custom root "default", but a fresh shell
    resolves that name to the REAL default root — so the advice must not use it.

    The home is the caller's own HERMES_HOME here: ``hermes doctor --fix`` in the
    shell that ran the update already resolves it.
    """
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home",
                        lambda: home.parent / "not-this-home")
    monkeypatch.setattr("hermes_constants._default_hermes_root_memo", None)
    _pre_pm_binary(home / "bin" / "uv")

    _notice_legacy_managed_uv()

    out = capsys.readouterr().out
    assert "shadow your own uv" in out
    assert "hermes doctor --fix" in out
    assert "hermes -p default" not in out, (
        f"a custom root is not the default profile; -p default would target another home: {out}"
    )
    assert "HERMES_HOME=" not in out


@pytest.mark.platforms("posix")
def test_the_notice_says_so_when_the_platform_cannot_fix_it(home, monkeypatch, capsys, store_has_uv):
    """A fail-closed platform must still be told, and told the manual step.

    ``automatic_cleanup()`` is False on Windows (no directory-relative deletion),
    so the remedy there is manual. Silence is not the remedy: it is the reason the
    shadow went unnoticed. Simulated by forcing the capability verdict, which is
    the same switch the real call site reads.
    """
    monkeypatch.setattr(legacy_uv, "automatic_cleanup", lambda: False)
    _pre_pm_binary(home / "bin" / "uv")

    _notice_legacy_managed_uv()

    out = capsys.readouterr().out
    assert "shadow your own uv" in out, "a platform that cannot auto-fix must still report"
    assert "not performed on this platform" in out, (
        f"the notice must not imply a --fix that can only repeat itself: {out}"
    )
    assert "by hand" in out, "the manual remedy must be stated"
    assert "doctor --fix" not in out, "no automatic repair exists here to offer"


@pytest.mark.platforms("posix")
def test_the_notice_is_quiet_when_there_is_nothing_to_report(home, capsys, store_has_uv):
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    _notice_legacy_managed_uv()

    assert "shadow your own uv" not in capsys.readouterr().out


def test_the_notice_never_fails_the_update_tail(home, monkeypatch, capsys):
    """A notice is best-effort: a damaged install must not turn an update into a failure."""
    import pm

    def boom(name):
        raise ValueError("pm/lock.json is corrupt")

    monkeypatch.setattr(pm, "is_installed", boom)
    _pre_pm_binary(home / "bin" / "uv")

    _notice_legacy_managed_uv()  # must not raise

    assert "shadow your own uv" not in capsys.readouterr().out


def test_the_notice_survives_a_ctrl_z_inside_the_binary(home, monkeypatch, capsys, store_has_uv):
    """The notice's gate is ``classify_leftover(...) == REMOVABLE``, so anything that
    makes a real binary classify UNPROVEN silently kills this surface.

    Windows' CRT reads text-mode without ``O_BINARY`` and reports EOF at the first
    ``0x1A``; a real ``uv.exe`` has one at offset 2410 with its identity marker
    ~38 MB further in. That is exactly the state a Windows user was in: a genuine
    legacy uv on PATH, and ``hermes update`` printing nothing at all.

    The fixture carries that byte, so the notice must fire on every host — on
    POSIX because the scan simply runs to the end, on Windows because the
    descriptor is opened binary.
    """
    _uv_binary_with_ctrl_z(home / "bin" / "uv")

    _notice_legacy_managed_uv()

    out = capsys.readouterr().out
    assert "shadow your own uv" in out, (
        "a real uv.exe shape (marker past a 0x1A) produced no notice; the "
        "identity scan stopped at the Ctrl-Z"
    )
