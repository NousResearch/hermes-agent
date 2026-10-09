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


def _uv_binary(path: Path) -> None:
    """The pre-PM ``uv`` shape: a full astral build carrying uv's own marker."""
    path.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    path.chmod(0o755)


def _uvx_binary(path: Path) -> None:
    """The real pre-PM ``uvx``: astral's thin launcher (~330 KiB, its own message,
    none of ``uv``'s markers) — the shape a ``_uv_binary`` fixture would miss."""
    path.write_bytes(
        b"\x7fELF" + b"\0" * (200 << 10)
        + b"Could not determine the location of the `uvx` binary"
    )
    path.chmod(0o755)


def _pre_pm_binary(path: Path) -> None:
    """The real pre-PM shape for *path*'s name: thin ``uvx``, full ``uv``."""
    (_uvx_binary if "uvx" in path.name.lower() else _uv_binary)(path)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with the ``bin/`` dir a launcher install publishes."""
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    return hermes_home


@pytest.mark.platforms("posix")
def test_doctor_reports_a_pre_pm_uv_shadow(home, capsys):
    _uvx_binary(home / "bin" / "uvx")
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert any("hermes doctor --fix" in issue for issue in finding.issues)
    assert "shadow your own uv" in capsys.readouterr().out
    assert finding.fixed == 0
    # A report must not delete: the user gets to choose how to clear it.
    assert (home / "bin" / "uvx").is_file()


@pytest.mark.platforms("posix")
def test_doctor_names_the_update_prerequisite_before_offering_fix(home, monkeypatch):
    """The no-``--fix`` report states the ``hermes update`` prerequisite up front while PM's
    store has no uv, instead of sending the user into a ``--fix`` that can only refuse."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    _uv_binary(home / "bin" / "uv")

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert finding.fixed == 0
    assert finding.issues
    assert "hermes update" in finding.issues[0]
    assert "hermes doctor --fix" in finding.issues[0]
    assert (home / "bin" / "uv").is_file(), "a report must never delete"


@pytest.mark.platforms("posix")
def test_doctor_keeps_an_unprobeable_uv_unresolved(home, monkeypatch):
    """A file disappearing between discovery and sizing must not abort or look clean."""
    _uv_binary(home / "bin" / "uv")
    real_lstat = os.lstat

    def refuse_lstat(path, *args, **kwargs):
        if Path(path).name == "uv":
            raise PermissionError("raced with cleanup")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", refuse_lstat)

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert finding.fixed == 0
    # ``_fail_and_issue`` records the FIX instruction, not the headline text.
    assert finding.issues
    assert "inaccessible" in finding.issues[0]


@pytest.mark.platforms("windows")
def test_doctor_keeps_an_unprobeable_uv_manual_on_windows(home, monkeypatch):
    """The same race on a fail-closed platform: automatic_cleanup() is POSIX-only,
    so a retry-``--fix`` hint would be advice no run can satisfy — the finding
    lands in the manual bucket that survives --fix instead."""
    _uv_binary(home / "bin" / "uv")
    real_lstat = os.lstat

    def refuse_lstat(path, *args, **kwargs):
        if Path(path).name == "uv":
            raise PermissionError("raced with cleanup")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", refuse_lstat)

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert finding.fixed == 0
    assert finding.manual_issues, "the manual step must survive a --fix run"
    assert "not performed on this platform" in finding.manual_issues[0]
    assert "manually" in finding.manual_issues[0]
    assert not finding.issues, "nothing here is auto-fixable on Windows"


@pytest.mark.platforms("posix")
def test_doctor_fix_removes_only_the_uv_family(home, monkeypatch):
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    for name in ("uv", "uvx", "uv.exe", "uvx.exe"):
        _pre_pm_binary(home / "bin" / name)
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


@pytest.mark.platforms("posix")
def test_doctor_fix_keeps_the_family_until_the_store_has_uv(home, monkeypatch, capsys):
    """``--fix`` must not trade the install's only uv for none: while PM's store has no uv the
    legacy binary is still what this install runs on (same guard as
    ``update_cmd_maint._notice_legacy_managed_uv``: act only once a private
    target exists). The shadowing is still reported, so the run exits non-zero with a next step."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    _uv_binary(home / "bin" / "uv")
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


@pytest.mark.platforms("posix")
def test_legacy_bin_uv_shadows_the_users_uv_until_the_cleanup(home, tmp_path):
    """End-to-end: on a launcher layout the leftover wins, after cleanup the user's does."""
    from hermes_cli.legacy_uv import remove_legacy_managed_uv
    from tools.environments.local import _append_missing_sane_path_entries, _managed_runtime_path_entries

    user_bin = tmp_path / "user-bin"
    user_bin.mkdir()
    _uv_binary(home / "bin" / "uv")
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


@pytest.mark.platforms("posix")
def test_remove_legacy_uv_refuses_a_linked_bin(tmp_path, monkeypatch):
    """``$HERMES_HOME/bin`` symlinked into an ordinary user bin: ``is_file()``/``unlink()``
    follow that PARENT, so the four-name allowlist would delete the USER's real files — the
    opposite of this helper's whole guarantee. Refuse before the first unlink; the link and
    the external files all stay."""
    from hermes_cli.legacy_uv import LEGACY_MANAGED_UV_NAMES, remove_legacy_managed_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    outside = tmp_path / "user-bin"
    home.mkdir()
    outside.mkdir()
    for name in LEGACY_MANAGED_UV_NAMES:
        (outside / name).write_text("user-owned; preserve", encoding="utf-8")
    (home / "bin").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert remove_legacy_managed_uv(home) == []
    assert (home / "bin").is_symlink()
    for name in LEGACY_MANAGED_UV_NAMES:
        assert (outside / name).read_text(encoding="utf-8") == "user-owned; preserve"


@pytest.mark.platforms("posix")
def test_missing_bin_never_authorizes_a_later_path_unlink(tmp_path, monkeypatch):
    """No directory anchor at probe time means no deletion at all.

    The probe and the leaf work are not atomic: a concurrent writer can publish
    ``bin -> user-bin`` after ``_open_home_bin`` found no ``bin`` and before the
    first ``is_file()``. Both that lookup and the unlink then traverse the new
    parent, past the containment check and the ``O_NOFOLLOW`` fd. The callback
    below calls the REAL acquisition and mutates the tree only afterwards, so
    the ordering — not a stubbed decision — is what the assertion exercises.
    """
    from hermes_cli import legacy_uv

    home, outside, store = (tmp_path / name for name in ("home", "outside", "store"))
    for directory in (home, outside, store):
        directory.mkdir()
    for name in legacy_uv.LEGACY_MANAGED_UV_NAMES:
        (outside / name).write_text("USER-OWNED", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    original = legacy_uv._open_home_bin

    def publish_after_probe(hermes_home):
        anchor = original(hermes_home)
        assert anchor is None, "the probe must find no bin/ before the link is published"
        (hermes_home / "bin").symlink_to(outside, target_is_directory=True)
        return anchor

    monkeypatch.setattr(legacy_uv, "_open_home_bin", publish_after_probe)

    assert legacy_uv.remove_legacy_managed_uv(home) == []

    for name in legacy_uv.LEGACY_MANAGED_UV_NAMES:
        assert (outside / name).read_text(encoding="utf-8") == "USER-OWNED", name


@pytest.mark.platforms("posix")
def test_doctor_does_not_name_a_linked_bin_as_removable(tmp_path, monkeypatch, capsys):
    """``$HERMES_HOME/bin`` symlinked into an ordinary user bin: doctor must not report the
    user's own uv behind that link as a pre-PM leftover to remove, and ``--fix`` must not
    touch it. The deletion side already refuses (the uninstall twin above); this asserts the
    REPORT side no longer contradicts it by telling the user to delete their own files."""
    from hermes_cli.legacy_uv import LEGACY_MANAGED_UV_NAMES

    home = tmp_path / "hermes"
    outside = tmp_path / "user-bin"
    home.mkdir()
    outside.mkdir()
    for name in LEGACY_MANAGED_UV_NAMES:
        _uv(outside / name, body="users-own-uv")
    (home / "bin").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(doctor, "HERMES_HOME", home)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert not finding.issues, "the user's own uv is not a removable finding"
    assert finding.fixed == 0
    for name in LEGACY_MANAGED_UV_NAMES:
        assert "users-own-uv" in (outside / name).read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "outside the Hermes home" in out
    assert "Remove" not in out


@pytest.mark.platforms("posix")
def test_remove_legacy_uv_skips_the_leaf_symlink(tmp_path, monkeypatch):
    """A leaf symlink into a user's own uv is skipped with its target intact: the
    pre-PM installer staged real binaries and never links, so a ``bin/uv`` symlink
    cannot be proven Hermes's — the parent-dir link refusal is the separate
    ``bin_escapes_home`` guard, and unrelated scripts are never candidates."""
    from hermes_cli.legacy_uv import remove_legacy_managed_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    user_uv = tmp_path / "user-bin" / "uv"
    user_uv.parent.mkdir()
    user_uv.write_text("users-own-uv", encoding="utf-8")
    (bin_dir / "uv").symlink_to(user_uv)
    (bin_dir / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    removed = remove_legacy_managed_uv(home)

    # A leaf symlink into a user's own uv is NOT deletable: the pre-PM
    # installer left real binaries, never symlinks, so this name cannot be
    # proven Hermes's — skip and keep the target intact.
    assert removed == []
    assert (bin_dir / "uv").is_symlink()
    assert user_uv.read_text(encoding="utf-8") == "users-own-uv"
    assert (bin_dir / "hermes").is_file()


@pytest.mark.platforms("posix")
def test_remove_legacy_uv_accepts_a_symlinked_home(tmp_path, monkeypatch):
    """A symlinked *home* is anchored, not an escape.

    ``realpath(bin)`` equals ``realpath(home)/bin``, so the cleanup still runs; the containment
    check must stay realpath identity rather than degrade into a string-prefix comparison that
    would refuse a home reached through a link."""
    from hermes_cli.legacy_uv import remove_legacy_managed_uv

    real_home = tmp_path / "real-home"
    (real_home / "bin").mkdir(parents=True)
    legacy = real_home / "bin" / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(linked_home))

    assert remove_legacy_managed_uv(linked_home) == [linked_home / "bin" / "uv"]
    assert not legacy.exists()


@pytest.mark.platforms("posix")
def test_remove_legacy_uv_stays_anchored_after_a_link_swap(tmp_path, monkeypatch):
    """After a successful acquisition, replacing ``bin`` with a link out of the home cannot
    redirect the unlinks: they run through the directory fd pinned at open time, so the
    external file behind the newly published link is never the target."""
    from hermes_cli import legacy_uv

    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "uv").write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "uv"
    sentinel.write_text("USER-OWNED", encoding="utf-8")
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    original = legacy_uv._open_home_bin

    def swap_after_acquire(hermes_home):
        anchor = original(hermes_home)
        assert anchor is not None and anchor.fd is not None
        (hermes_home / "bin").rename(hermes_home / "bin.orig")
        (hermes_home / "bin").symlink_to(outside, target_is_directory=True)
        return anchor

    monkeypatch.setattr(legacy_uv, "_open_home_bin", swap_after_acquire)

    assert legacy_uv.remove_legacy_managed_uv(home) == [home / "bin" / "uv"]
    assert not (home / "bin.orig" / "uv").exists()
    assert sentinel.read_text(encoding="utf-8") == "USER-OWNED"


@pytest.mark.platforms("posix")
def test_cleanup_pins_home_before_opening_bin(tmp_path, monkeypatch):
    """The home is pinned BEFORE ``bin`` is opened, and the open is relative to that pin.

    The window is between the containment check and the ``bin`` open: replacing the HOME
    with a link to an external directory there makes a path-based open acquire the external
    ``bin`` while ``O_NOFOLLOW`` still passes (the final component is an ordinary dir). The
    callback below swaps the home after the REAL containment check; the open must already
    be anchored to the pinned home, so the external files survive.
    """
    from hermes_cli import legacy_uv

    home = tmp_path / "home"
    outside = tmp_path / "outside"
    # Full-size candidates: the per-kind size floor skips anything under it, so
    # tiny fixtures would leave this passing even with an unpinned open.
    for root in (home, outside):
        (root / "bin").mkdir(parents=True)
        for name in legacy_uv.LEGACY_MANAGED_UV_NAMES:
            _pre_pm_binary(root / "bin" / name)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "absent-store"))
    check = legacy_uv.bin_escapes_home

    def swap_after_real_check(checked_home):
        escapes = check(checked_home)
        assert escapes is False
        checked_home.rename(tmp_path / "saved-home")
        checked_home.symlink_to(outside, target_is_directory=True)
        return escapes

    monkeypatch.setattr(legacy_uv, "bin_escapes_home", swap_after_real_check)

    assert legacy_uv.remove_legacy_managed_uv(home) == []
    for name in legacy_uv.LEGACY_MANAGED_UV_NAMES:
        assert (outside / "bin" / name).read_bytes().startswith(b"\x7fELF"), name


@pytest.mark.platforms("posix")
def test_cleanup_skips_user_own_uv_by_shape(tmp_path, monkeypatch, capsys):
    """A user's wrapper/symlink/tiny shim named ``uv``/``uvx`` is never deletable by
    name — only the astral binaries' shape is; the rest is reported for review."""
    from hermes_cli.legacy_uv import remove_legacy_managed_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    user_uv = tmp_path / "user-uv"
    user_uv.write_text("#!/bin/sh\nexec ~/.local/bin/uv-real\n", encoding="utf-8")
    (bin_dir / "uv").symlink_to(user_uv)
    wrapper = bin_dir / "uvx"
    wrapper.write_text("#!/bin/sh\nexec /opt/uv/bin/uvx-real\"$@\"\n", encoding="utf-8")
    (bin_dir / "uv.exe").write_text("MZ...", encoding="utf-8")
    shim = bin_dir / "uvx.exe"
    shim.write_text("x" * 64, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert remove_legacy_managed_uv(home) == []
    assert (bin_dir / "uv").is_symlink() and user_uv.exists()
    assert wrapper.read_text(encoding="utf-8").startswith("#!/bin/sh")
    assert (bin_dir / "uv.exe").exists()
    assert shim.exists()
    out = capsys.readouterr().out
    assert "Skipping" in out and "manually" in out


@pytest.mark.platforms("posix")
def test_cleanup_still_removes_the_real_pre_pm_shape(tmp_path, monkeypatch):
    """The other half of the shape contract: a genuine pre-PM pair is still removed,
    so the gate does not turn every install into a manual chore."""
    from hermes_cli.legacy_uv import remove_legacy_managed_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    real = bin_dir / "uv"
    _uv_binary(real)
    _uvx_binary(bin_dir / "uvx")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert sorted(p.name for p in remove_legacy_managed_uv(home)) == ["uv", "uvx"]
    assert not real.exists()


@pytest.mark.platforms("posix")
def test_cleanup_preserves_a_padded_non_native_wrapper(tmp_path, monkeypatch, capsys):
    """Size alone is not ownership: a shell wrapper padded above the floor lacks the
    native magic, so both the cleanup and the report-only classifier skip it."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "uv"
    wrapper.write_bytes(b"#!/bin/sh\n" + b"# padding\n" * (256 * 1024))  # ~2 MiB
    wrapper.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert wrapper.stat().st_size >= legacy_uv._MIN_BINARY_BYTES["uv"]

    assert legacy_uv.remove_legacy_managed_uv(home) == []
    assert wrapper.is_file(), "a padded wrapper is not a native binary"
    assert wrapper.read_bytes().startswith(b"#!/bin/sh\n")
    assert legacy_uv.classify_leftover(wrapper) == legacy_uv.UNPROVEN
    out = capsys.readouterr().out
    assert "Skipping" in out and "not a native uv binary" in out


@pytest.mark.platforms("posix")
def test_cleanup_preserves_an_unrelated_large_native_file(tmp_path, monkeypatch, capsys):
    """A native ``uv`` without a uv marker is not a pre-PM leftover: the gate needs
    the identity marker too, so the file survives and is never called removable."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    unrelated = bin_dir / "uv"
    unrelated.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))  # native, over the floor, not uv
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert unrelated.stat().st_size >= legacy_uv._MIN_BINARY_BYTES["uv"]

    assert legacy_uv.classify_leftover(unrelated) == legacy_uv.UNPROVEN
    assert legacy_uv.remove_legacy_managed_uv(home) == []
    assert unrelated.is_file(), "an unrelated native binary must never be deleted"
    assert unrelated.read_bytes().startswith(b"\x7fELF")
    assert "Skipping" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_cleanup_removes_a_positively_identified_legacy_artifact(tmp_path, monkeypatch):
    """The gate's acceptance half: a native binary carrying uv's marker is identified,
    reported removable, and really deleted."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"astral-sh/uv")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert legacy_uv.classify_leftover(legacy) == legacy_uv.REMOVABLE
    assert legacy_uv.remove_legacy_managed_uv(home) == [legacy]
    assert not legacy.exists()


@pytest.mark.platforms("posix")
def test_cleanup_removes_the_real_thin_uvx_launcher(tmp_path, monkeypatch):
    """The real ``uvx`` is under the ``uv`` floor and has none of uv's markers; this
    pins the per-kind floor + marker set end to end."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    _uvx_binary(bin_dir / "uvx")
    uvx_bytes = (bin_dir / "uvx").read_bytes()
    assert (bin_dir / "uvx").stat().st_size < legacy_uv._MIN_BINARY_BYTES["uv"]
    assert not any(m in uvx_bytes for m in legacy_uv._UV_IDENTITY_MARKERS)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert legacy_uv.classify_leftover(bin_dir / "uvx") == legacy_uv.REMOVABLE
    assert legacy_uv.remove_legacy_managed_uv(home) == [bin_dir / "uvx"]
    assert not (bin_dir / "uvx").exists()


@pytest.mark.platforms("posix")
def test_cleanup_preserves_an_unrelated_small_native_uvx(tmp_path, monkeypatch):
    """The lower ``uvx`` floor is not a size-only gate: a small native file named
    ``uvx`` without a marker stays ``UNPROVEN`` and survives."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    unrelated = bin_dir / "uvx"
    unrelated.write_bytes(b"\x7fELF" + b"\0" * (200 << 10))  # native, over the floor, not uvx
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert unrelated.stat().st_size >= legacy_uv._MIN_BINARY_BYTES["uvx"]

    assert legacy_uv.classify_leftover(unrelated) == legacy_uv.UNPROVEN
    assert legacy_uv.remove_legacy_managed_uv(home) == []
    assert unrelated.is_file(), "a markerless native file must never be deleted"
    assert unrelated.read_bytes().startswith(b"\x7fELF")


@pytest.mark.platforms("posix")
def test_cleanup_deletes_only_the_removable_verdict(tmp_path, monkeypatch):
    """Deletion is a POSITIVE whitelist — only ``REMOVABLE`` is unlinked, so a new or
    unexpected verdict can only under-delete."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    monkeypatch.setenv("HERMES_HOME", str(home))

    monkeypatch.setattr(
        legacy_uv, "_classify", lambda kind, st, magic, has_identity: "some-future-verdict"
    )

    assert legacy_uv.remove_legacy_managed_uv(home) == []
    assert legacy.exists(), "an unrecognized verdict must never authorize a delete"


@pytest.mark.platforms("posix")
def test_cleanup_never_unlinks_a_replacement_for_the_probed_leaf(tmp_path, monkeypatch):
    """The probe and unlink are not atomic: a file that takes the name between them
    must survive, because deletion binds to the probed inode, not the name."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    monkeypatch.setenv("HERMES_HOME", str(home))

    replacement_bytes = b"#!/bin/sh\n# the user's own uv\nexec /opt/uv/bin/uv \"$@\"\n"
    # Stage it separately so its inode differs (unlink+recreate could reuse it).
    staged = tmp_path / "user-uv"
    staged.write_bytes(replacement_bytes)
    staged.chmod(0o755)

    real_probe = legacy_uv._probe_leaf

    def swap_after_probe(anchor, uv_binary, uv_name):
        verdict, st = real_probe(anchor, uv_binary, uv_name)
        if verdict == legacy_uv.REMOVABLE and uv_name == "uv":
            os.replace(staged, legacy)  # a different file takes the name
        return verdict, st

    monkeypatch.setattr(legacy_uv, "_probe_leaf", swap_after_probe)

    assert legacy_uv.remove_legacy_managed_uv(home) == []
    assert legacy.is_file(), "the replacement must not be deleted"
    assert legacy.read_bytes() == replacement_bytes


@pytest.mark.platforms("posix")
def test_cleanup_never_clobbers_a_foreign_staging_name(tmp_path, monkeypatch, capsys):
    """A file already at the staging name survives: ``os.rename`` would replace it,
    so the delete refuses and leaves the leaf under its own, retryable name."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    _uv_binary(legacy)
    occupied = bin_dir / f".uv.hermes-cleanup-{os.getpid()}"
    occupied.write_text("USER-OWNED", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert legacy_uv.remove_legacy_managed_uv(home) == []
    assert occupied.read_text(encoding="utf-8") == "USER-OWNED"
    assert legacy.is_file(), "the leaf must keep its own name for a retry"
    assert "Skipping" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_report_only_rejects_a_special_file_before_any_open(home, monkeypatch):
    """Metadata-first gate: a FIFO (whose O_RDONLY open blocks until a writer
    appears) must never reach the content opener on the report-only path."""
    from hermes_cli import legacy_uv

    fifo = home / "bin" / "uv"
    os.mkfifo(fifo)

    def refuse_open(*args, **kwargs):
        raise AssertionError("report-only must not open a non-regular entry")

    monkeypatch.setattr(legacy_uv.os, "open", refuse_open)

    assert legacy_uv.classify_leftover(fifo) == legacy_uv.UNPROVEN


@pytest.mark.platforms("posix")
def test_open_verified_refuses_the_descriptor_of_a_replacement(tmp_path):
    """The opened fd must fstat back to the exact object the gate stat'd: a name
    whose inode changed between stat and open yields no descriptor."""
    from hermes_cli import legacy_uv

    leaf = tmp_path / "uv"
    leaf.write_bytes(b"a" * 16)
    st = leaf.stat()
    other = tmp_path / "other"
    other.write_bytes(b"b" * 16)
    os.replace(other, leaf)  # same name, different inode
    assert legacy_uv._open_verified(leaf, st) is None

    leaf.write_bytes(b"a" * 16)
    fd = legacy_uv._open_verified(leaf, leaf.stat())
    assert fd is not None
    os.close(fd)


@pytest.mark.platforms("windows")
def test_open_verified_accepts_a_stationary_exe(tmp_path):
    """A readable ``.exe`` fixture whose descriptor and path stats report
    different mode bits must still be inspected and classified.

    Windows' path-based stat synthesizes execute bits for an ``.exe`` name that
    the descriptor's stat omits, so comparing the full ``st_mode`` would send a
    valid legacy ``uv.exe`` to ``UNPROBEABLE`` and hide the report-only doctor's
    real provenance diagnostic."""
    from hermes_cli import legacy_uv

    exe = tmp_path / "uv.exe"
    exe.write_bytes(b"MZ" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")

    fd = legacy_uv._open_verified(exe, os.stat(exe))
    assert fd is not None, "an unchanged .exe must not be rejected by mode bits"
    os.close(fd)
    # The report-only path must reach the provenance verdict, not UNPROBEABLE.
    assert legacy_uv.classify_leftover(exe) == legacy_uv.REMOVABLE


def test_open_verified_rejects_a_type_shifted_descriptor(tmp_path, monkeypatch):
    """A descriptor whose type bits differ from the gated stat is a swap, not
    the object the caller proved regular: same device/inode/size must not be
    enough when the leaf became a symlink/dir/device at the name."""
    from hermes_cli import legacy_uv

    leaf = tmp_path / "uv"
    leaf.write_bytes(b"a" * 16)
    st = leaf.stat()
    real_fstat = os.fstat

    def type_shifted_fstat(fd):
        real = real_fstat(fd)
        return os.stat_result((
            stat.S_IFLNK | (real.st_mode & 0o7777),
            real.st_ino, real.st_dev, real.st_nlink, real.st_uid, real.st_gid,
            real.st_size, real.st_atime, real.st_mtime, real.st_ctime,
        ))

    monkeypatch.setattr(legacy_uv.os, "fstat", type_shifted_fstat)
    assert legacy_uv._open_verified(leaf, st) is None


@pytest.mark.platforms("posix")
def test_cleanup_keeps_a_repopulated_public_name_and_quarantines_the_copy(
    tmp_path, monkeypatch, capsys
):
    """Disposal fails AFTER the verified inode was moved aside while another writer
    repopulates the public name: recovery must not overwrite that new file. The
    moved copy is kept under the quarantine name for manual recovery."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    _uv_binary(bin_dir / "uv")
    monkeypatch.setenv("HERMES_HOME", str(home))
    replacement = b"#!/bin/sh\n# a brand-new, unrelated file at the public name\n"

    real_unlink = os.unlink

    def fail_disposal_with_the_name_repopulated(name, *, dir_fd=None):
        if os.path.basename(os.fsdecode(name)) == f".uv.hermes-cleanup-{os.getpid()}":
            # The leaf is staged; a different writer now owns the public name.
            (bin_dir / "uv").write_bytes(replacement)
            raise PermissionError("simulated staging unlink failure")
        return real_unlink(name, dir_fd=dir_fd) if dir_fd is not None else real_unlink(name)

    monkeypatch.setattr("hermes_cli.legacy_uv.os.unlink", fail_disposal_with_the_name_repopulated)

    assert legacy_uv.remove_legacy_managed_uv(home) == []
    assert (bin_dir / "uv").read_bytes() == replacement, "the repopulated name must survive"
    quarantined = bin_dir / f".uv.hermes-cleanup-{os.getpid()}"
    assert quarantined.is_file(), "the moved copy must stay quarantined, never overwrite"
    assert quarantined.read_bytes().startswith(b"\x7fELF")
    out = capsys.readouterr().out
    assert "another file" in out


@pytest.mark.platforms("posix")
def test_cleanup_deletes_when_the_verdict_is_removable(tmp_path, monkeypatch):
    """The whitelist's other half: the proven shape really does delete."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert legacy_uv.remove_legacy_managed_uv(home) == [legacy]
    assert not legacy.exists()


@pytest.mark.platforms("windows")
def test_cleanup_skips_user_own_symlinked_uv_exe(tmp_path, monkeypatch, capsys):
    """Windows has no retained deletion anchor, so automatic cleanup fails closed:
    nothing under ``bin`` is ever unlinked, and the skip names the manual step.
    This pins the user-visible half of that posture — a user's own ``bin/uv.exe``
    SYMLINK and its large external target both survive untouched. (The POSIX twin
    proves the same preservation through the anchored leaf probes.)"""
    from hermes_cli.legacy_uv import remove_legacy_managed_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    target = tmp_path / "custom-uv.exe"
    target.write_bytes(b"MZ" + b"\0" * (2 << 20))
    link = bin_dir / "uv.exe"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"creating a Windows symlink requires an enabled privilege: {exc}")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert remove_legacy_managed_uv(home) == []
    assert link.is_symlink()
    assert target.exists()
    out = capsys.readouterr().out
    assert "manual" in out, out


@pytest.mark.platforms("windows")
def test_doctor_names_the_manual_step_on_the_fail_closed_platform(home, capsys):
    """Windows gets the manual-removal instruction, not a retry: on a platform the
    cleanup fails closed on, a "retry --fix" hint would be a finding no fix run
    could ever clear — so it lands in the bucket that survives --fix, and the
    summary still shows it (``_print_summary`` reads issues + manual_issues)."""
    _uv_binary(home / "bin" / "uv.exe")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "the manual step must survive a --fix run"
    assert "manual" in finding.manual_issues[0]
    assert "retry" not in finding.manual_issues[0]
    assert not finding.issues, "nothing here is auto-fixable, so no auto-fix issue"
    assert (home / "bin" / "uv.exe").is_file()
    assert "shadow your own uv" in capsys.readouterr().out


@pytest.mark.platforms("windows")
def test_remove_legacy_uv_is_silent_when_the_family_is_absent(home, capsys):
    """A launcher install's ``bin/`` always exists (``hermes.exe`` and friends), and on
    Windows ``_open_home_bin`` fails closed before it can probe for the family. The
    fail-closed branch must therefore check the family first: an uninstall of a box that
    never had a legacy uv must not print "remove the uv family ... manually" about
    nothing — every Windows uninstall would carry that phantom finding."""
    from hermes_cli.legacy_uv import remove_legacy_managed_uv

    (home / "bin" / "hermes.exe").write_bytes(b"MZ" + b"\0" * 64)

    assert remove_legacy_managed_uv(home) == []
    out = capsys.readouterr().out
    assert "manually" not in out, out
    assert (home / "bin" / "hermes.exe").is_file()


@pytest.mark.platforms("posix")
def test_fail_closed_cleanup_says_nothing_when_the_family_is_absent(home, capsys, monkeypatch):
    """Pin the fail-closed branch's precondition on a host that runs everywhere.

    The branch itself is platform policy (``automatic_cleanup``); flipping the flag
    here exercises the same code an uninstall runs on Windows without needing that
    host, where the marked twin above covers the real thing. The rule the branch
    must hold: fail closed only when there is a family to fail closed ON, else every
    uninstall prints "remove the uv family ... manually" about a launcher-only bin.
    """
    from hermes_cli import legacy_uv

    monkeypatch.setattr(legacy_uv, "automatic_cleanup", lambda: False)
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    assert legacy_uv.remove_legacy_managed_uv(home) == []
    out = capsys.readouterr().out
    assert "manually" not in out, out
    assert (home / "bin" / "hermes").is_file()

    # The family present is what the instruction exists for: it still fails closed.
    _uv_binary(home / "bin" / "uv")
    assert legacy_uv.remove_legacy_managed_uv(home) == []
    out = capsys.readouterr().out
    assert "manually" in out, out
    assert (home / "bin" / "uv").is_file(), "fail closed means nothing is deleted"


@pytest.mark.platforms("posix")
def test_manual_step_lands_in_the_bucket_no_fix_run_can_clear(home, capsys, monkeypatch):
    """The fail-closed instruction is a MANUAL issue on any host that reaches that branch.

    It says "delete these yourself": no ``--fix`` run can ever clear it, so filing it
    under auto-fixable issues would keep it out of the bucket the summary labels
    "require manual intervention". The branch is platform policy, so flipping the flag
    pins the bucket split on every lane; the marked Windows test above covers the real
    gate — same wording, same bucket.
    """
    from hermes_cli import legacy_uv

    monkeypatch.setattr(legacy_uv, "automatic_cleanup", lambda: False)
    _uv_binary(home / "bin" / "uv.exe")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "the manual step must survive a --fix run"
    assert "manual" in finding.manual_issues[0]
    assert "retry" not in finding.manual_issues[0]
    assert not finding.issues, "nothing here is auto-fixable"
    assert (home / "bin" / "uv.exe").is_file()
    assert "shadow your own uv" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_manual_bucket_when_the_flag_is_rejected_at_runtime(home, monkeypatch):
    """A filesystem that rejects the no-clobber flag at runtime must stop doctor
    advertising a ``--fix`` it cannot run.

    ``automatic_cleanup()`` re-reads the rename binding; the import-time platform
    gate alone would keep saying "retry --fix" forever after the first EINVAL.
    """
    from hermes_cli import legacy_uv

    assert legacy_uv.automatic_cleanup() is (legacy_uv._bind_noreplace_rename() is not False)
    monkeypatch.setattr(legacy_uv, "_noreplace_binding", False)
    assert legacy_uv.automatic_cleanup() is False

    _uv_binary(home / "bin" / "uv.exe")
    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "the manual step must survive a --fix run"
    assert not finding.issues, "nothing here is auto-fixable"
    assert (home / "bin" / "uv.exe").is_file()


@pytest.mark.platforms("posix")
def test_darwin_flag_rejection_flips_the_binding(monkeypatch):
    """A macOS filesystem that refuses ``RENAME_EXCL`` answers ENOTSUP (man renameatx_np),
    not EINVAL — that errno must flip the binding too, or doctor keeps advertising a
    ``--fix`` the filesystem can never run. The Linux EINVAL/ENOSYS twin already flips."""
    import ctypes
    import errno

    from hermes_cli import legacy_uv

    def refuse(*_args):
        ctypes.set_errno(errno.ENOTSUP)
        return -1

    monkeypatch.setattr(legacy_uv, "_noreplace_binding", None)
    monkeypatch.setattr(legacy_uv, "_bind_noreplace_rename", lambda: ("darwin", refuse))
    with pytest.raises(legacy_uv._NoReplaceUnsupported):
        legacy_uv._rename_noreplace(0, "src", "dst")
    assert legacy_uv._noreplace_binding is False, "ENOTSUP must degrade to the manual step"


@pytest.mark.platforms("posix")
def test_transient_errno_does_not_flip_the_binding(monkeypatch):
    """EPERM is a genuine permission failure, not an unsupported flag — it must keep the
    binding usable so a retry after the permissions are fixed can still run."""
    import ctypes
    import errno

    from hermes_cli import legacy_uv

    def refuse_eperm(*_args):
        ctypes.set_errno(errno.EPERM)
        return -1

    monkeypatch.setattr(legacy_uv, "_noreplace_binding", None)
    monkeypatch.setattr(legacy_uv, "_bind_noreplace_rename", lambda: ("darwin", refuse_eperm))
    with pytest.raises(OSError) as exc:
        legacy_uv._rename_noreplace(0, "src", "dst")
    assert not isinstance(exc.value, legacy_uv._NoReplaceUnsupported)
    assert legacy_uv._noreplace_binding is not False, "a transient EPERM must not disable the binding"


@pytest.mark.platforms("posix")
def test_fix_routes_to_manual_when_the_rename_rejects_the_flag(home, monkeypatch):
    """A rename that rejects the no-clobber flag mid-run must land the SAME --fix run in
    manual_issues, not a "retry --fix" that can never succeed on this filesystem."""
    import errno

    import pm
    from hermes_cli import legacy_uv

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    _uv_binary(home / "bin" / "uv")

    def reject(*_args, **_kwargs):
        legacy_uv._noreplace_binding = False
        raise legacy_uv._NoReplaceUnsupported(errno.ENOTSUP, "ENOTSUP", "uv")

    monkeypatch.setattr(legacy_uv, "_noreplace_binding", None)
    monkeypatch.setattr(legacy_uv, "_rename_noreplace", reject)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "flag rejection must route to the manual step"
    assert not finding.issues, "a retry --fix can never succeed here"
    assert (home / "bin" / "uv").is_file()


@pytest.mark.platforms("posix")
def test_unprobeable_uv_is_manual_on_the_fail_closed_platform(home, capsys, monkeypatch):
    """An unprobeable file obeys the same platform gate as a removable one.

    On a fail-closed platform no ``--fix`` run deletes anything, so the POSIX
    "retry ``hermes doctor --fix``" wording is advice that can never be satisfied —
    exactly the finding the removable branch avoids. Before the gate, this path
    returned early into ``issues`` (the auto-fixable bucket) and told Windows users
    to retry a repair that cannot run.
    """
    from hermes_cli import legacy_uv

    monkeypatch.setattr(legacy_uv, "automatic_cleanup", lambda: False)
    _uv_binary(home / "bin" / "uv.exe")
    real_lstat = os.lstat

    def refuse_lstat(path, *args, **kwargs):
        if Path(path).name == "uv.exe":
            raise PermissionError("locked by another process")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", refuse_lstat)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 0
    assert finding.manual_issues, "no --fix run can clear this, so it is a manual issue"
    assert "manually" in finding.manual_issues[0]
    assert "retry" not in finding.manual_issues[0]
    assert not finding.issues, "nothing here is auto-fixable"
    assert (home / "bin" / "uv.exe").is_file()
    assert "could not be inspected" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_doctor_reports_the_unproven_names_beside_the_removable_ones(home, capsys):
    """One home holding BOTH a provable leftover and a too-small/symlinked name: the
    report-only run never executes the cleanup whose warning would name the unproven
    files, so the headline alone would hide them — doctor must print both buckets."""
    _uv_binary(home / "bin" / "uv.exe")
    _uv(home / "bin" / "uvx", body="tiny, not the astral binary")

    finding = doctor_state._check_legacy_uv_shadow(False)

    out = capsys.readouterr().out
    assert "not shaped like a pre-PM uv binary" in out
    assert "uvx" in out, "the unproven name must appear in the review line"
    assert finding.issues, "the removable family stays an unresolved finding"
    assert not any("uvx" in issue for issue in finding.issues), (
        "an unproven file is never a removable finding"
    )
    assert (home / "bin" / "uv.exe").is_file(), "report-only must not delete"
    assert (home / "bin" / "uvx").is_file()


@pytest.mark.platforms("posix")
def test_doctor_reports_only_the_unproven_bucket_when_nothing_is_provable(home, capsys):
    """The unproven-only path stays a clean report (no issue to fix), and the shared
    review line is the same wording the both-buckets path prints."""
    _uv(home / "bin" / "uv", body="the user's own tiny uv")

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert not finding.issues
    assert not finding.manual_issues
    out = capsys.readouterr().out
    assert "No legacy-shaped uv binaries" in out
    assert "not shaped like a pre-PM uv binary" in out
    assert (home / "bin" / "uv").is_file()


@pytest.mark.platforms("windows")
def test_remove_legacy_uv_refuses_a_junctioned_bin(tmp_path, monkeypatch):
    """Native Windows twin of the linked-bin refusal: a junction into the user's bin would
    delete the external directory's real files. ``mklink /J`` needs no elevation, so this
    is provable on a real Windows host (POSIX side above). Windows cleanup fails closed
    by design — the assertion is that nothing behind the junction is ever deleted, not
    that a leaf guard classified it."""
    import subprocess

    from hermes_cli.legacy_uv import LEGACY_MANAGED_UV_NAMES, remove_legacy_managed_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    outside = tmp_path / "user-bin"
    home.mkdir()
    outside.mkdir()
    for name in LEGACY_MANAGED_UV_NAMES:
        (outside / name).write_text("user-owned; preserve", encoding="utf-8")
    linked = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(home / "bin"), str(outside)],
        capture_output=True,
    )
    assert linked.returncode == 0, linked.stderr.decode(errors="replace")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert remove_legacy_managed_uv(home) == []
    for name in LEGACY_MANAGED_UV_NAMES:
        assert (outside / name).read_text(encoding="utf-8") == "user-owned; preserve"


@pytest.mark.platforms("posix")
def test_doctor_reports_partial_cleanup_as_unresolved(home, monkeypatch):
    """One name failing mid-family: the three that went count as fixes, the one that stayed
    remains an unresolved finding — a clean result over a live leftover is the bug."""
    import pm

    real_unlink = os.unlink

    def _fail_one(name, *, dir_fd=None):
        # Deletion uses the staging name ``.<name>.hermes-cleanup-<pid>``.
        if "uvx.exe" in os.path.basename(str(name)):
            raise PermissionError("simulated unlink failure")
        return real_unlink(name, dir_fd=dir_fd) if dir_fd is not None else real_unlink(name)

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    monkeypatch.setattr("hermes_cli.legacy_uv.os.unlink", _fail_one)
    for name in ("uv", "uvx", "uv.exe", "uvx.exe"):
        _pre_pm_binary(home / "bin" / name)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 3
    assert finding.issues, "the name that stayed is an unresolved finding"
    assert (home / "bin" / "uvx.exe").is_file()
    for name in ("uv", "uvx", "uv.exe"):
        assert not (home / "bin" / name).exists()


# --- Ownership policy: identity is WHAT, never WHO installed it -------------
# The two scenarios the review asked to be pinned separately (#124313, thread
# on legacy_uv.py:301): a copy the pre-PM installer itself migrated (it wrote
# no receipt) and an independent install of the same tool. Both are
# byte-indistinguishable from a genuine leftover, so neither report-only run
# nor any identity test may remove them — only the explicit `hermes doctor
# --fix` opt-in, after PM's own uv exists.

@pytest.mark.platforms("posix")
def test_policy_known_migrated_copy_survives_report_only(home, monkeypatch, capsys):
    """The 'known migration' case: a pre-PM uv exactly where the old installer put
    it. It matches every identity marker, so the report-only run PRESERVES it —
    identity proves what the file is, not that Hermes owns it. The finding names
    the missing receipt instead of treating the shape as provenance."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    legacy = home / "bin" / "uv"
    _uv_binary(legacy)  # exact pre-PM shape: native + uv marker, over the floor

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert legacy.is_file(), "a migrated copy survives the report-only run"
    assert finding.fixed == 0
    assert finding.issues, "the migrated copy is an unresolved finding while present"
    out = capsys.readouterr().out
    assert "shadow your own uv" in out


@pytest.mark.platforms("posix")
def test_policy_migrated_copy_removed_only_by_explicit_fix(home, monkeypatch):
    """The other half of the migrated-copy policy: removal is the explicit
    `--fix` opt-in, gated on PM's own uv existing first — never automatic."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    legacy = home / "bin" / "uv"
    _uv_binary(legacy)

    finding = doctor_state._check_legacy_uv_shadow(True)

    assert finding.fixed == 1, "only the explicit --fix decision removes the copy"
    assert not legacy.exists()


@pytest.mark.platforms("posix")
def test_policy_independent_install_is_indistinguishable_and_reported_as_own_call(home, monkeypatch, capsys):
    """The 'independent install of the same tool' case: a user's own astral uvx
    in the managed bin classifies REMOVABLE — identity CANNOT separate it from a
    genuine leftover. The policy therefore lives in the wording, not the gate:
    the report must tell the user shape alone is not provenance and deletion is
    their call, and the report-only run must never touch the file."""
    import pm

    monkeypatch.setattr(pm, "is_installed", lambda name: False)
    foreign = home / "bin" / "uvx"
    _uvx_binary(foreign)  # thin astral launcher: byte-shape of the pre-PM leftover

    from hermes_cli import legacy_uv
    assert legacy_uv.classify_leftover(foreign) == legacy_uv.REMOVABLE, (
        "identity cannot prove ownership — the policy must not pretend it can"
    )

    finding = doctor_state._check_legacy_uv_shadow(False)

    assert foreign.is_file(), "an independent install survives the report-only run"
    assert foreign.read_bytes().startswith(b"\x7fELF")
    assert finding.fixed == 0 and finding.issues
    assert any("cannot prove" in m for m in finding.issues + finding.manual_issues), (
        "the report must hand the ownership call to the user"
    )
