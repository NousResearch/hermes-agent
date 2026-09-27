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

from hermes_cli import doctor


def _uv(path: Path, body: str = "hermes") -> None:
    path.write_text(f"#!/usr/bin/env sh\n# {body}\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _uv_binary(path: Path) -> None:
    """A candidate in the pre-PM installer's shape: an astral standalone binary
    (regular executable file, over the size floor) — the only form the cleanup's
    ownership gate accepts as Hermes's own."""
    path.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
    path.chmod(0o755)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with the ``bin/`` dir a launcher install publishes."""
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    return hermes_home


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
    from hermes_cli.uninstall import remove_legacy_runtime_trees

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

    remove_legacy_runtime_trees(home)

    for name in legacy_uv.LEGACY_MANAGED_UV_NAMES:
        assert (outside / name).read_text(encoding="utf-8") == "USER-OWNED", name


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
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
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
    (home / "bin" / "uv").write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
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
    # Full-size candidates: the size floor skips anything under MIN_UV_BINARY_BYTES,
    # so tiny fixtures would leave this passing even with an unpinned open.
    for root in (home, outside):
        (root / "bin").mkdir(parents=True)
        for name in legacy_uv.LEGACY_MANAGED_UV_NAMES:
            (root / "bin" / name).write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
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
    """The pre-PM installer staged astral's standalone binaries (uv + uvx, >1 MiB,
    never scripts, never symlinks) and left no receipt — so file shape is the only
    ownership evidence there is. A user's own wrapper named ``uv``, a symlink to
    their real uv, or a tiny shim is NOT deletable by name: cleanup skips it and
    reports it for manual review instead."""
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
    """The other half of the shape contract: a genuine pre-PM leftover — the astral
    pair as regular files over the size floor — is still removed, so the ownership
    gate does not turn every install into a manual chore."""
    from hermes_cli.legacy_uv import remove_legacy_managed_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    real = bin_dir / "uv"
    real.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
    (bin_dir / "uvx").write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert sorted(p.name for p in remove_legacy_managed_uv(home)) == ["uv", "uvx"]
    assert not real.exists()


@pytest.mark.platforms("posix")
def test_cleanup_deletes_only_the_removable_verdict(tmp_path, monkeypatch):
    """Deletion is a POSITIVE whitelist: only ``REMOVABLE`` is ever unlinked.

    The guard is the whole point of this module, so it must not be an exclusion
    list: a classification that grows a new skip verdict (or returns anything
    unexpected) may only under-delete. The old dispatch unlinked every verdict
    that was not one of three known skips — a fail-open regression surface.
    """
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
    monkeypatch.setenv("HERMES_HOME", str(home))

    monkeypatch.setattr(legacy_uv, "_classify", lambda st: "some-future-verdict")

    assert legacy_uv.remove_legacy_managed_uv(home) == []
    assert legacy.exists(), "an unrecognized verdict must never authorize a delete"


@pytest.mark.platforms("posix")
def test_cleanup_deletes_when_the_verdict_is_removable(tmp_path, monkeypatch):
    """The whitelist's other half: the proven shape really does delete."""
    from hermes_cli import legacy_uv

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
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

    The branch itself is platform policy (``AUTOMATIC_CLEANUP``); flipping the flag
    here exercises the same code an uninstall runs on Windows without needing that
    host, where the marked twin above covers the real thing. The rule the branch
    must hold: fail closed only when there is a family to fail closed ON, else every
    uninstall prints "remove the uv family ... manually" about a launcher-only bin.
    """
    from hermes_cli import legacy_uv

    monkeypatch.setattr(legacy_uv, "AUTOMATIC_CLEANUP", False)
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


