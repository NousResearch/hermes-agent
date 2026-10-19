"""How ``legacy_uv`` removes a pre-PM ``uv``/``uvx`` family, leaf by leaf.

Every test here drives the real cleanup against a temporary Hermes home: what
counts as removable, that deletion is anchored to a retained directory identity
instead of a mutable path, that staging/restoration never clobber another
writer's file, and that a refused or partial run reports the copy it kept. The
user-facing wording and issue buckets are covered by
``tests/hermes_cli/test_doctor_legacy_uv.py``.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess

import pytest

from hermes_cli import legacy_uv
from hermes_cli.legacy_uv import LEGACY_MANAGED_UV_NAMES, remove_legacy_managed_uv
from tests.hermes_cli.legacy_uv_support import (
    _pre_pm_binary,
    _uv,
    _uv_binary,
    _uv_binary_with_ctrl_z,
    _uvx_binary,
)

pytest_plugins = ["tests.hermes_cli.legacy_uv_support"]


@pytest.mark.platforms("posix")
def test_legacy_bin_uv_shadows_the_users_uv_until_the_cleanup(home, tmp_path):
    """End-to-end: on a launcher layout the leftover wins, after cleanup the user's does."""
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

    assert [p.name for p in remove_legacy_managed_uv(home).removed] == ["uv"]

    assert shutil.which("uv", path=composed()) == str(user_bin / "uv")
    assert str(home / "bin") in composed().split(os.pathsep)


@pytest.mark.platforms("posix")
def test_remove_legacy_uv_refuses_a_linked_bin(tmp_path, monkeypatch):
    """``$HERMES_HOME/bin`` symlinked into an ordinary user bin: ``is_file()`` and
    ``unlink()`` follow that parent, so the four-name allowlist would delete the
    user's real files. Refuse before the first unlink; the link and the external
    files all stay."""

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    outside = tmp_path / "user-bin"
    home.mkdir()
    outside.mkdir()
    for name in LEGACY_MANAGED_UV_NAMES:
        (outside / name).write_text("user-owned; preserve", encoding="utf-8")
    (home / "bin").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert remove_legacy_managed_uv(home).removed == []
    assert (home / "bin").is_symlink()
    for name in LEGACY_MANAGED_UV_NAMES:
        assert (outside / name).read_text(encoding="utf-8") == "user-owned; preserve"


@pytest.mark.platforms("posix")
def test_missing_bin_never_authorizes_a_later_path_unlink(tmp_path, monkeypatch):
    """No directory anchor at probe time means no deletion at all.

        The probe and the leaf work are not atomic: a writer can publish
        ``bin -> user-bin`` after ``_open_home_bin`` found no ``bin`` and before the
        first lookup, and both would then traverse the new parent. The callback calls
        the real acquisition and mutates the tree only afterwards, so the ordering,
        not a stubbed decision, is what the assertion exercises."""

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

    assert legacy_uv.remove_legacy_managed_uv(home).removed == []

    for name in legacy_uv.LEGACY_MANAGED_UV_NAMES:
        assert (outside / name).read_text(encoding="utf-8") == "USER-OWNED", name


@pytest.mark.platforms("posix")
def test_remove_legacy_uv_skips_the_leaf_symlink(tmp_path, monkeypatch):
    """A leaf symlink into a user's own uv is skipped with its target intact: the
    pre-PM installer staged real binaries and never links, so a ``bin/uv`` symlink
    cannot be proven Hermes's — the parent-dir link refusal is the separate
    ``bin_escapes_home`` guard, and unrelated scripts are never candidates."""

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

    removed = remove_legacy_managed_uv(home).removed

    # A leaf symlink into a user's own uv is not deletable: the pre-PM
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

    real_home = tmp_path / "real-home"
    (real_home / "bin").mkdir(parents=True)
    legacy = real_home / "bin" / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(linked_home))

    assert remove_legacy_managed_uv(linked_home).removed == [linked_home / "bin" / "uv"]
    assert not legacy.exists()


@pytest.mark.platforms("posix")
def test_remove_legacy_uv_stays_anchored_after_a_link_swap(tmp_path, monkeypatch):
    """After a successful acquisition, replacing ``bin`` with a link out of the home cannot
    redirect the unlinks: they run through the directory fd pinned at open time, so the
    external file behind the newly published link is never the target."""

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

    assert legacy_uv.remove_legacy_managed_uv(home).removed == [home / "bin" / "uv"]
    assert not (home / "bin.orig" / "uv").exists()
    assert sentinel.read_text(encoding="utf-8") == "USER-OWNED"


@pytest.mark.platforms("posix")
def test_cleanup_pins_home_before_opening_bin(tmp_path, monkeypatch):
    """The home is pinned before ``bin`` is opened, and the open is relative to that pin.

        The window is between the containment check and the ``bin`` open: a home
        replaced by a link there makes a path-based open acquire the external
        ``bin``, while ``O_NOFOLLOW`` still passes because the final component is an
        ordinary directory. The callback swaps the home after the real check."""

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

    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    for name in legacy_uv.LEGACY_MANAGED_UV_NAMES:
        assert (outside / "bin" / name).read_bytes().startswith(b"\x7fELF"), name


@pytest.mark.platforms("posix")
def test_cleanup_skips_user_own_uv_by_shape(tmp_path, monkeypatch, capsys):
    """A user's wrapper/symlink/tiny shim named ``uv``/``uvx`` is never deletable by
    name — only the astral binaries' shape is; the rest is reported for review."""

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

    assert remove_legacy_managed_uv(home).removed == []
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

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    real = bin_dir / "uv"
    _uv_binary(real)
    _uvx_binary(bin_dir / "uvx")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert sorted(p.name for p in remove_legacy_managed_uv(home).removed) == ["uv", "uvx"]
    assert not real.exists()


@pytest.mark.platforms("posix")
def test_cleanup_preserves_a_padded_non_native_wrapper(tmp_path, monkeypatch, capsys):
    """Size alone is not ownership: a shell wrapper padded above the floor lacks the
    native magic, so both the cleanup and the report-only classifier skip it."""

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "uv"
    wrapper.write_bytes(b"#!/bin/sh\n" + b"# padding\n" * (256 * 1024))  # ~2 MiB
    wrapper.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert wrapper.stat().st_size >= legacy_uv._MIN_BINARY_BYTES["uv"]

    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    assert wrapper.is_file(), "a padded wrapper is not a native binary"
    assert wrapper.read_bytes().startswith(b"#!/bin/sh\n")
    assert legacy_uv.classify_leftover(wrapper) == legacy_uv.UNPROVEN
    out = capsys.readouterr().out
    assert "Skipping" in out and "not a native uv binary" in out


@pytest.mark.platforms("posix")
def test_cleanup_preserves_an_unrelated_large_native_file(tmp_path, monkeypatch, capsys):
    """A native ``uv`` without a uv marker is not a pre-PM leftover: the gate needs
    the identity marker too, so the file survives and is never called removable."""

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    unrelated = bin_dir / "uv"
    unrelated.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))  # native, over the floor, not uv
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert unrelated.stat().st_size >= legacy_uv._MIN_BINARY_BYTES["uv"]

    assert legacy_uv.classify_leftover(unrelated) == legacy_uv.UNPROVEN
    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    assert unrelated.is_file(), "an unrelated native binary must never be deleted"
    assert unrelated.read_bytes().startswith(b"\x7fELF")
    assert "Skipping" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_cleanup_removes_a_positively_identified_legacy_artifact(tmp_path, monkeypatch):
    """The gate's acceptance half: a native binary carrying uv's marker is identified,
    reported removable, and really deleted."""

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"astral-sh/uv")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert legacy_uv.classify_leftover(legacy) == legacy_uv.REMOVABLE
    assert legacy_uv.remove_legacy_managed_uv(home).removed == [legacy]
    assert not legacy.exists()


@pytest.mark.platforms("posix")
def test_cleanup_removes_the_real_thin_uvx_launcher(tmp_path, monkeypatch):
    """The real ``uvx`` is under the ``uv`` floor and has none of uv's markers; this
    pins the per-kind floor + marker set end to end."""

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
    assert legacy_uv.remove_legacy_managed_uv(home).removed == [bin_dir / "uvx"]
    assert not (bin_dir / "uvx").exists()


@pytest.mark.platforms("posix")
def test_cleanup_preserves_an_unrelated_small_native_uvx(tmp_path, monkeypatch):
    """The lower ``uvx`` floor is not a size-only gate: a small native file named
    ``uvx`` without a marker stays ``UNPROVEN`` and survives."""

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    unrelated = bin_dir / "uvx"
    unrelated.write_bytes(b"\x7fELF" + b"\0" * (200 << 10))  # native, over the floor, not uvx
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert unrelated.stat().st_size >= legacy_uv._MIN_BINARY_BYTES["uvx"]

    assert legacy_uv.classify_leftover(unrelated) == legacy_uv.UNPROVEN
    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    assert unrelated.is_file(), "a markerless native file must never be deleted"
    assert unrelated.read_bytes().startswith(b"\x7fELF")


@pytest.mark.platforms("posix")
def test_cleanup_deletes_only_the_removable_verdict(tmp_path, monkeypatch):
    """Deletion is a POSITIVE whitelist — only ``REMOVABLE`` is unlinked, so a new or
    unexpected verdict can only under-delete."""

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

    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    assert legacy.exists(), "an unrecognized verdict must never authorize a delete"


@pytest.mark.platforms("posix")
def test_cleanup_never_unlinks_a_replacement_for_the_probed_leaf(tmp_path, monkeypatch):
    """The probe and unlink are not atomic: a file that takes the name between them
    must survive, because deletion binds to the probed inode, not the name."""

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

    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    assert legacy.is_file(), "the replacement must not be deleted"
    assert legacy.read_bytes() == replacement_bytes


@pytest.mark.platforms("posix")
def test_cleanup_never_clobbers_a_foreign_staging_name(tmp_path, monkeypatch, capsys):
    """A file already at the staging name survives: ``os.rename`` would replace it,
    so the delete refuses and leaves the leaf under its own, retryable name."""

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    _uv_binary(legacy)
    occupied = bin_dir / f".uv.hermes-cleanup-{os.getpid()}"
    occupied.write_text("USER-OWNED", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    assert occupied.read_text(encoding="utf-8") == "USER-OWNED"
    assert legacy.is_file(), "the leaf must keep its own name for a retry"
    assert "Skipping" in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_report_only_rejects_a_special_file_before_any_open(home, monkeypatch):
    """Metadata-first gate: a FIFO (whose O_RDONLY open blocks until a writer
    appears) must never reach the content opener on the report-only path."""

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
    """A readable ``.exe`` whose path and descriptor stats disagree on mode bits.

        Windows' path-based stat synthesizes execute bits for an ``.exe`` name that
        the descriptor's stat omits, so comparing the full ``st_mode`` would send a
        valid legacy ``uv.exe`` to ``UNPROBEABLE`` and hide the provenance
        diagnostic."""

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

    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    assert (bin_dir / "uv").read_bytes() == replacement, "the repopulated name must survive"
    quarantined = bin_dir / f".uv.hermes-cleanup-{os.getpid()}"
    assert quarantined.is_file(), "the moved copy must stay quarantined, never overwrite"
    assert quarantined.read_bytes().startswith(b"\x7fELF")
    out = capsys.readouterr().out
    assert "another file" in out


@pytest.mark.platforms("posix")
def test_cleanup_deletes_when_the_verdict_is_removable(tmp_path, monkeypatch):
    """The whitelist's other half: the proven shape really does delete."""

    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    legacy = bin_dir / "uv"
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert legacy_uv.remove_legacy_managed_uv(home).removed == [legacy]
    assert not legacy.exists()


@pytest.mark.platforms("windows")
def test_cleanup_skips_user_own_symlinked_uv_exe(tmp_path, monkeypatch, capsys):
    """Windows has no retained deletion anchor, so automatic cleanup fails closed:
    nothing under ``bin`` is ever unlinked, and the skip names the manual step.
    This pins the user-visible half of that posture — a user's own ``bin/uv.exe``
    SYMLINK and its large external target both survive untouched. (The POSIX twin
    proves the same preservation through the anchored leaf probes.)"""

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

    assert remove_legacy_managed_uv(home).removed == []
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

    (home / "bin" / "hermes.exe").write_bytes(b"MZ" + b"\0" * 64)

    assert remove_legacy_managed_uv(home).removed == []
    out = capsys.readouterr().out
    assert "manually" not in out, out
    assert (home / "bin" / "hermes.exe").is_file()


@pytest.mark.platforms("posix")
def test_fail_closed_cleanup_says_nothing_when_the_family_is_absent(home, capsys, monkeypatch):
    """The fail-closed branch must stay silent when there is no family to remove.

        The branch is platform policy (``automatic_cleanup``); flipping the flag
        exercises the code an uninstall runs on Windows without that host. Fail
        closed only when a family exists, or every uninstall prints "remove the uv
        family manually" about a launcher-only bin."""

    monkeypatch.setattr(legacy_uv, "automatic_cleanup", lambda: False)
    (home / "bin" / "hermes").write_text("#!/usr/bin/env sh\n", encoding="utf-8")

    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    out = capsys.readouterr().out
    assert "manually" not in out, out
    assert (home / "bin" / "hermes").is_file()

    # The family present is what the instruction exists for: it still fails closed.
    _uv_binary(home / "bin" / "uv")
    assert legacy_uv.remove_legacy_managed_uv(home).removed == []
    out = capsys.readouterr().out
    assert "manually" in out, out
    assert (home / "bin" / "uv").is_file(), "fail closed means nothing is deleted"


@pytest.mark.platforms("posix")
def test_darwin_flag_rejection_flips_the_binding(monkeypatch):
    """A macOS filesystem that refuses ``RENAME_EXCL`` answers ENOTSUP (man renameatx_np),
    not EINVAL — that errno must flip the binding too, or doctor keeps advertising a
    ``--fix`` the filesystem can never run. The Linux EINVAL/ENOSYS twin already flips."""
    import ctypes
    import errno


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


    def refuse_eperm(*_args):
        ctypes.set_errno(errno.EPERM)
        return -1

    monkeypatch.setattr(legacy_uv, "_noreplace_binding", None)
    monkeypatch.setattr(legacy_uv, "_bind_noreplace_rename", lambda: ("darwin", refuse_eperm))
    with pytest.raises(OSError) as exc:
        legacy_uv._rename_noreplace(0, "src", "dst")
    assert not isinstance(exc.value, legacy_uv._NoReplaceUnsupported)
    assert legacy_uv._noreplace_binding is not False, "a transient EPERM must not disable the binding"


@pytest.mark.platforms("windows")
def test_remove_legacy_uv_refuses_a_junctioned_bin(tmp_path, monkeypatch):
    """Native Windows twin of the linked-bin refusal: a junction into the user's bin would
    delete the external directory's real files. ``mklink /J`` needs no elevation, so this
    is provable on a real Windows host (POSIX side above). Windows cleanup fails closed
    by design — the assertion is that nothing behind the junction is ever deleted, not
    that a leaf guard classified it."""


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
        check=False,
    )
    assert linked.returncode == 0, linked.stderr.decode(errors="replace")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert remove_legacy_managed_uv(home).removed == []
    for name in LEGACY_MANAGED_UV_NAMES:
        assert (outside / name).read_text(encoding="utf-8") == "user-owned; preserve"


def test_identity_scan_reads_past_a_ctrl_z_byte(tmp_path, monkeypatch):
    """A marker past a ``0x1A`` must still be found — the descriptor is opened binary.

    Windows' CRT opens text-mode unless ``O_BINARY`` is passed, where ``os.read``
    reports EOF at the first Ctrl-Z. A real ``uv.exe`` carries one at offset 2410
    and ``UV_CACHE_DIR`` ~38 MB further in, so a text-mode scan ended 2409 bytes in
    and EVERY genuine Windows binary classified UNPROVEN: the ``hermes update``
    notice never fired, and doctor called a real legacy uv "not shaped like a pre-PM
    uv binary".

    The assertion is the same on every host; what differs is which half can fail.
    POSIX has no text mode, so this passes with or without the fix and only pins
    that a Ctrl-Z is not itself disqualifying. On Windows it is the regression
    guard: without ``O_BINARY`` the verdict is UNPROVEN and this test goes red.

    No ``sys.platform`` fake is involved — the production code path is the one that
    decides, which is also why no fixture in this suite carried a ``0x1A`` before
    (a Ctrl-Z-free shape reads to EOF under text mode too, hiding the bug).
    """
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _uv_binary_with_ctrl_z(home / "bin" / "uv")

    assert legacy_uv.classify_leftover(home / "bin" / "uv") == legacy_uv.REMOVABLE, (
        "the identity marker sits past a 0x1A byte; a text-mode read reports EOF "
        "there and the leaf is wrongly reported as not-a-uv "
        "(os.O_BINARY is missing from _open_verified's flags on Windows)"
    )


@pytest.mark.platforms("posix")
def test_a_ctrl_z_bearing_legacy_uv_is_actually_removed(tmp_path, monkeypatch):
    """End to end on the platform that can delete: found, then gone."""
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _uv_binary_with_ctrl_z(home / "bin" / "uv")

    outcome = remove_legacy_managed_uv(home)

    assert [p.name for p in outcome.removed] == ["uv"]
    assert not (home / "bin" / "uv").exists()
