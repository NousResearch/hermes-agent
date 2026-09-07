"""Regression tests for #34107 — Docker UID/GID handling in ensure_hermes_home.

When Hermes runs in Docker with ``HERMES_UID=1000`` / ``HERMES_GID=911``,
the entrypoint chowns the top-level ``HERMES_HOME`` once at startup. But
subdirectories created at runtime by ``ensure_hermes_home()`` — especially
for profile namespaces under ``profiles/<name>/`` spawned by kanban
workers — were landing as ``root:root`` and blocking subsequent
uid-mapped worker invocations with ``PermissionError [Errno 13]``.

The fix is a ``_chown_to_hermes_uid`` helper that reads the env vars and
applies chown after ``mkdir``, invoked from ``_secure_dir`` (which already
runs after every directory creation in the home-init path).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# _resolve_hermes_uid_gid
# ---------------------------------------------------------------------------


class TestResolveHermesUidGid:
    def test_returns_parsed_values_when_both_set(self, monkeypatch):
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        from hermes_cli.config import _resolve_hermes_uid_gid
        uid, gid = _resolve_hermes_uid_gid()
        assert uid == 1000
        assert gid == 911


    # ``windows_only`` rather than ``skipif(sys.platform != "win32")``: the
    # Windows CI job selects ``-m windows_only``, so a bare skipif would leave
    # this test skipped on Linux AND unselected on the Windows lane — dead on
    # every host.
    @pytest.mark.windows_only
    def test_windows_returns_none_none(self, monkeypatch):
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        from hermes_cli.config import _resolve_hermes_uid_gid
        uid, gid = _resolve_hermes_uid_gid()
        assert uid is None
        assert gid is None


# ---------------------------------------------------------------------------
# _chown_to_hermes_uid
# ---------------------------------------------------------------------------


class TestChownToHermesUid:
    def test_calls_os_chown_when_both_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        from hermes_cli import config as cfg

        d = tmp_path / "subdir"
        d.mkdir()

        with patch.object(cfg.os, "chown") as mock_chown:
            cfg._chown_to_hermes_uid(d)
        mock_chown.assert_called_once_with(d, 1000, 911)


    def test_eperm_is_silently_swallowed(self, tmp_path, monkeypatch):
        """When running as non-root, os.chown raises EPERM. That's fine —
        the entrypoint's startup chown -R will pick it up on restart, and
        in most cases the dir was already correctly-owned by the calling
        user anyway."""
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        from hermes_cli import config as cfg

        d = tmp_path / "subdir"
        d.mkdir()

        def _raises_eperm(*args, **kwargs):
            raise PermissionError("operation not permitted")

        with patch.object(cfg.os, "chown", side_effect=_raises_eperm):
            # Must not raise — the catch is non-fatal.
            cfg._chown_to_hermes_uid(d)

    def test_attributeerror_swallowed_for_windows_compat(self, tmp_path, monkeypatch):
        """os.chown doesn't exist on Windows. Catching AttributeError keeps
        the helper portable."""
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        from hermes_cli import config as cfg

        d = tmp_path / "subdir"
        d.mkdir()

        with patch.object(cfg.os, "chown", side_effect=AttributeError("no chown on this platform")):
            cfg._chown_to_hermes_uid(d)  # must not raise


# ---------------------------------------------------------------------------
# End-to-end: _secure_dir now also chowns
# ---------------------------------------------------------------------------


class TestSecureDirChown:
    @pytest.mark.skipif(sys.platform == "win32", reason="chown is no-op on Windows")
    def test_secure_dir_invokes_chown_when_env_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        from hermes_cli import config as cfg

        d = tmp_path / "subdir"
        d.mkdir()

        with patch.object(cfg.os, "chown") as mock_chown:
            cfg._secure_dir(d)
        mock_chown.assert_called_once_with(d, 1000, 911)

    @pytest.mark.skipif(sys.platform == "win32", reason="chown is no-op on Windows")
    def test_secure_dir_no_chown_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_UID", raising=False)
        monkeypatch.delenv("HERMES_GID", raising=False)
        from hermes_cli import config as cfg

        d = tmp_path / "subdir"
        d.mkdir()

        with patch.object(cfg.os, "chown") as mock_chown:
            cfg._secure_dir(d)
        mock_chown.assert_not_called()


class TestBrokenSymlinkSubdirs:
    """#104771: a subdir under HERMES_HOME that is a symlink to a target that
    does not exist yet (dotfiles-managed: GNU Stow / Chezmoi / git repos) used
    to crash ensure_hermes_home() with FileExistsError — pathlib's
    mkdir(exist_ok=True) treats the broken link as "exists, not a dir" and
    re-raises EEXIST."""

    def test_broken_symlink_subdir_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        target = tmp_path / "external-repo" / "hooks"  # does not exist yet
        (tmp_path / "hooks").symlink_to(target)

        from hermes_cli.config import ensure_hermes_home

        ensure_hermes_home()

        # The link is preserved and its target directory now exists; the
        # rest of the skeleton is created as usual.
        assert (tmp_path / "hooks").is_symlink()
        assert (tmp_path / "hooks").resolve().is_dir()
        assert (tmp_path / "skills").is_dir()
        assert (tmp_path / "sessions").is_dir()

    def test_healthy_symlink_subdir_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        target = tmp_path / "real-hooks"
        target.mkdir()
        (tmp_path / "hooks").symlink_to(target)

        from hermes_cli.config import ensure_hermes_home

        ensure_hermes_home()

        assert (tmp_path / "hooks").is_symlink()
        assert (tmp_path / "hooks").resolve() == target

    def test_plain_subdirs_still_secured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from hermes_cli.config import ensure_hermes_home

        ensure_hermes_home()

        mode = (tmp_path / "skills").stat().st_mode & 0o777
        assert mode == 0o700
