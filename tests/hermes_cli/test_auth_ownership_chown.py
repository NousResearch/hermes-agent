"""Tests for auth.json ownership remediation (issue #15718).

When `hermes login` runs as root inside a container (e.g. via `docker exec`)
but the gateway runs as a non-root user, the freshly-written ``auth.json``
ends up owned by ``root:root`` and the gateway can no longer read it,
surfacing as a misleading "no credentials stored" error.

These tests cover:

* The shared :func:`utils.chown_to_match_parent` helper:
  - Chowns to the parent's UID/GID when running as root and ownerships differ.
  - Does nothing when running as a non-root user.
  - Does nothing when the file already matches the parent's owner.
  - Is a safe no-op on platforms without ``os.chown`` (Windows).
  - Swallows ``OSError`` from ``chown`` so a remediation failure never
    breaks the surrounding write.
* :func:`hermes_cli.auth._save_auth_store` invokes the helper after writing.
* :func:`hermes_cli.auth._save_qwen_cli_tokens` invokes the helper after
  writing.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

import utils
from hermes_cli import auth as auth_module


# ---------------------------------------------------------------------------
# utils.chown_to_match_parent
# ---------------------------------------------------------------------------


class TestChownToMatchParent:
    def test_chowns_when_root_and_owners_differ(self, tmp_path):
        target = tmp_path / "auth.json"
        target.write_text("{}", encoding="utf-8")

        parent_stat = os.stat(tmp_path)
        file_stat = os.stat(target)

        # Simulate: root caller (geteuid==0), parent owned by uid 10000,
        # file currently owned by uid 0.
        fake_parent = type(parent_stat)(
            (
                parent_stat.st_mode,
                parent_stat.st_ino,
                parent_stat.st_dev,
                parent_stat.st_nlink,
                10000,            # parent uid
                10000,            # parent gid
                parent_stat.st_size,
                parent_stat.st_atime,
                parent_stat.st_mtime,
                parent_stat.st_ctime,
            )
        )
        fake_file = type(file_stat)(
            (
                file_stat.st_mode,
                file_stat.st_ino,
                file_stat.st_dev,
                file_stat.st_nlink,
                0,                # file uid (root)
                0,                # file gid (root)
                file_stat.st_size,
                file_stat.st_atime,
                file_stat.st_mtime,
                file_stat.st_ctime,
            )
        )

        recorded = {}

        def fake_stat(path):
            # Parent stat path (real os.stat is patched to this).
            if Path(path) == tmp_path:
                return fake_parent
            return fake_file

        def fake_lstat(path):
            # File stat path (helper now uses lstat for the target).
            return fake_file

        def fake_chown(path, uid, gid, **kwargs):
            recorded["args"] = (Path(path), uid, gid)
            recorded["kwargs"] = kwargs

        with patch.object(utils.os, "geteuid", create=True, return_value=0), \
             patch.object(utils.os, "chown", create=True, side_effect=fake_chown) as chown_mock, \
             patch.object(utils.os, "lstat", side_effect=fake_lstat), \
             patch.object(utils.os, "stat", side_effect=fake_stat), \
             patch.object(utils.os, "supports_follow_symlinks", {chown_mock}):
            assert utils.chown_to_match_parent(target) is True

        assert recorded["args"] == (target, 10000, 10000)
        # Symlink-safety: chown must be invoked with follow_symlinks=False.
        assert recorded["kwargs"].get("follow_symlinks") is False

    def test_noop_when_not_root(self, tmp_path):
        target = tmp_path / "auth.json"
        target.write_text("{}", encoding="utf-8")

        recorded = {"called": False}

        def fake_chown(*_args, **_kwargs):
            recorded["called"] = True

        with patch.object(utils.os, "geteuid", create=True, return_value=10000), \
             patch.object(utils.os, "chown", create=True, side_effect=fake_chown):
            assert utils.chown_to_match_parent(target) is False

        assert recorded["called"] is False

    def test_noop_when_owners_already_match(self, tmp_path):
        target = tmp_path / "auth.json"
        target.write_text("{}", encoding="utf-8")

        recorded = {"called": False}

        def fake_chown(*_args, **_kwargs):
            recorded["called"] = True

        # Real stat — parent and file both owned by the test runner.
        with patch.object(utils.os, "geteuid", create=True, return_value=0), \
             patch.object(utils.os, "chown", create=True, side_effect=fake_chown):
            assert utils.chown_to_match_parent(target) is False

        assert recorded["called"] is False

    def test_noop_on_non_posix(self, tmp_path, monkeypatch):
        target = tmp_path / "auth.json"
        target.write_text("{}", encoding="utf-8")

        # Simulate Windows: os.chown / os.geteuid don't exist.
        # Use monkeypatch.delattr for hermetic isolation — manual
        # delattr + try/finally is fragile (a KeyboardInterrupt between
        # the delattr and the restore would permanently break os.chown
        # for the rest of the test session).
        monkeypatch.delattr(utils.os, "chown", raising=False)
        monkeypatch.delattr(utils.os, "geteuid", raising=False)

        assert utils.chown_to_match_parent(target) is False

    def test_refuses_to_chown_symlink_target(self, tmp_path):
        """If auth.json is replaced with a symlink between os.replace and
        chown_to_match_parent (TOCTOU window), the helper MUST NOT
        re-own the link's destination — that would let an unprivileged
        attacker who can write into the parent dir steer root to chown
        arbitrary files like /etc/shadow.
        """
        # Create a "destination" file we must protect from being chowned.
        secret = tmp_path / "shadow"
        secret.write_text("root:x:...", encoding="utf-8")
        secret_stat_before = os.stat(secret)

        # The "auth.json" path is actually a symlink to the secret.
        target = tmp_path / "auth.json"
        os.symlink(secret, target)

        recorded = {"called": False, "args": None}

        def fake_chown(*args, **kwargs):
            recorded["called"] = True
            recorded["args"] = (args, kwargs)

        # Simulate root caller. Don't patch os.lstat — we want the real
        # one to detect the symlink.
        with patch.object(utils.os, "geteuid", create=True, return_value=0), \
             patch.object(utils.os, "chown", create=True, side_effect=fake_chown):
            assert utils.chown_to_match_parent(target) is False

        # Crucial: chown must never have been invoked at all.
        assert recorded["called"] is False, (
            f"chown was called with {recorded['args']} on a symlink — "
            "this would re-own the link's destination, a security bug."
        )
        # And the destination file is untouched.
        secret_stat_after = os.stat(secret)
        assert secret_stat_after.st_uid == secret_stat_before.st_uid
        assert secret_stat_after.st_gid == secret_stat_before.st_gid

    def test_chown_uses_follow_symlinks_false_when_supported(self, tmp_path):
        """Belt-and-suspenders: even when the target passes the lstat
        symlink check, the actual chown call must be invoked with
        follow_symlinks=False (or fall through to os.lchown) so a TOCTOU
        race between lstat and chown still cannot re-own a link target.
        """
        target = tmp_path / "auth.json"
        target.write_text("{}", encoding="utf-8")

        parent_stat = os.stat(tmp_path)
        file_stat = os.stat(target)
        fake_file_stat = type(file_stat)(
            (
                file_stat.st_mode, file_stat.st_ino, file_stat.st_dev,
                file_stat.st_nlink, 0, 0, file_stat.st_size,
                file_stat.st_atime, file_stat.st_mtime, file_stat.st_ctime,
            )
        )
        fake_parent_stat = type(parent_stat)(
            (
                parent_stat.st_mode, parent_stat.st_ino, parent_stat.st_dev,
                parent_stat.st_nlink, 10000, 10000, parent_stat.st_size,
                parent_stat.st_atime, parent_stat.st_mtime, parent_stat.st_ctime,
            )
        )

        recorded = {"args": None, "kwargs": None}

        def fake_chown(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs

        # Patch chown, then put the patched function into
        # supports_follow_symlinks so the helper takes the
        # follow_symlinks=False code path.
        with patch.object(utils.os, "geteuid", create=True, return_value=0), \
             patch.object(utils.os, "chown", create=True, side_effect=fake_chown) as chown_mock, \
             patch.object(utils.os, "lstat", return_value=fake_file_stat), \
             patch.object(utils.os, "stat", return_value=fake_parent_stat), \
             patch.object(utils.os, "supports_follow_symlinks", {chown_mock}):
            assert utils.chown_to_match_parent(target) is True

        assert recorded["kwargs"].get("follow_symlinks") is False, (
            "chown must be called with follow_symlinks=False to avoid "
            "TOCTOU symlink-substitution attacks."
        )
        assert recorded["args"] == (target, 10000, 10000)

    def test_chown_falls_back_to_lchown_when_follow_symlinks_unsupported(self, tmp_path):
        """On exotic platforms where os.chown doesn't accept
        follow_symlinks, the helper must fall through to os.lchown
        (Unix-equivalent symlink-safe chown).
        """
        target = tmp_path / "auth.json"
        target.write_text("{}", encoding="utf-8")

        parent_stat = os.stat(tmp_path)
        file_stat = os.stat(target)
        fake_file_stat = type(file_stat)(
            (
                file_stat.st_mode, file_stat.st_ino, file_stat.st_dev,
                file_stat.st_nlink, 0, 0, file_stat.st_size,
                file_stat.st_atime, file_stat.st_mtime, file_stat.st_ctime,
            )
        )
        fake_parent_stat = type(parent_stat)(
            (
                parent_stat.st_mode, parent_stat.st_ino, parent_stat.st_dev,
                parent_stat.st_nlink, 10000, 10000, parent_stat.st_size,
                parent_stat.st_atime, parent_stat.st_mtime, parent_stat.st_ctime,
            )
        )

        chown_calls = {"called": False}
        lchown_calls = {"args": None}

        def fake_chown(*args, **kwargs):
            chown_calls["called"] = True

        def fake_lchown(*args, **kwargs):
            lchown_calls["args"] = args

        # Empty supports_follow_symlinks set => fall back to os.lchown.
        with patch.object(utils.os, "geteuid", create=True, return_value=0), \
             patch.object(utils.os, "chown", create=True, side_effect=fake_chown), \
             patch.object(utils.os, "lchown", create=True, side_effect=fake_lchown), \
             patch.object(utils.os, "lstat", return_value=fake_file_stat), \
             patch.object(utils.os, "stat", return_value=fake_parent_stat), \
             patch.object(utils.os, "supports_follow_symlinks", set()):
            assert utils.chown_to_match_parent(target) is True

        assert chown_calls["called"] is False, (
            "Must use os.lchown when os.chown lacks follow_symlinks support, "
            "not invoke the symlink-following os.chown."
        )
        assert lchown_calls["args"] == (target, 10000, 10000)

    def test_chown_oserror_is_swallowed(self, tmp_path):
        target = tmp_path / "auth.json"
        target.write_text("{}", encoding="utf-8")

        parent_stat = os.stat(tmp_path)
        file_stat = os.stat(target)
        fake_parent = type(parent_stat)(
            (
                parent_stat.st_mode, parent_stat.st_ino, parent_stat.st_dev,
                parent_stat.st_nlink, 10000, 10000, parent_stat.st_size,
                parent_stat.st_atime, parent_stat.st_mtime, parent_stat.st_ctime,
            )
        )
        fake_file = type(file_stat)(
            (
                file_stat.st_mode, file_stat.st_ino, file_stat.st_dev,
                file_stat.st_nlink, 0, 0, file_stat.st_size,
                file_stat.st_atime, file_stat.st_mtime, file_stat.st_ctime,
            )
        )

        def fake_stat(path):
            return fake_parent

        def fake_lstat(path):
            return fake_file

        def boom(*_a, **_kw):
            raise PermissionError("EPERM")

        with patch.object(utils.os, "geteuid", create=True, return_value=0), \
             patch.object(utils.os, "chown", create=True, side_effect=boom) as chown_mock, \
             patch.object(utils.os, "lstat", side_effect=fake_lstat), \
             patch.object(utils.os, "stat", side_effect=fake_stat), \
             patch.object(utils.os, "supports_follow_symlinks", {chown_mock}):
            # MUST NOT raise — write site should not crash on chown failure.
            assert utils.chown_to_match_parent(target) is False


# ---------------------------------------------------------------------------
# Integration with auth.py write paths
# ---------------------------------------------------------------------------


class TestAuthSaveInvokesChown:
    def test_save_auth_store_calls_chown_to_match_parent(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        captured = []

        def fake_chown(path):
            captured.append(Path(path))
            return False

        with patch.object(auth_module, "chown_to_match_parent", side_effect=fake_chown):
            auth_module._save_auth_store({"providers": {"openai-codex": {"tokens": {}}}})

        # The remediation hook MUST be invoked on the resulting auth.json,
        # not on the temp file or anything else.
        assert (hermes_home / "auth.json") in captured

    def test_save_qwen_cli_tokens_calls_chown_to_match_parent(self, tmp_path, monkeypatch):
        # Redirect Qwen CLI path into tmp so we don't touch ~/.qwen.
        fake_qwen_path = tmp_path / ".qwen" / "oauth_creds.json"
        monkeypatch.setattr(
            auth_module,
            "_qwen_cli_auth_path",
            lambda: fake_qwen_path,
        )

        captured = []

        def fake_chown(path):
            captured.append(Path(path))
            return False

        with patch.object(auth_module, "chown_to_match_parent", side_effect=fake_chown):
            auth_module._save_qwen_cli_tokens({"access_token": "x", "refresh_token": "y"})

        assert fake_qwen_path in captured
