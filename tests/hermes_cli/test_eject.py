"""Tests for ``hermes eject`` (phase 3, task 3.5).

See ``docs/plans/updater-rework/04-phase3-ejected-dev.md`` task 3.5.

These tests exercise behavior via the function call — they do NOT read
source code (AGENTS.md §"Never read source code in tests").
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.subcommands import eject as eject_mod
from hermes_cli.subcommands.eject import cmd_eject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> SimpleNamespace:
    """Build a minimal args namespace for cmd_eject."""
    defaults = {
        "dir": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_slot(tmp_path: Path, git_sha: str = "abc123def456") -> Path:
    """Create a fake managed slot directory with a manifest.json.

    Returns the slot root path.
    """
    slot = tmp_path / "slot" / "hermes-agent"
    slot.mkdir(parents=True)
    manifest = {
        "schema": 1,
        "version": "2026.07.14",
        "channel": "stable",
        "git_sha": git_sha,
        "platform": "linux-x64",
    }
    (slot / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return slot


def _make_checkout(tmp_path: Path) -> Path:
    """Create a fake source checkout directory with .git and pyproject.toml."""
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    (checkout / ".git").mkdir()
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\n', encoding="utf-8"
    )
    return checkout


# ---------------------------------------------------------------------------
# (1) From a slot: clones, runs dev sync, re-points symlink, records
#     .pre-eject-target
# ---------------------------------------------------------------------------

class TestEjectFromSlot:
    """``hermes eject`` from a managed slot install."""

    def test_eject_from_slot_full_flow(self, monkeypatch, tmp_path):
        """From a slot: clones at git_sha, runs dev sync (mock), re-points
        symlink, records .pre-eject-target."""
        git_sha = "abc123def456"
        slot = _make_slot(tmp_path, git_sha=git_sha)
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        expected_dest = hermes_home / "source"

        # Track calls to the mocked functions.
        clone_calls: list[tuple] = []
        sync_calls: list[tuple] = []

        def fake_clone(dest: Path, sha: str, **kw) -> Path:
            clone_calls.append((dest, sha))
            dest.mkdir(parents=True, exist_ok=True)
            # Simulate the clone creating bin/hermes.
            (dest / "bin").mkdir(parents=True, exist_ok=True)
            (dest / "bin" / "hermes").write_text("#!/bin/sh\n")
            return dest

        def fake_sync(checkout_dir: Path) -> None:
            sync_calls.append((checkout_dir,))

        monkeypatch.setattr(eject_mod, "_clone_at_sha", fake_clone)
        monkeypatch.setattr(eject_mod, "_run_dev_sync", fake_sync)
        monkeypatch.setattr(eject_mod, "_get_path_symlink",
                            lambda: tmp_path / "bin" / "hermes")

        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(slot)):
            cmd_eject(_make_args())

        # 1. Clone was called with the right sha and destination.
        assert len(clone_calls) == 1
        cloned_dest, cloned_sha = clone_calls[0]
        assert cloned_sha == git_sha
        assert cloned_dest == expected_dest

        # 2. dev sync was called with the checkout dir.
        assert len(sync_calls) == 1
        assert sync_calls[0][0] == expected_dest

        # 3. Symlink was re-pointed at the checkout's bin/hermes.
        symlink = tmp_path / "bin" / "hermes"
        assert symlink.is_symlink()
        assert symlink.resolve() == (expected_dest / "bin" / "hermes").resolve()

        # 4. .pre-eject-target was recorded.
        pre_eject = hermes_home / ".pre-eject-target"
        assert pre_eject.exists()
        assert str(slot) in pre_eject.read_text()

    def test_eject_clone_uses_slot_git_sha(self, monkeypatch, tmp_path):
        """The git_sha from the slot manifest is passed to the clone."""
        git_sha = "deadbeefcafe"
        slot = _make_slot(tmp_path, git_sha=git_sha)
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()

        captured_sha: list[str] = []

        def fake_clone(dest: Path, sha: str, **kw) -> Path:
            captured_sha.append(sha)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "bin").mkdir(parents=True, exist_ok=True)
            (dest / "bin" / "hermes").write_text("#!/bin/sh\n")
            return dest

        monkeypatch.setattr(eject_mod, "_clone_at_sha", fake_clone)
        monkeypatch.setattr(eject_mod, "_run_dev_sync", lambda d: None)
        monkeypatch.setattr(eject_mod, "_get_path_symlink",
                            lambda: tmp_path / "bin" / "hermes")

        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(slot)):
            cmd_eject(_make_args())

        assert captured_sha == [git_sha]

    def test_eject_prints_caveats(self, monkeypatch, tmp_path, capsys):
        """The ejected-contract caveats are printed."""
        slot = _make_slot(tmp_path)
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()

        monkeypatch.setattr(eject_mod, "_clone_at_sha",
                            lambda d, s, **kw: d.mkdir(parents=True, exist_ok=True) or
                            ((d / "bin").mkdir(parents=True, exist_ok=True) or
                             (d / "bin" / "hermes").write_text("#") or d))
        monkeypatch.setattr(eject_mod, "_run_dev_sync", lambda d: None)
        monkeypatch.setattr(eject_mod, "_get_path_symlink",
                            lambda: tmp_path / "bin" / "hermes")

        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(slot)):
            cmd_eject(_make_args())

        err = capsys.readouterr().err
        # Check for key caveat fragments from §2.5.
        assert "build locally" in err.lower()
        assert "syntax guard" in err.lower()
        assert "desktop" in err.lower()
        assert "adopt" in err.lower()

    def test_eject_clone_failure_exits_nonzero(self, monkeypatch, tmp_path):
        """If the clone fails, eject exits with code 1."""
        slot = _make_slot(tmp_path)
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()

        def boom(dest, sha, **kw):
            raise RuntimeError("network error")

        monkeypatch.setattr(eject_mod, "_clone_at_sha", boom)

        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(slot)):
            with pytest.raises(SystemExit) as excinfo:
                cmd_eject(_make_args())

        assert excinfo.value.code == 1

    def test_eject_dev_sync_failure_does_not_block(self, monkeypatch, tmp_path):
        """If dev sync fails, eject continues (warns but re-points symlink)."""
        slot = _make_slot(tmp_path)
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()

        def fake_clone(dest, sha, **kw):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "bin").mkdir(parents=True, exist_ok=True)
            (dest / "bin" / "hermes").write_text("#")
            return dest

        monkeypatch.setattr(eject_mod, "_clone_at_sha", fake_clone)
        monkeypatch.setattr(eject_mod, "_run_dev_sync",
                            lambda d: (_ for _ in ()).throw(
                                RuntimeError("sync boom")))
        monkeypatch.setattr(eject_mod, "_get_path_symlink",
                            lambda: tmp_path / "bin" / "hermes")


        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(slot)):
            # Should NOT exit nonzero — just warn.
            cmd_eject(_make_args())

        # Symlink should still be re-pointed.
        symlink = tmp_path / "bin" / "hermes"
        assert symlink.is_symlink()


# ---------------------------------------------------------------------------
# (2) From a checkout: exits 0 with "already ejected"
# ---------------------------------------------------------------------------

class TestEjectFromCheckout:
    """``hermes eject`` from a source checkout → already ejected."""

    def test_already_ejected_exits_zero(self, monkeypatch, tmp_path, capsys):
        """From a checkout, eject exits 0 with 'already ejected' message."""
        checkout = _make_checkout(tmp_path)



        with patch("hermes_cli.main.PROJECT_ROOT", str(checkout)):
            cmd_eject(_make_args())

        out = capsys.readouterr().out
        assert "already ejected" in out.lower()

    def test_already_ejected_shows_status(self, monkeypatch, tmp_path, capsys):
        """From a checkout, eject prints the checkout path and symlink status."""
        checkout = _make_checkout(tmp_path)

        # Create a symlink to simulate an active PATH symlink.
        link_dir = tmp_path / "bin"
        link_dir.mkdir()
        target = checkout / "bin" / "hermes"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#")
        link = link_dir / "hermes"
        link.symlink_to(target)

        monkeypatch.setattr(eject_mod, "_get_path_symlink", lambda: link)


        with patch("hermes_cli.main.PROJECT_ROOT", str(checkout)):
            cmd_eject(_make_args())

        out = capsys.readouterr().out
        assert str(checkout) in out
        assert "symlink" in out.lower()


# ---------------------------------------------------------------------------
# (3) --dir override
# ---------------------------------------------------------------------------

class TestEjectDirOverride:
    """``hermes eject --dir PATH`` uses the specified destination."""

    def test_dir_override_changes_destination(self, monkeypatch, tmp_path):
        """The --dir flag overrides the default $HERMES_HOME/source path."""
        slot = _make_slot(tmp_path)
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        custom_dir = tmp_path / "custom" / "checkout"

        clone_calls: list[tuple] = []

        def fake_clone(dest: Path, sha: str, **kw) -> Path:
            clone_calls.append((dest, sha))
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "bin").mkdir(parents=True, exist_ok=True)
            (dest / "bin" / "hermes").write_text("#")
            return dest

        monkeypatch.setattr(eject_mod, "_clone_at_sha", fake_clone)
        monkeypatch.setattr(eject_mod, "_run_dev_sync", lambda d: None)
        monkeypatch.setattr(eject_mod, "_get_path_symlink",
                            lambda: tmp_path / "bin" / "hermes")


        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(slot)):
            cmd_eject(_make_args(dir=str(custom_dir)))

        assert len(clone_calls) == 1
        assert clone_calls[0][0] == custom_dir

        # Symlink should point at the custom dir's launcher.
        symlink = tmp_path / "bin" / "hermes"
        assert symlink.is_symlink()
        assert symlink.resolve() == (custom_dir / "bin" / "hermes").resolve()


# ---------------------------------------------------------------------------
# (4) Edge cases
# ---------------------------------------------------------------------------

class TestEjectEdgeCases:
    """Edge-case handling for ``hermes eject``."""

    def test_not_a_slot_not_a_checkout_exits_nonzero(
        self, monkeypatch, tmp_path, capsys
    ):
        """If the project root is neither a slot nor a checkout, exit 1."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()



        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(empty_dir)):
            with pytest.raises(SystemExit) as excinfo:
                cmd_eject(_make_args())

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "cannot eject" in err.lower()

    def test_slot_without_git_sha_exits_nonzero(
        self, monkeypatch, tmp_path, capsys
    ):
        """A slot manifest without git_sha causes exit 1."""
        slot = tmp_path / "slot"
        slot.mkdir()
        manifest = {"schema": 1, "version": "1.0"}  # no git_sha
        (slot / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()



        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(slot)):
            with pytest.raises(SystemExit) as excinfo:
                cmd_eject(_make_args())

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "git_sha" in err.lower()


# ---------------------------------------------------------------------------
# (5) .pre-eject-target content
# ---------------------------------------------------------------------------

class TestPreEjectTarget:
    """The .pre-eject-target file records the slot path for undo symmetry."""

    def test_pre_eject_target_contains_slot_path(self, monkeypatch, tmp_path):
        """The .pre-eject-target file contains the slot root path."""
        git_sha = "abc123"
        slot = _make_slot(tmp_path, git_sha=git_sha)
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()

        monkeypatch.setattr(eject_mod, "_clone_at_sha",
                            lambda d, s, **kw: d.mkdir(parents=True, exist_ok=True) or
                            ((d / "bin").mkdir(parents=True, exist_ok=True) or
                             (d / "bin" / "hermes").write_text("#") or d))
        monkeypatch.setattr(eject_mod, "_run_dev_sync", lambda d: None)
        monkeypatch.setattr(eject_mod, "_get_path_symlink",
                            lambda: tmp_path / "bin" / "hermes")


        with patch("hermes_constants.get_hermes_home",
                   return_value=hermes_home), \
             patch("hermes_cli.main.PROJECT_ROOT", str(slot)):
            cmd_eject(_make_args())

        pre_eject = hermes_home / ".pre-eject-target"
        assert pre_eject.exists()
        content = pre_eject.read_text().strip()
        assert str(slot) == content
