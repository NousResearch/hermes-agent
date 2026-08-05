"""``atomic_write_text``'s opt-in metadata preservation (mode + owner).

``os.replace`` swaps mkstemp's 0600 temp file (owned by the writing user)
onto the target, so a bare atomic rewrite of an existing user-authored file
tightens its permission bits and — for root-run callers on Docker/NAS
volumes — flips its ownership.  ``preserve_mode=True`` carries both across
the replace, exactly like ``atomic_yaml_write`` does unconditionally;
``create_mode=`` sets the bits when the target does not exist yet.

These guard the follow-up to PR #79323, which collapsed three hand-rolled
stat/write/chmod blocks (xai migration, uninstaller shell-rc rewrite,
dashboard SOUL.md editor) into these kwargs.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from utils import atomic_write_text, atomic_yaml_write


pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX permission bits"
)


class TestPreserveMode:
    def test_existing_mode_survives_the_rewrite(self, tmp_path: Path) -> None:
        """A 0640 managed config must not tighten to mkstemp's 0600."""
        target = tmp_path / "config.yaml"
        target.write_text("old: true\n", encoding="utf-8")
        os.chmod(target, 0o640)

        atomic_write_text(target, "new: true\n", preserve_mode=True)

        assert target.read_text(encoding="utf-8") == "new: true\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o640

    def test_default_still_leaves_mkstemp_mode(self, tmp_path: Path) -> None:
        """Without opt-in, behavior is unchanged: the file lands 0600."""
        target = tmp_path / "notes.md"
        target.write_text("old\n", encoding="utf-8")
        os.chmod(target, 0o644)

        atomic_write_text(target, "new\n")

        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_mode_is_applied_before_the_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The temp fd gets fchmod'd, so the target never transits 0600."""
        target = tmp_path / "config.yaml"
        target.write_text("old\n", encoding="utf-8")
        os.chmod(target, 0o640)

        import utils as utils_mod

        real_replace = utils_mod.atomic_replace
        seen: list[int] = []

        def spying_replace(tmp, dst):
            seen.append(stat.S_IMODE(os.stat(tmp).st_mode))
            return real_replace(tmp, dst)

        monkeypatch.setattr(utils_mod, "atomic_replace", spying_replace)
        atomic_write_text(target, "new\n", preserve_mode=True)

        assert seen == [0o640]

    def test_owner_is_restored_on_the_real_symlink_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Root-run rewrites of a user-owned file must not flip ownership.

        Mirrors test_atomic_yaml_write_restores_owner_on_real_symlink_target:
        forces a preserved uid/gid so the test does not need root.
        """
        real = tmp_path / "zshrc"
        link = tmp_path / ".zshrc"
        real.write_text("export A=1\n", encoding="utf-8")
        link.symlink_to(real)

        chown_calls: list[tuple[Path, int, int]] = []
        monkeypatch.setattr("utils._preserve_file_owner", lambda _p: (123, 456))
        monkeypatch.setattr(
            "utils.os.chown",
            lambda path, uid, gid: chown_calls.append((Path(path), uid, gid)),
        )

        atomic_write_text(link, "export B=2\n", preserve_mode=True)

        assert chown_calls == [(real, 123, 456)]
        assert link.is_symlink()
        assert real.read_text(encoding="utf-8") == "export B=2\n"

    def test_no_owner_calls_without_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "mem.md"
        target.write_text("old\n", encoding="utf-8")

        chown_calls: list[tuple] = []
        monkeypatch.setattr("utils._preserve_file_owner", lambda _p: (123, 456))
        monkeypatch.setattr(
            "utils.os.chown", lambda *a: chown_calls.append(a)
        )

        atomic_write_text(target, "new\n")

        assert chown_calls == []


class TestCreateMode:
    def test_create_mode_applies_when_target_is_new(self, tmp_path: Path) -> None:
        target = tmp_path / "SOUL.md"
        assert not target.exists()

        atomic_write_text(
            target, "# Persona\n", preserve_mode=True, create_mode=0o644
        )

        assert stat.S_IMODE(target.stat().st_mode) == 0o644

    def test_existing_mode_beats_create_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "SOUL.md"
        target.write_text("old\n", encoding="utf-8")
        os.chmod(target, 0o600)

        atomic_write_text(
            target, "new\n", preserve_mode=True, create_mode=0o644
        )

        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_create_mode_never_rewrites_an_existing_file(
        self, tmp_path: Path
    ) -> None:
        """create_mode without preserve_mode must not chmod an existing file."""
        target = tmp_path / "notes.md"
        target.write_text("old\n", encoding="utf-8")
        os.chmod(target, 0o640)

        atomic_write_text(target, "new\n", create_mode=0o644)

        # The write is a plain (non-preserving) atomic rewrite: mkstemp 0600.
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_windows_fallback_branch_applies_mode_after_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without os.fchmod (Windows), the mode is applied post-replace."""
        target = tmp_path / "config.yaml"
        target.write_text("old\n", encoding="utf-8")
        os.chmod(target, 0o640)

        monkeypatch.delattr(os, "fchmod")
        atomic_write_text(target, "new\n", preserve_mode=True)

        assert target.read_text(encoding="utf-8") == "new\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o640

    def test_atomic_yaml_write_create_mode(self, tmp_path: Path) -> None:
        """write_manifest's create path: new file lands 0644, not 0600."""
        target = tmp_path / "distribution.yaml"
        assert not target.exists()

        atomic_yaml_write(target, {"name": "t"}, create_mode=0o644)

        assert stat.S_IMODE(target.stat().st_mode) == 0o644

    def test_atomic_yaml_write_existing_mode_beats_create_mode(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "distribution.yaml"
        target.write_text("name: old\n", encoding="utf-8")
        os.chmod(target, 0o600)

        atomic_yaml_write(target, {"name": "new"}, create_mode=0o644)

        assert stat.S_IMODE(target.stat().st_mode) == 0o600
