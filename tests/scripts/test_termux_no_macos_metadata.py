"""The installed-package gate rejects macOS xattr sidecars shipped in the deb (#126097)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.termux.validate_installed import APPLEDOUBLE_MAGIC, assert_no_macos_metadata

pytestmark = pytest.mark.platforms("posix")


def test_clean_payload_tree_passes(tmp_path, capsys):
    (tmp_path / "venv/bin").mkdir(parents=True)
    (tmp_path / "venv/bin/python3").write_text("#!interpreter\n")
    (tmp_path / "app/skills").mkdir(parents=True)
    (tmp_path / "app/skills/SKILL.md").write_text("---\nname: demo\n---\nbody\n")
    assert_no_macos_metadata(tmp_path)
    assert "NO_MACOS_METADATA_OK" in capsys.readouterr().out


def test_appledouble_sidecar_fails_with_its_shipped_path(tmp_path):
    sidecar = tmp_path / "venv/bin/._python3"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(APPLEDOUBLE_MAGIC + b"\x00" * 159)
    with pytest.raises(AssertionError, match=r"venv/bin/\._python3"):
        assert_no_macos_metadata(tmp_path)


def test_ds_store_anywhere_fails(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app/.DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    with pytest.raises(AssertionError, match=r"app/\.DS_Store"):
        assert_no_macos_metadata(tmp_path)


def test_same_named_file_without_appledouble_magic_is_not_a_false_alarm(tmp_path, capsys):
    # A payload file that merely starts with `._` stays shippable: only real
    # AppleDouble binaries (magic header) are build-host leakage.
    (tmp_path / "site-packages/._notes.md").parent.mkdir(parents=True)
    (tmp_path / "site-packages/._notes.md").write_text("plain text, not an xattr sidecar\n")
    assert_no_macos_metadata(tmp_path)
    assert "NO_MACOS_METADATA_OK" in capsys.readouterr().out
