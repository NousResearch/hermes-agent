"""The installed-package gate rejects macOS xattr sidecars shipped in the deb (#126097)."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.termux.validate_installed import assert_no_macos_metadata

pytestmark = pytest.mark.platforms("posix")

# Real sidecars start with the AppleDouble magic; used to stage a realistic leak.
APPLEDOUBLE_MAGIC = b"\x00\x05\x16\x07"

SCRIPTS = Path(__file__).parents[2] / "scripts/termux"


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


def test_any_dot_underscore_file_fails_even_without_appledouble_magic(tmp_path):
    # Any `._*`-named file in a Termux payload is build-host leakage by
    # definition -- the Finder/xattr convention has no legitimate use there --
    # so the gate stays name-based and symmetric with the build-side purge.
    (tmp_path / "site-packages").mkdir()
    (tmp_path / "site-packages/._notes.md").write_text(
        "plain text, not an xattr sidecar\n"
    )
    with pytest.raises(AssertionError, match=r"site-packages/\._notes\.md"):
        assert_no_macos_metadata(tmp_path)


def test_build_purge_removes_every_metadata_file_from_a_staged_tree(tmp_path):
    stage = tmp_path / "stage"
    (stage / "venv/bin").mkdir(parents=True)
    (stage / "venv/bin/._python3").write_bytes(APPLEDOUBLE_MAGIC + b"\x00" * 159)
    (stage / "app").mkdir()
    (stage / "app/.DS_Store").write_bytes(b"Bud1")
    (stage / "app/real.py").write_text("payload file that must survive\n")
    purged = int(
        subprocess.run(
            ["bash", str(SCRIPTS / "purge_macos_metadata.sh"), str(stage)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert purged == 2
    assert not list(stage.rglob("._*"))
    assert not list(stage.rglob(".DS_Store"))
    assert (stage / "app/real.py").exists()


def test_build_deb_wires_the_purge_in():
    # Wiring check: the assembly script must invoke the purge on the staged
    # tree, or the deb ships whatever the payload producers leaked.
    assert "purge_macos_metadata.sh" in (SCRIPTS / "build_deb.sh").read_text()
