"""The linux relaunch gate must compare canonical paths, not spellings.

``--install-root`` keeps whatever spelling the app's Hermes root had (a symlinked
``~/.hermes/hermes-agent``, or ``/home`` on Fedora/ostree where it is a link to
``/var/home``), while ``--relaunch-target`` comes from ``process.execPath`` /
``/proc/<pid>/exe`` and is already resolved.  A raw prefix compare then reads the
binary the rebuild just replaced as a foreign AppImage/deb and refuses to relaunch.

Drives the real ``--self-test-gate`` entry point of ``posix.sh`` (the same one
``scripts/desktop-update/repro.sh gate`` uses).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

POSIX_SH = Path(__file__).resolve().parent.parent / "scripts" / "desktop-update" / "posix.sh"

pytestmark = pytest.mark.linux_only


def _gate(install_root: Path, relaunch_target: Path) -> str:
    out = subprocess.run(
        ["bash", str(POSIX_SH), "--self-test-gate", "--install-root", str(install_root),
         "--relaunch-target", str(relaunch_target)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().split(":", 1)[0]


def _checkout(root: Path) -> Path:
    unpacked = root / "apps" / "desktop" / "release" / "linux-unpacked"
    unpacked.mkdir(parents=True)
    (unpacked / "Hermes").touch()
    return unpacked


@pytest.mark.parametrize("root_spelling,target_spelling", [("link", "real"), ("real", "link")])
def test_symlinked_spelling_on_either_side_still_relaunches(tmp_path, root_spelling, target_spelling):
    real = tmp_path / "real"
    _checkout(real)
    (tmp_path / "link").symlink_to(real)
    root = tmp_path / root_spelling
    target = tmp_path / target_spelling / "apps" / "desktop" / "release" / "linux-unpacked" / "Hermes"
    assert _gate(root, target) == "relaunch"


def test_target_outside_the_checkout_is_still_skew(tmp_path):
    real = tmp_path / "real"
    unpacked = _checkout(real)
    (tmp_path / "link").symlink_to(real)
    foreign = tmp_path / "opt" / "Hermes"
    foreign.mkdir(parents=True)
    (foreign / "hermes").touch()
    assert _gate(tmp_path / "link", foreign / "hermes") == "skew"
    # A sibling directory sharing the prefix must not be mistaken for the checkout either.
    assert _gate(tmp_path / "link", Path(str(unpacked) + "-evil") / "Hermes") == "skew"
