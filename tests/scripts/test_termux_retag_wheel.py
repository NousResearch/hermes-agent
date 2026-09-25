"""Behavior contracts for scripts/termux/retag_wheel.py (PEP 738 retagger).

Fixture wheels are tiny fake wheels built with stdlib zipfile -- no network,
no real native compilation. The contracts assert HOW the retagged wheel must
relate to the original (filename/WHEEL/RECORD agreement, valid RECORD hashes,
native .so presence), not snapshots of any real package.

Run: scripts/run_tests.sh tests/test_termux_retag_wheel.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest
from tests.termux_fixtures import write_wheel, verify_record

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "termux"
sys.path.insert(0, str(SCRIPTS_DIR))

import retag_wheel  # noqa: E402

ANDROID_TAG = "android_24_arm64_v8a"


@pytest.fixture
def wheel(tmp_path: Path) -> Path:
    write_wheel(tmp_path, "fakedep", "1.2.3", "linux_aarch64")
    return tmp_path / "fakedep-1.2.3-py3-none-linux_aarch64.whl"


def test_filename_and_wheel_tags_rewritten_consistently(wheel: Path) -> None:
    new_path = Path(retag_wheel.retag_wheel(str(wheel), ANDROID_TAG))

    assert new_path.name == f"fakedep-1.2.3-py3-none-{ANDROID_TAG}.whl"
    verify_record(new_path)
    assert not wheel.exists(), "the original wheel must be replaced, not left beside the new one"

    with zipfile.ZipFile(new_path) as zf:
        wheel_txt = zf.read("fakedep-1.2.3.dist-info/WHEEL").decode("utf-8")
    tag_lines = [l for l in wheel_txt.splitlines() if l.startswith("Tag:")]
    assert tag_lines == [f"Tag: py3-none-{ANDROID_TAG}"], wheel_txt


def test_native_extension_presence_required(tmp_path: Path) -> None:
    write_wheel(tmp_path, "puredist", "0.1.0", "linux_aarch64", include_so=False)
    pure = tmp_path / "puredist-0.1.0-py3-none-linux_aarch64.whl"
    with pytest.raises(retag_wheel.RetagError, match="native"):
        retag_wheel.retag_wheel(str(pure), ANDROID_TAG)


def test_refuses_version_mismatch_between_filename_and_metadata(tmp_path: Path) -> None:
    # METADATA says 9.9.9 while the filename says 1.2.3 -- a lie the
    # retagger must refuse rather than launder.
    write_wheel(tmp_path, "fakedep", "1.2.3", "linux_aarch64", metadata_version="9.9.9")
    lying = tmp_path / "fakedep-1.2.3-py3-none-linux_aarch64.whl"
    with pytest.raises(retag_wheel.RetagError):
        retag_wheel.retag_wheel(str(lying), ANDROID_TAG)


def test_refuses_invalid_target_platform_tag(wheel: Path) -> None:
    with pytest.raises(retag_wheel.RetagError):
        retag_wheel.retag_wheel(str(wheel), "not_a_platform")


def test_self_check_passes() -> None:
    # The built-in round-trip self-check is the builder's smoke gate.
    assert retag_wheel.self_check() == 0
