"""Behavior contracts for scripts/termux/retag_wheel.py (PEP 738 retagger).

Fixture wheels are tiny fake wheels built with stdlib zipfile -- no network,
no real native compilation. The contracts assert HOW the retagged wheel must
relate to the original (filename/WHEEL/RECORD agreement, valid RECORD hashes,
native .so presence), not snapshots of any real package.

Run: scripts/run_tests.sh tests/test_termux_retag_wheel.py
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "termux"
sys.path.insert(0, str(SCRIPTS_DIR))

import retag_wheel  # noqa: E402

ANDROID_TAG = "android_24_arm64_v8a"


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _write_wheel(
    path: Path,
    distribution: str,
    version: str,
    platform_tag: str,
    *,
    metadata_version: str | None = None,
    include_so: bool = True,
) -> None:
    """Build a tiny fake wheel with the same shape the builder produces."""
    dist_info = f"{distribution}-{version}.dist-info"
    members: list[tuple[str, bytes]] = []
    if include_so:
        # A fake native extension -- retagging a pure wheel onto a platform
        # tag would be a lie, so fixtures default to carrying one.
        members.append((f"{distribution}/_native.cpython-314-aarch64-linux-gnu.so", b"\x7fELFfake"))
    members.append((f"{distribution}/__init__.py", b""))
    metadata_version = metadata_version or version
    members.append((f"{dist_info}/METADATA", f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {metadata_version}\n".encode()))
    members.append((f"{dist_info}/WHEEL", f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-{platform_tag}\nGenerator: fixture\n".encode()))

    # RECORD with real hashes so the retagger's ZIP-integrity/consistency
    # checks exercise the true path.
    rows: list[list[str]] = []
    for name, data in members:
        rows.append([name, _record_hash(data), str(len(data))])
    rows.append([f"{dist_info}/RECORD", "", ""])
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    members.append((f"{dist_info}/RECORD", buf.getvalue().encode()))

    filename = f"{distribution}-{version}-py3-none-{platform_tag}.whl"
    with zipfile.ZipFile(path / filename, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    return zf.read(name)


def _record_rows(zf: zipfile.ZipFile, dist_info: str) -> dict[str, tuple[str, str]]:
    text = zf.read(f"{dist_info}/RECORD").decode("utf-8")
    return {row[0]: (row[1], row[2]) for row in csv.reader(io.StringIO(text)) if row}


@pytest.fixture
def wheel(tmp_path: Path) -> Path:
    _write_wheel(tmp_path, "fakedep", "1.2.3", "linux_aarch64")
    return tmp_path / "fakedep-1.2.3-py3-none-linux_aarch64.whl"


def test_filename_and_wheel_tags_rewritten_consistently(wheel: Path) -> None:
    new_path = Path(retag_wheel.retag_wheel(str(wheel), ANDROID_TAG))

    assert new_path.name == f"fakedep-1.2.3-py3-none-{ANDROID_TAG}.whl"
    assert not wheel.exists(), "the original wheel must be replaced, not left beside the new one"

    with zipfile.ZipFile(new_path) as zf:
        wheel_txt = zf.read("fakedep-1.2.3.dist-info/WHEEL").decode("utf-8")
    tag_lines = [l for l in wheel_txt.splitlines() if l.startswith("Tag:")]
    assert tag_lines == [f"Tag: py3-none-{ANDROID_TAG}"], wheel_txt


def test_record_rows_valid_after_retag(wheel: Path) -> None:
    new_path = Path(retag_wheel.retag_wheel(str(wheel), ANDROID_TAG))

    with zipfile.ZipFile(new_path) as zf:
        rows = _record_rows(zf, "fakedep-1.2.3.dist-info")
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if member == "fakedep-1.2.3.dist-info/RECORD":
                # RECORD's own row is digest-less by spec; checked separately
                continue
            data = _read_member(zf, member)
            digest, size = rows[member]
            assert digest == _record_hash(data), f"RECORD hash stale for {member}"
            assert size == str(len(data)), f"RECORD size stale for {member}"
        record_row = rows["fakedep-1.2.3.dist-info/RECORD"]
        assert record_row == ("", ""), "RECORD's own row must be digest-less"


def test_native_extension_presence_required(tmp_path: Path) -> None:
    _write_wheel(tmp_path, "puredist", "0.1.0", "linux_aarch64", include_so=False)
    pure = tmp_path / "puredist-0.1.0-py3-none-linux_aarch64.whl"
    with pytest.raises(retag_wheel.RetagError, match="native"):
        retag_wheel.retag_wheel(str(pure), ANDROID_TAG)


def test_refuses_version_mismatch_between_filename_and_metadata(tmp_path: Path) -> None:
    # METADATA says 9.9.9 while the filename says 1.2.3 -- a lie the
    # retagger must refuse rather than launder.
    _write_wheel(tmp_path, "fakedep", "1.2.3", "linux_aarch64", metadata_version="9.9.9")
    lying = tmp_path / "fakedep-1.2.3-py3-none-linux_aarch64.whl"
    with pytest.raises(retag_wheel.RetagError):
        retag_wheel.retag_wheel(str(lying), ANDROID_TAG)


def test_refuses_invalid_target_platform_tag(wheel: Path) -> None:
    with pytest.raises(retag_wheel.RetagError):
        retag_wheel.retag_wheel(str(wheel), "not_a_platform")


def test_self_check_passes() -> None:
    # The built-in round-trip self-check is the builder's smoke gate.
    assert retag_wheel.self_check() == 0
