#!/usr/bin/env python3
"""PEP 738 wheel retagger for the Termux/Android wheelhouse.

Rewrites a built (e.g. linux_aarch64 / manylinux) wheel so it installs on
Android/bionic under Termux:

  * wheel filename platform field (last dash-field before ``.whl``)
  * every ``Tag:`` line inside ``<dist>.dist-info/WHEEL``
  * ``RECORD`` is regenerated from the new member contents

Everything is verified before anything is rewritten:

  * the wheel's filename package/version must match its ``METADATA``
  * the archive must pass a full ZIP integrity check
  * the wheel must contain at least one native extension (``.so``) --
    retagging a pure wheel onto a platform tag would be a lie
  * the target platform tag must be a valid PEP 738 android tag

Pure stdlib. ASCII only.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import sys
import tempfile
import zipfile

ANDROID_TAG_RE = re.compile(r"^android_(\d+)_arm64_v8a$")
RECORD_SUFFIX = "dist-info/RECORD"
WHEEL_SUFFIX = "dist-info/WHEEL"


class RetagError(RuntimeError):
    """Raised when a wheel cannot be safely retagged."""


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def parse_wheel_filename(filename: str) -> tuple[str, str, str, str, str]:
    """Return (distribution, version, python_tag, abi_tag, platform_tag)."""
    if not filename.endswith(".whl"):
        raise RetagError(f"not a wheel filename: {filename!r}")
    stem = filename[: -len(".whl")]
    parts = stem.split("-")
    if len(parts) != 5:
        raise RetagError(
            f"unexpected wheel filename shape (want 5 dash fields): {filename!r}"
        )
    distribution, version, python_tag, abi_tag, platform_tag = parts
    if not distribution or not version:
        raise RetagError(f"empty distribution/version in {filename!r}")
    return distribution, version, python_tag, abi_tag, platform_tag


def _read_metadata(archive: zipfile.ZipFile) -> tuple[str, str]:
    """Return (name, version) parsed from METADATA."""
    records = [n for n in archive.namelist() if n.endswith(RECORD_SUFFIX)]
    if len(records) != 1:
        raise RetagError(f"expected exactly one dist-info/RECORD, found {records!r}")
    metadata_name = records[0][: -len("RECORD")] + "METADATA"
    try:
        text = archive.read(metadata_name).decode("utf-8")
    except KeyError:
        raise RetagError(f"missing {metadata_name}") from None
    name = None
    version = None
    for line in text.splitlines():
        if line.startswith("Name:"):
            name = line[len("Name:"):].strip()
        elif line.startswith("Version:"):
            version = line[len("Version:"):].strip()
        if name is not None and version is not None:
            break
    if not name or not version:
        raise RetagError(f"incomplete METADATA in {metadata_name}")
    return name, version


def _validate(archive: zipfile.ZipFile, filename: str, new_platform: str) -> str:
    """Run all pre-retag checks; return the wheel's dist-info prefix."""
    distribution, version, _, _, platform_tag = parse_wheel_filename(filename)

    if ANDROID_TAG_RE.match(platform_tag):
        raise RetagError(f"{filename} is already an android-tagged wheel")

    name, meta_version = _read_metadata(archive)
    # PEP 503-normalized comparison: setuptools writes canonical lowercase
    # filenames while METADATA carries the display name (markupsafe vs
    # MarkupSafe). Case is not identity for package names.
    def _norm(n: str) -> str:
        return re.sub(r"[-._]+", "-", n).lower()

    if name and distribution and _norm(name) != _norm(distribution):
        raise RetagError(
            f"wheel filename package {distribution!r} does not match "
            f"METADATA Name {name!r}"
        )
    if meta_version != version:
        raise RetagError(
            f"wheel filename version {version!r} does not match "
            f"METADATA Version {meta_version!r}"
        )

    # No archive-wide testzip(): every member read below (archive.read)
    # already verifies the member CRC and raises zipfile.BadZipFile on
    # mismatch, so the full-archive decompress would be pure duplicated work.

    if not any(n.endswith(".so") for n in archive.namelist()):
        raise RetagError(
            f"{filename} contains no native extension (.so); refusing to "
            "stamp a platform tag onto a pure wheel"
        )

    dist_info_prefix = f"{distribution}-{version}.dist-info/"
    wheel_member = dist_info_prefix + "WHEEL"
    if wheel_member not in archive.namelist():
        raise RetagError(f"missing {wheel_member}")
    tags = [
        line[len("Tag:"):].strip()
        for line in archive.read(wheel_member).decode("utf-8").splitlines()
        if line.startswith("Tag:")
    ]
    if not tags:
        raise RetagError(f"no Tag: lines in {wheel_member}")
    for tag in tags:
        fields = tag.split("-")
        if len(fields) != 3 or fields[2] != platform_tag:
            raise RetagError(
                f"WHEEL tag {tag!r} inconsistent with filename platform "
                f"{platform_tag!r}"
            )

    if not ANDROID_TAG_RE.match(new_platform):
        raise RetagError(
            f"target platform tag {new_platform!r} is not a PEP 738 "
            "android_<abi>_arm64_v8a style tag"
        )
    return dist_info_prefix


def retag_wheel(wheel_path: str, new_platform: str) -> str:
    """Retag the wheel at *wheel_path*; return the new wheel path."""
    wheel_path = os.path.abspath(wheel_path)
    wheel_name = os.path.basename(wheel_path)
    distribution, version, python_tag, abi_tag, _ = parse_wheel_filename(wheel_name)
    new_wheel_name = "-".join(
        (distribution, version, python_tag, abi_tag, new_platform)
    ) + ".whl"

    with zipfile.ZipFile(wheel_path, "r") as archive:
        dist_info_prefix = _validate(archive, wheel_name, new_platform)

        wheel_member = dist_info_prefix + "WHEEL"
        old_wheel_text = archive.read(wheel_member).decode("utf-8")

        new_wheel_lines = []
        for line in old_wheel_text.splitlines():
            if line.startswith("Tag:"):
                fields = line[len("Tag:"):].strip().split("-")
                new_tag = "-".join((fields[0], fields[1], new_platform))
                new_wheel_lines.append("Tag: " + new_tag)
            else:
                new_wheel_lines.append(line)
        new_wheel_text = "\n".join(new_wheel_lines) + "\n"

        members: list[tuple[str, bytes]] = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            members.append((info.filename, archive.read(info.filename)))

    new_members: list[tuple[str, bytes]] = []
    record_rows: list[tuple[str, str, str]] = []
    for name, data in members:
        if name == wheel_member:
            data = new_wheel_text.encode("utf-8")
            new_members.append((name, data))
            record_rows.append((name, record_hash(data), str(len(data))))
        elif name == dist_info_prefix + "RECORD":
            continue  # regenerated below
        else:
            new_members.append((name, data))
            record_rows.append((name, record_hash(data), str(len(data))))
    record_rows.append((dist_info_prefix + "RECORD", "", ""))

    record_buf = io.StringIO()
    writer = csv.writer(record_buf, lineterminator="\n")
    writer.writerows(record_rows)
    new_members.append(
        (dist_info_prefix + "RECORD", record_buf.getvalue().encode("utf-8"))
    )

    out_dir = os.path.dirname(wheel_path)
    fd, tmp_path = tempfile.mkstemp(prefix=".retag-", suffix=".whl", dir=out_dir)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as out:
            for name, data in new_members:
                out.writestr(name, data)
        new_wheel_path = os.path.join(out_dir, new_wheel_name)
        os.replace(tmp_path, new_wheel_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if os.path.abspath(new_wheel_path) != wheel_path:
        os.unlink(wheel_path)
    return new_wheel_path


def self_check() -> int:
    """Refuse to run unless the module's invariants hold (used by the build gate)."""
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "demo_pkg-1.0-cp312-cp312-linux_aarch64.whl")
        with zipfile.ZipFile(good, "w") as zf:
            zf.writestr("demo_pkg/__init__.py", "x = 1\n")
            zf.writestr("demo_pkg/_native.so", b"\x7fELFfake")
            metadata = "Metadata-Version: 2.1\nName: demo_pkg\nVersion: 1.0\n"
            zf.writestr("demo_pkg-1.0.dist-info/METADATA", metadata)
            zf.writestr(
                "demo_pkg-1.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nTag: cp312-cp312-linux_aarch64\n",
            )
            zf.writestr("demo_pkg-1.0.dist-info/RECORD", "")
        retag_wheel(good, "android_24_arm64_v8a")
        retagged = os.path.join(
            tmp, "demo_pkg-1.0-cp312-cp312-android_24_arm64_v8a.whl"
        )
        if not os.path.exists(retagged):
            raise SystemExit("self_check: retagged wheel missing")
        with zipfile.ZipFile(retagged) as zf:
            names = zf.namelist()
            wheel_txt = zf.read("demo_pkg-1.0.dist-info/WHEEL").decode("utf-8")
            record_txt = zf.read("demo_pkg-1.0.dist-info/RECORD").decode("utf-8")
        assert "Tag: cp312-cp312-android_24_arm64_v8a" in wheel_txt
        assert any(n.endswith("_native.so") for n in names)
        assert "demo_pkg-1.0.dist-info/RECORD,," in record_txt
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retag one or more wheels for PEP 738 Android/Termux."
    )
    parser.add_argument(
        "wheels",
        nargs="*",
        help="paths to the wheels to retag (batch mode: all processed "
        "in-process, stopping at the first error)",
    )
    parser.add_argument(
        "--platform-tag",
        default="android_24_arm64_v8a",
        help="target platform tag (default: android_24_arm64_v8a)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the built-in round-trip self-check instead of retagging",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    if not args.wheels:
        parser.error("at least one wheel path is required unless --self-check is given")
    for wheel in args.wheels:
        try:
            new_path = retag_wheel(wheel, args.platform_tag)
        except (RetagError, zipfile.BadZipFile, OSError) as exc:
            print(f"retag_wheel: error: {wheel}: {exc}", file=sys.stderr)
            return 1
        print(new_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
