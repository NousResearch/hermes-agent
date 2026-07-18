"""Safe helpers for building CPython extension wheels on Android/Termux.

Setuptools normally labels a native Termux build as ``linux_<arch>`` even
though Android/Bionic is not ABI-compatible with glibc Linux. The functions
here apply a package-local ``bdist_wheel.plat_name`` override so the build gets
a PEP 738 Android tag without exporting ``_PYTHON_HOST_PLATFORM`` into uv.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import stat
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_ANDROID_API_LEVEL = 24
ANDROID_ABI_BY_MACHINE = {
    "aarch64": "arm64_v8a",
    "arm64": "arm64_v8a",
    "armv7l": "armeabi_v7a",
    "armv8l": "armeabi_v7a",
    "arm": "armeabi_v7a",
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "i386": "x86",
    "i486": "x86",
    "i586": "x86",
    "i686": "x86",
    "x86": "x86",
}
_BDIST_WHEEL_SECTION_RE = re.compile(
    r"(?ms)^\[bdist_wheel\]\s*\n(?P<body>.*?)(?=^\[|\Z)"
)
_PLAT_NAME_RE = re.compile(r"(?m)^plat_name\s*=.*$")


class AndroidWheelError(RuntimeError):
    """Raised when an Android source build cannot be prepared safely."""


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def android_wheel_platform_tag(
    *, api_level: int | str | None = None, machine: str | None = None
) -> str:
    raw_level = api_level if api_level is not None else DEFAULT_ANDROID_API_LEVEL
    try:
        level = int(raw_level)
    except (TypeError, ValueError) as exc:
        raise AndroidWheelError(f"Invalid Android API level: {raw_level!r}") from exc
    if not 21 <= level <= 99:
        raise AndroidWheelError(
            f"Android API level must be between 21 and 99, got {level}"
        )

    normalized_machine = (machine or platform.machine()).strip().lower()
    abi = ANDROID_ABI_BY_MACHINE.get(normalized_machine)
    if abi is None:
        raise AndroidWheelError(
            f"Unsupported Android architecture: {normalized_machine or '<empty>'}"
        )
    return f"android_{level}_{abi}"


def configure_setuptools_android_tag(src_root: Path, platform_tag: str) -> None:
    """Write a package-scoped wheel platform override into ``setup.cfg``."""
    setup_cfg = src_root / "setup.cfg"
    try:
        content = setup_cfg.read_text(encoding="utf-8") if setup_cfg.exists() else ""
    except OSError as exc:
        raise AndroidWheelError(f"Failed to read {setup_cfg}") from exc

    section = _BDIST_WHEEL_SECTION_RE.search(content)
    if section is None:
        separator = "" if not content else "" if content.endswith("\n\n") else "\n"
        if content and not content.endswith("\n"):
            separator = "\n\n"
        elif content and content.endswith("\n") and not content.endswith("\n\n"):
            separator = "\n"
        updated = f"{content}{separator}[bdist_wheel]\nplat_name = {platform_tag}\n"
    else:
        body = section.group("body")
        if _PLAT_NAME_RE.search(body):
            new_body = _PLAT_NAME_RE.sub(f"plat_name = {platform_tag}", body, count=1)
        else:
            new_body = f"plat_name = {platform_tag}\n{body}"
        updated = (
            content[: section.start("body")] + new_body + content[section.end("body") :]
        )

    try:
        setup_cfg.write_text(updated, encoding="utf-8")
    except OSError as exc:
        raise AndroidWheelError(f"Failed to write {setup_cfg}") from exc


def _safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if path.is_absolute() or not parts or ".." in parts:
        raise AndroidWheelError(f"Unsafe archive member path: {name!r}")
    return parts


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            target = destination.joinpath(*_safe_parts(member.name))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise AndroidWheelError(
                    f"Unsupported archive member type: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                raise AndroidWheelError(f"Cannot read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            try:
                target.chmod(member.mode & 0o777)
            except OSError:
                pass


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = destination.joinpath(*_safe_parts(info.filename))
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise AndroidWheelError(
                    f"Archive symlink is not allowed: {info.filename}"
                )
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def safe_extract_sdist(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        _extract_zip(archive, destination)
    elif tarfile.is_tarfile(archive):
        _extract_tar(archive, destination)
    else:
        raise AndroidWheelError(f"Unsupported sdist archive: {archive.name}")

    roots = sorted(path for path in destination.iterdir() if path.is_dir())
    if len(roots) != 1:
        raise AndroidWheelError(
            f"Expected one source directory in {archive.name}, found {len(roots)}"
        )
    return roots[0]


def pypi_sdist(name: str, version: str, destination: Path) -> Path:
    """Download an exact PyPI sdist and verify its repository SHA-256."""
    api_url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(api_url, timeout=60) as response:  # noqa: S310
            metadata = json.load(response)
    except Exception as exc:
        raise AndroidWheelError(
            f"Failed to read PyPI metadata for {name}=={version}"
        ) from exc

    candidates = [
        item for item in metadata.get("urls", []) if item.get("packagetype") == "sdist"
    ]
    if not candidates:
        raise AndroidWheelError(f"PyPI has no sdist for {name}=={version}")
    candidate = candidates[0]
    url = candidate.get("url")
    expected = (candidate.get("digests") or {}).get("sha256")
    filename = candidate.get("filename") or f"{name}-{version}.tar.gz"
    if not url or not expected:
        raise AndroidWheelError(f"Incomplete PyPI sdist metadata for {name}=={version}")

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / filename
    try:
        urllib.request.urlretrieve(url, archive)  # noqa: S310
    except Exception as exc:
        raise AndroidWheelError(f"Failed to download {name}=={version} sdist") from exc
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        archive.unlink(missing_ok=True)
        raise AndroidWheelError(
            f"SHA-256 mismatch for {name}=={version}: expected {expected}, got {actual}"
        )
    return archive


def patch_psutil_android_detection(src_root: Path) -> None:
    common = src_root / "psutil" / "_common.py"
    marker = 'LINUX = sys.platform.startswith("linux")'
    replacement = 'LINUX = sys.platform.startswith(("linux", "android"))'
    try:
        content = common.read_text(encoding="utf-8")
    except OSError as exc:
        raise AndroidWheelError("psutil sdist is missing psutil/_common.py") from exc
    if replacement in content:
        return
    if marker not in content:
        raise AndroidWheelError("psutil Android compatibility marker not found")
    common.write_text(content.replace(marker, replacement), encoding="utf-8")
