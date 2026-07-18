"""Helpers for the temporary psutil-on-Android compatibility installer."""

from __future__ import annotations

import os
import platform
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath

# Pin a version we know patches cleanly. Update when a newer psutil
# changes the marker line shape and we need to follow upstream.
PSUTIL_URL = (
    "https://files.pythonhosted.org/packages/aa/c6/"
    "d1ddf4abb55e93cebc4f2ed8b5d6dbad109ecb8d63748dd2b20ab5e57ebe/"
    "psutil-7.2.2.tar.gz"
)

MARKER = 'LINUX = sys.platform.startswith("linux")'
REPLACEMENT = 'LINUX = sys.platform.startswith(("linux", "android"))'

_DEFAULT_ANDROID_API_LEVEL = 24
_ANDROID_ABI_BY_MACHINE = {
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


class PsutilAndroidInstallError(RuntimeError):
    """Raised when the pinned psutil sdist is missing or unsafe."""


def android_wheel_platform_tag(
    *,
    api_level: int | str | None = None,
    machine: str | None = None,
) -> str:
    """Return the PEP 738 wheel platform tag for the current Termux target.

    Termux packages target Android API 24 even when the phone itself runs a
    newer Android release. Using the runtime API (for example 36) makes uv
    reject a locally built wheel because its interpreter target is API 24.
    ``HERMES_ANDROID_API_LEVEL`` is an explicit build-target override; the
    generic ``ANDROID_API_LEVEL`` variable is intentionally ignored because
    installers commonly populate it from ``getprop ro.build.version.sdk``.
    """
    raw_api_level = (
        api_level
        if api_level is not None
        else os.environ.get("HERMES_ANDROID_API_LEVEL", _DEFAULT_ANDROID_API_LEVEL)
    )
    try:
        normalized_api_level = int(raw_api_level)
    except (TypeError, ValueError) as exc:
        raise PsutilAndroidInstallError(
            f"Invalid Android API level for wheel tag: {raw_api_level!r}"
        ) from exc
    if normalized_api_level < 21:
        raise PsutilAndroidInstallError(
            f"Android API level must be 21 or newer, got {normalized_api_level}"
        )

    normalized_machine = (machine or platform.machine()).strip().lower()
    abi = _ANDROID_ABI_BY_MACHINE.get(normalized_machine)
    if abi is None:
        raise PsutilAndroidInstallError(
            "Unsupported Android architecture for wheel tag: "
            f"{normalized_machine or '<empty>'}"
        )
    return f"android_{normalized_api_level}_{abi}"


def _normalize_member_parts(member_name: str) -> tuple[str, ...]:
    path = PurePosixPath(member_name)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if path.is_absolute() or ".." in parts or not parts:
        raise PsutilAndroidInstallError(
            f"Unsafe archive member path: {member_name!r}"
        )
    return parts


def _safe_extract_tar_gz(archive: Path, destination: Path) -> None:
    """Extract a tar.gz without allowing traversal or link members."""
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            parts = _normalize_member_parts(member.name)
            target = destination.joinpath(*parts)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                raise PsutilAndroidInstallError(
                    f"Unsupported archive member type: {member.name}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise PsutilAndroidInstallError(
                    f"Cannot read archive member: {member.name}"
                )

            with extracted, open(target, "wb") as dst:
                shutil.copyfileobj(extracted, dst)

            try:
                target.chmod(member.mode & 0o777)
            except OSError:
                pass


def _configure_android_wheel_tag(src_root: Path, platform_tag: str) -> None:
    """Make setuptools emit an Android wheel without leaking env into uv.

    Exporting ``_PYTHON_HOST_PLATFORM`` around ``uv pip`` changes the platform
    uv sees while inspecting the interpreter, which fails with ``Unknown
    operating system: android_...``. A package-local ``bdist_wheel`` setting
    affects only the wheel build and leaves uv's interpreter probe untouched.
    """
    setup_cfg = src_root / "setup.cfg"
    try:
        content = setup_cfg.read_text(encoding="utf-8") if setup_cfg.exists() else ""
    except OSError as exc:
        raise PsutilAndroidInstallError("Failed to read psutil setup.cfg") from exc

    section_match = _BDIST_WHEEL_SECTION_RE.search(content)
    if section_match is None:
        separator = "" if not content else "\n" if content.endswith("\n") else "\n\n"
        updated = (
            f"{content}{separator}[bdist_wheel]\n"
            f"plat_name = {platform_tag}\n"
        )
    else:
        body = section_match.group("body")
        if _PLAT_NAME_RE.search(body):
            updated_body = _PLAT_NAME_RE.sub(
                f"plat_name = {platform_tag}", body, count=1
            )
        else:
            updated_body = f"plat_name = {platform_tag}\n{body}"
        updated = (
            content[: section_match.start("body")]
            + updated_body
            + content[section_match.end("body") :]
        )

    try:
        setup_cfg.write_text(updated, encoding="utf-8")
    except OSError as exc:
        raise PsutilAndroidInstallError("Failed to write psutil setup.cfg") from exc


def prepare_patched_psutil_sdist(
    archive: Path,
    destination: Path,
    *,
    platform_tag: str | None = None,
) -> Path:
    """Safely extract psutil and patch Android detection and wheel metadata."""
    _safe_extract_tar_gz(archive, destination)

    src_roots = sorted(
        (
            path
            for path in destination.iterdir()
            if path.is_dir() and path.name.startswith("psutil-")
        ),
        key=lambda path: path.name,
    )
    if not src_roots:
        raise PsutilAndroidInstallError(
            "psutil sdist did not contain a psutil-* directory"
        )

    src_root = src_roots[0]
    common_py = src_root / "psutil" / "_common.py"
    if not common_py.is_file():
        raise PsutilAndroidInstallError(
            f"psutil sdist did not contain {common_py.relative_to(src_root)!s}"
        )
    try:
        content = common_py.read_text(encoding="utf-8")
    except OSError as exc:
        raise PsutilAndroidInstallError(
            f"Failed to read {common_py.relative_to(src_root)!s}"
        ) from exc
    if MARKER not in content:
        raise PsutilAndroidInstallError(
            "psutil Android compatibility patch marker not found"
        )
    try:
        common_py.write_text(
            content.replace(MARKER, REPLACEMENT),
            encoding="utf-8",
        )
    except OSError as exc:
        raise PsutilAndroidInstallError(
            f"Failed to write {common_py.relative_to(src_root)!s}"
        ) from exc

    _configure_android_wheel_tag(
        src_root,
        platform_tag or android_wheel_platform_tag(),
    )
    return src_root
