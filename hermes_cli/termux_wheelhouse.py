#!/usr/bin/env python3
"""Pinned immutable Android wheelhouse support for native Termux installs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

RELEASE_REPOSITORY = "adybag14-cyber/termux-hermes"
RELEASE_TAG = "wheelhouse-cp313-android24-arm64-20260719.1"
RELEASE_COMMIT = "b042ce9e662f14794ef6b5664c9ae711330df31b"
RELEASE_BASE_URL = (
    f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{RELEASE_TAG}"
)
SHA256SUMS_SHA256 = "916ff13af7e5283f75952b810fb6b7eef86ab3422bc5004c1ee1440d5163ade5"
MIN_ANDROID_API = 24
SUPPORTED_PYTHON = (3, 13)
SUPPORTED_ARCHES = frozenset({"aarch64", "arm64"})

LOCKED_WHEELS: dict[str, tuple[str, str, str]] = {
    "cffi": (
        "2.0.0",
        "cffi-2.0.0-cp313-cp313-android_24_arm64_v8a.whl",
        "2dc0a19762e92645cebc0d466221aeb1b77757f4e8f0daab33caa4cfa3f25db7",
    ),
    "cryptography": (
        "46.0.7",
        "cryptography-46.0.7-cp313-abi3-android_24_arm64_v8a.whl",
        "dd2cfdc605e647a5795479d3060d909a25e54ccd1bab4ac56de7c9827cfd859f",
    ),
    "jiter": (
        "0.13.0",
        "jiter-0.13.0-cp313-cp313-android_24_arm64_v8a.whl",
        "240eb59a6105e47f82ce9ce97e219958e9bd3871b390da31af75171a2b82dcc7",
    ),
    "markupsafe": (
        "3.0.3",
        "markupsafe-3.0.3-cp313-cp313-android_24_arm64_v8a.whl",
        "17d4ae49861da832cbb8d6d5211c23c65e047eb67fdb20f41e5ab9a6539a35e2",
    ),
    "pillow": (
        "12.2.0",
        "pillow-12.2.0-cp313-cp313-android_24_arm64_v8a.whl",
        "f0c0d2256e05318958d88db9740717a59b88e1906d423111c4e5e62ee7d5ef5f",
    ),
    "psutil": (
        "7.2.2",
        "psutil-7.2.2-cp36-abi3-android_24_arm64_v8a.whl",
        "c9d6df3d4595b302afd1aadb97d88fe24d0d1e47456573302ae50eb387ca0a5a",
    ),
    "pydantic-core": (
        "2.46.4",
        "pydantic_core-2.46.4-cp313-cp313-android_24_arm64_v8a.whl",
        "5c14345f4e6bf7d6cb6fd18ed3fa6828fc093f24182eaca3ed48131b5814f46a",
    ),
    "pyyaml": (
        "6.0.3",
        "pyyaml-6.0.3-cp313-cp313-android_24_arm64_v8a.whl",
        "e6f4bd9aa8f87d5868fcf7a73304b51a885f332713e8067bbf21107b9a6788b2",
    ),
    "rpds-py": (
        "0.30.0",
        "rpds_py-0.30.0-cp313-cp313-android_24_arm64_v8a.whl",
        "6a4c73b9ad80e2f95b5b33716dde1fe11068ebb86ea6efcd9b8332657205a5a2",
    ),
    "ruamel-yaml-clib": (
        "0.2.15",
        "ruamel_yaml_clib-0.2.15-cp313-cp313-android_24_arm64_v8a.whl",
        "114e76d80a60a6e35bb02ab89443e585e37eb3189928485db4de380db3312652",
    ),
}

_REQ_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")


class TermuxWheelhouseError(RuntimeError):
    """Base failure for a supported target or immutable release."""


class TermuxWheelhouseUnsupported(TermuxWheelhouseError):
    """The runtime cannot consume this CPython/Android/ARM64 wheel set."""


class TermuxWheelhouseMismatch(TermuxWheelhouseError):
    """Hermes dependency pins no longer match the immutable wheel release."""


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_versions() -> dict[str, str]:
    return {name: metadata[0] for name, metadata in LOCKED_WHEELS.items()}


def _check_versions(actual: dict[str, set[str]], *, source: str) -> None:
    problems: list[str] = []
    for name, expected in locked_versions().items():
        versions = actual.get(name, set())
        if versions != {expected}:
            rendered = ", ".join(sorted(versions)) if versions else "missing"
            problems.append(f"{name}: expected {expected}, found {rendered}")
    if problems:
        raise TermuxWheelhouseMismatch(
            f"{source} is incompatible with immutable Termux wheelhouse {RELEASE_TAG}: "
            + "; ".join(problems)
        )


def validate_requirements_text(text: str, *, source: str = "requirements") -> None:
    actual: dict[str, set[str]] = {}
    for raw in text.splitlines():
        match = _REQ_RE.match(raw.strip())
        if not match:
            continue
        name = canonicalize_name(match.group(1))
        actual.setdefault(name, set()).add(match.group(2))
    _check_versions(actual, source=source)


def validate_requirements_file(path: Path) -> None:
    validate_requirements_text(path.read_text(encoding="utf-8"), source=str(path))


def validate_uv_lock_text(text: str, *, source: str = "uv.lock") -> None:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise TermuxWheelhouseMismatch(f"Could not parse {source}: {exc}") from exc
    actual: dict[str, set[str]] = {}
    for package in data.get("package", []):
        raw_name = package.get("name")
        version = package.get("version")
        if not raw_name or not version:
            continue
        name = canonicalize_name(str(raw_name))
        if name in LOCKED_WHEELS:
            actual.setdefault(name, set()).add(str(version))
    _check_versions(actual, source=source)


def validate_uv_lock(path: Path) -> None:
    validate_uv_lock_text(path.read_text(encoding="utf-8"), source=str(path))


def _android_api_level() -> int | None:
    getter = getattr(sys, "getandroidapilevel", None)
    if callable(getter):
        try:
            return int(getter())
        except (TypeError, ValueError, OSError):
            return None
    return None


def validate_runtime(
    *,
    python_version: tuple[int, int] | None = None,
    machine: str | None = None,
    android_api: int | None = None,
    sys_platform: str | None = None,
) -> None:
    version = python_version or (sys.version_info.major, sys.version_info.minor)
    arch = (machine or platform.machine()).lower()
    platform_name = sys_platform or sys.platform
    api = _android_api_level() if android_api is None else android_api
    if platform_name != "android":
        raise TermuxWheelhouseUnsupported(
            f"immutable Termux wheels require Android Python, not {platform_name}"
        )
    if version != SUPPORTED_PYTHON:
        raise TermuxWheelhouseUnsupported(
            f"immutable Termux wheels require CPython 3.13 (current: {version[0]}.{version[1]})"
        )
    if arch not in SUPPORTED_ARCHES:
        raise TermuxWheelhouseUnsupported(
            f"immutable Termux wheels require arm64/aarch64, not {arch}"
        )
    if api is None:
        raise TermuxWheelhouseUnsupported("could not determine the Android API level")
    if api < MIN_ANDROID_API:
        raise TermuxWheelhouseUnsupported(
            f"immutable Termux wheels require Android API {MIN_ANDROID_API}+, not {api}"
        )


def parse_sha256sums(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            digest, filename = raw.split(maxsplit=1)
        except ValueError as exc:
            raise TermuxWheelhouseError(f"invalid SHA256SUMS line: {raw!r}") from exc
        digest = digest.lower()
        filename = filename.strip().lstrip("*")
        if (
            len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or Path(filename).name != filename
            or filename in parsed
        ):
            raise TermuxWheelhouseError(f"invalid SHA256SUMS line: {raw!r}")
        parsed[filename] = digest
    return parsed


def _release_marker() -> dict[str, object]:
    return {
        "repository": RELEASE_REPOSITORY,
        "tag": RELEASE_TAG,
        "commit": RELEASE_COMMIT,
        "sha256sums_sha256": SHA256SUMS_SHA256,
        "wheels": sorted(metadata[1] for metadata in LOCKED_WHEELS.values()),
    }


def verify_wheelhouse(path: Path) -> None:
    sums_path = path / "SHA256SUMS"
    marker_path = path / "release.json"
    if not sums_path.is_file() or not marker_path.is_file():
        raise TermuxWheelhouseError("wheelhouse metadata is incomplete")
    if sha256_file(sums_path) != SHA256SUMS_SHA256:
        raise TermuxWheelhouseError(
            "SHA256SUMS digest does not match the pinned release"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TermuxWheelhouseError("wheelhouse release marker is invalid") from exc
    if marker != _release_marker():
        raise TermuxWheelhouseError("wheelhouse release marker does not match the pin")
    sums = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    for _name, (_version, filename, expected) in LOCKED_WHEELS.items():
        if sums.get(filename) != expected:
            raise TermuxWheelhouseError(
                f"release checksum manifest mismatch: {filename}"
            )
        wheel = path / filename
        if not wheel.is_file() or sha256_file(wheel) != expected:
            raise TermuxWheelhouseError(f"wheel checksum mismatch: {filename}")


def _download(curl: str, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            curl,
            "-fL",
            "--retry",
            "3",
            "--retry-all-errors",
            "--proto",
            "=https",
            url,
            "-o",
            str(destination),
        ],
        check=True,
    )


def ensure_wheelhouse(
    cache_root: Path,
    *,
    requirements: Path | None = None,
    uv_lock: Path | None = None,
    curl: str | None = None,
    check_runtime: bool = True,
) -> Path:
    if requirements is not None:
        validate_requirements_file(requirements)
    if uv_lock is not None:
        validate_uv_lock(uv_lock)
    if check_runtime:
        validate_runtime()

    cache_root = cache_root.expanduser().resolve()
    final = cache_root / RELEASE_TAG
    try:
        verify_wheelhouse(final)
        return final
    except TermuxWheelhouseError:
        pass

    curl_path = curl or shutil.which("curl")
    if not curl_path:
        raise TermuxWheelhouseError(
            "curl is required to download immutable Termux wheels"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    stage = cache_root / f".{RELEASE_TAG}.tmp-{os.getpid()}"
    backup = cache_root / f".{RELEASE_TAG}.previous"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    try:
        sums_path = stage / "SHA256SUMS"
        _download(curl_path, f"{RELEASE_BASE_URL}/SHA256SUMS", sums_path)
        if sha256_file(sums_path) != SHA256SUMS_SHA256:
            raise TermuxWheelhouseError(
                "downloaded SHA256SUMS does not match the immutable release pin"
            )
        sums = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
        for _name, (_version, filename, expected) in LOCKED_WHEELS.items():
            if sums.get(filename) != expected:
                raise TermuxWheelhouseError(
                    f"immutable release manifest does not contain the pinned {filename}"
                )
            wheel = stage / filename
            _download(curl_path, f"{RELEASE_BASE_URL}/{filename}", wheel)
            if sha256_file(wheel) != expected:
                raise TermuxWheelhouseError(
                    f"downloaded wheel checksum mismatch: {filename}"
                )
        (stage / "release.json").write_text(
            json.dumps(_release_marker(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_wheelhouse(stage)

        shutil.rmtree(backup, ignore_errors=True)
        if final.exists():
            final.rename(backup)
        try:
            stage.rename(final)
        except BaseException:
            if backup.exists() and not final.exists():
                backup.rename(final)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return final
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def binary_install_options(wheelhouse: Path) -> list[str]:
    return ["--find-links", str(wheelhouse), "--only-binary", ":all:"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--uv-lock", type=Path)
    parser.add_argument("--curl")
    args = parser.parse_args(argv)
    try:
        path = ensure_wheelhouse(
            args.cache_root,
            requirements=args.requirements,
            uv_lock=args.uv_lock,
            curl=args.curl,
        )
    except TermuxWheelhouseUnsupported as exc:
        print(f"unsupported immutable wheel target: {exc}", file=sys.stderr)
        return 2
    except (TermuxWheelhouseError, subprocess.CalledProcessError, OSError) as exc:
        print(f"immutable Termux wheelhouse failed: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
