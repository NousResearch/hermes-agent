#!/usr/bin/env python3
"""Build and install Android-tagged wheels for known setuptools extensions."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli.android_wheels import (  # noqa: E402
    DEFAULT_ANDROID_API_LEVEL,
    AndroidWheelError,
    android_wheel_platform_tag,
    canonicalize_name,
    configure_setuptools_android_tag,
    patch_psutil_android_detection,
    pypi_sdist,
    safe_extract_sdist,
)

DEFAULT_PACKAGES = (
    "psutil",
    "markupsafe",
    "pyyaml",
    "cffi",
    "pillow",
    "ruamel-yaml-clib",
)
_REQ_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")


def versions_from_requirements(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = _REQ_RE.match(line)
        if match:
            versions[canonicalize_name(match.group(1))] = match.group(2)
    return versions


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_one(
    *, name: str, version: str, uv: str, python: str, platform_tag: str, work: Path
) -> Path:
    package_dir = work / canonicalize_name(name)
    archive = pypi_sdist(name, version, package_dir / "download")
    src_root = safe_extract_sdist(archive, package_dir / "src")
    if canonicalize_name(name) == "psutil":
        patch_psutil_android_detection(src_root)
    configure_setuptools_android_tag(src_root, platform_tag)

    out = package_dir / "wheelhouse"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["UV_PYTHON"] = python
    env.pop("_PYTHON_HOST_PLATFORM", None)
    command = [
        uv,
        "build",
        "--wheel",
        "--no-build-isolation",
        "--out-dir",
        str(out),
        str(src_root),
    ]
    print("  $", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)
    wheels = sorted(out.glob("*.whl"))
    if len(wheels) != 1:
        raise AndroidWheelError(
            f"Expected one wheel for {name}=={version}, found {len(wheels)}"
        )
    wheel = wheels[0]
    if platform_tag not in wheel.name:
        raise AndroidWheelError(
            f"{name} built an incompatible wheel {wheel.name}; expected {platform_tag}"
        )
    return wheel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument(
        "--android-api-level", type=int, default=DEFAULT_ANDROID_API_LEVEL
    )
    parser.add_argument("--package", action="append", dest="packages")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)

    versions = versions_from_requirements(args.requirements)
    requested = tuple(args.packages or DEFAULT_PACKAGES)
    platform_tag = android_wheel_platform_tag(api_level=args.android_api_level)
    owned_temp = None
    if args.work_dir is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="hermes-android-wheels-")
        work = Path(owned_temp.name)
    else:
        work = args.work_dir
        work.mkdir(parents=True, exist_ok=True)

    try:
        for raw_name in requested:
            name = canonicalize_name(raw_name)
            version = versions.get(name)
            if version is None:
                print(f"→ {name}: not in resolved Termux requirements; skipping")
                continue
            if installed_version(name) == version:
                print(f"✓ {name}=={version} already installed")
                continue
            print(f"→ Building {name}=={version} for {platform_tag}")
            wheel = build_one(
                name=name,
                version=version,
                uv=args.uv,
                python=args.python,
                platform_tag=platform_tag,
                work=work,
            )
            command = [
                args.uv,
                "pip",
                "install",
                "--python",
                args.python,
                "--no-deps",
                str(wheel),
            ]
            print("  $", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            print(f"✓ Installed {wheel.name}")
    except (AndroidWheelError, subprocess.CalledProcessError) as exc:
        print(f"Android wheel build failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if owned_temp is not None:
            owned_temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
