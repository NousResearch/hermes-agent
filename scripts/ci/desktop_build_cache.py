"""Describe the desktop dependency snapshot using only the runner's stdlib.

Actions transports provider directories directly, including npm's existing
receipt. Restored bytes are candidates: preparation must still admit them.
This module never provisions tools or certifies a cache hit as prepared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.setup_toolchain import current_target, file_commands


CACHE_DOMAINS = ("tools", "python/runtime", "python/build", "npm/_cacache", "native", "packager")
_INPUT_FILES = (
    "package-lock.json", "pyproject.toml", "uv.lock", "pm/lock.json", "pm/pyproject.toml", "pm/uv.lock",
    "scripts/ci/desktop_build_cache.py", "scripts/ci/setup_toolchain.py",
    "scripts/bundles/desktop_prepare.py", "scripts/bundles/desktop_toolchain.py",
    "scripts/bundles/desktop_inputs.py", "scripts/bundles/native.py", "scripts/bundles/native_prepared.py",
    "scripts/bundles/native_build.py",
    "scripts/build/node-deps.mjs", "scripts/build/icon_environment.py",
    "scripts/build/windows_deps.py", "scripts/windows-build-deps.ps1",
    "apps/desktop/scripts/stage-native-deps.mjs", "apps/desktop/scripts/prepare-packaging-tools.mjs",
    "apps/desktop/scripts/windows-bundle-tools.mjs", "apps/desktop/scripts/prepared-native-deps.mjs",
    "apps/desktop/scripts/prepared-packaging.mjs", "apps/desktop/scripts/prepare-dmgbuild.mjs",
    "apps/desktop/scripts/prepare_dmgbuild.py", "apps/desktop/scripts/probe-prepared-native.mjs",
)


def _input_digest(source: Path, lock: dict) -> str:
    # A coarse lookup hint, not a second receipt or dependency resolver. Owner
    # admission remains necessary even when this digest matches exactly.
    files: set[str] = set(_INPUT_FILES)
    files.update(str(path.relative_to(source)) for path in (source / "pm").glob("*.py"))
    files.update(str(Path(path) / "package.json") for path in lock["packages"]
                 if "node_modules" not in path.split("/"))
    digest = hashlib.sha256()
    for name in sorted(files):
        path = source / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _workspace_path(source: Path, name: str) -> None:
    if (not isinstance(name, str) or not name or
            any(char in name for char in "\\:*?[]!\r\n\0") or
            PurePosixPath(name).is_absolute() or
            any(part in ("", ".", "..", "node_modules") for part in name.split("/"))):
        raise ValueError(f"invalid locked workspace path: {name!r}")
    path = source / name
    if not path.resolve().is_relative_to(source) or not (path / "package.json").is_file():
        raise ValueError(f"missing or foreign locked workspace: {name!r}")


def _cache_path(path: Path) -> str:
    if any(char in str(path) for char in "*?[]!\r\n\0") or path.resolve() != path:
        raise ValueError(f"unsafe cache path: {path}")
    return str(path)


def describe_cache(source: Path, cache: Path, producer: str, *, work: Path | None = None) -> dict:
    source, cache = source.resolve(), cache.resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", producer):
        raise ValueError("producer must be a nonempty cache namespace")
    if os.environ.get("HERMES_HOME") and cache.is_relative_to(Path(os.environ["HERMES_HOME"]).resolve()):
        raise ValueError("cache must be separate from HERMES_HOME")
    payload_only = producer == "payload-test"
    lock = {"packages": {}} if payload_only else json.loads(
        (source / "package-lock.json").read_text(encoding="utf-8-sig"))
    # Match the Node owner's locked layout without needing Node before restore.
    workspaces = sorted({entry["resolved"] for entry in lock["packages"].values() if entry.get("link")})
    for workspace in set(workspaces) | {name for name in lock["packages"] if name and "node_modules" not in name.split("/")}:
        _workspace_path(source, workspace)
    domains = ("tools", "python/runtime", "native") if payload_only else CACHE_DOMAINS
    paths = [_cache_path(cache / domain) for domain in domains]
    if not payload_only:
        paths += [_cache_path(source / workspace / "node_modules") for workspace in ["", *workspaces]]
    private_roots = {"source": source, "work": work,
                     "HERMES_HOME": Path(os.environ["HERMES_HOME"]) if os.environ.get("HERMES_HOME") else None}
    for name, private in private_roots.items():
        if private is not None and any(private.resolve().is_relative_to(Path(path)) for path in paths):
            raise ValueError(f"{name} must be outside every reusable cache path")
    target = current_target()
    host = hashlib.sha256(json.dumps([
        platform.platform(), os.environ.get("ImageOS", ""), os.environ.get("ImageVersion", ""),
    ]).encode()).hexdigest()[:16]
    prefix = f"desktop-inputs-v1-{producer}-{target}-{host}-"
    input_prefix = f"{prefix}{_input_digest(source, lock)}-"
    run = "-".join(os.environ.get(name, default) for name, default in (
        ("GITHUB_RUN_ID", "local"), ("GITHUB_RUN_ATTEMPT", "1"), ("GITHUB_JOB", "desktop"),
    ))
    return {"target": target, "paths": paths, "restore_keys": [input_prefix, prefix],
            "key": input_prefix + run}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["describe"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--producer", required=True)
    args = parser.parse_args()
    description = describe_cache(args.source, args.cache, args.producer, work=args.work)
    if os.environ.get("GITHUB_OUTPUT"):
        file_commands("GITHUB_OUTPUT", {
            "cache-key": description["key"], "cache-paths": json.dumps(description["paths"]),
            "input-prefix": description["restore_keys"][0], "restore-prefix": description["restore_keys"][1],
        })
    print(json.dumps(description))


if __name__ == "__main__":
    main()