"""Describe the desktop dependency snapshot using only the runner's stdlib.

Actions transports provider directories directly, including pnpm's
content-addressed store. Restored bytes are candidates: preparation must still
admit them. This module never provisions tools or certifies a cache hit as prepared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.pnpm_lock import section_keys
from scripts.ci.setup_toolchain import current_target, file_commands


CACHE_DOMAINS = ("tools", "python/runtime", "python/build", "pnpm/store", "native", "packager")
_INPUT_FILES = (
    "pnpm-lock.yaml", "pnpm-workspace.yaml",
    "pyproject.toml", "uv.lock", "pm/lock.json", "pm/pyproject.toml", "pm/uv.lock",
    "scripts/ci/desktop_build_cache.py", "scripts/ci/setup_toolchain.py", "scripts/ci/pnpm_lock.py",
    "scripts/build/node-deps.mjs", "scripts/build/electron-runtime.mjs", "scripts/build/pnpm-lock.mjs",
    "scripts/build/node-deps-lock.mjs", "scripts/build/node-deps-heartbeat.mjs",
    "scripts/build/node-deps-watcher.mjs", "scripts/build/node-deps-relay.mjs",
    "scripts/build/frontend-common.mjs", "scripts/build/python.mjs",
    "scripts/build/windows-deps.ps1", "scripts/windows-build-deps.ps1",
    "apps/desktop/scripts/stage-native-deps.mjs", "apps/desktop/scripts/prepare-packaging-tools.mjs",
    "apps/desktop/scripts/utils.mjs", "apps/desktop/scripts/macos-sysroot.mjs",
    "apps/desktop/scripts/build-command-screenshot-monitor.mjs",
    "apps/desktop/scripts/build-hud-modifier-monitor.mjs",
    "apps/desktop/electron/native/command-screenshot-monitor.m",
    "apps/desktop/electron/native/hud-modifier-gesture.h",
    "apps/desktop/electron/native/hud-modifier-gesture.cs",
    "apps/desktop/electron/native/hud-modifier-monitor.m",
    "apps/desktop/electron/native/hud-modifier-monitor-win.cs",
    "apps/desktop/electron/native/hud-modifier-monitor-x11.c",
    "apps/desktop/scripts/windows-bundle-tools.mjs", "apps/desktop/scripts/prepared-native-deps.mjs",
    "apps/desktop/scripts/prepared-packaging.mjs", "apps/desktop/scripts/prepare-dmgbuild.mjs",
    "apps/desktop/scripts/probe-prepared-native.mjs",
    # Shared Python modules preparation imports from outside the globbed packages below, mostly
    # inside functions. A new late import of such a module must be added here by hand.
    "scripts/releases/versioning.py", "scripts/releases/semver.py",
    "scripts/releases/bundle_env.py", "scripts/releases/commit_build.py", "scripts/termux/deb_version.py",
    "hermes_cli/__init__.py", "hermes_cli/release_channels.py", "hermes_cli/update_channel.py",
    "hermes_cli/runtime_state.py", "hermes_constants.py", "hermes_platform/__init__.py",
    "hermes_platform/resolver/__init__.py", "hermes_platform/resolver/base.py",
    "hermes_platform/resolver/core.py", "hermes_platform/resolver/known_dirs.py",
)
# Python packages preparation owns (or, for pm/, depends on throughout): keyed whole, so a new
# module or a new late import inside them needs no list edit.
_INPUT_GLOBS = ("pm/*.py", "scripts/bundles/*.py", "scripts/build/*.py", "apps/desktop/scripts/*.py")


def input_files(source: Path, importers: list[str]) -> list[str]:
    """The repo-relative files the exact-input restore key hashes."""
    files: set[str] = set(_INPUT_FILES)
    for pattern in _INPUT_GLOBS:
        files.update(path.relative_to(source).as_posix() for path in source.glob(pattern))
    files.update((Path(path) / "package.json").as_posix() for path in importers)
    return sorted(files)


def _input_digest(source: Path, importers: list[str]) -> str:
    # A coarse lookup hint, not a second receipt or dependency resolver. Owner
    # admission remains necessary even when this digest matches exactly.
    digest = hashlib.sha256()
    for name in input_files(source, importers):
        path = source / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


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
    importers = [] if payload_only else section_keys(
        (source / "pnpm-lock.yaml").read_text(encoding="utf-8-sig"), "importers")
    # node_modules is never transported: it lives in the checkout, which no cache
    # path may contain. pnpm's content-addressed store (pnpm/store) is what saves
    # work on the next run.
    domains = ("tools", "python/runtime", "native") if payload_only else CACHE_DOMAINS
    paths = [_cache_path(cache / domain) for domain in domains]
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
    input_prefix = f"{prefix}{_input_digest(source, importers)}-"
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