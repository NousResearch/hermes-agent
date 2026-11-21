#!/usr/bin/env python3
"""Refresh the bionic python pin to the archive the Termux pool serves today.

The Termux pool rolls: a python rebuild deletes the previous archive, so a
bionic pin is correct only until the next rebuild. ``pm/packages.py`` says so
in as many words -- the bionic row is a deliberate manual pin, derived from
nothing, and ``pm update`` leaves it alone.

That leaves the pin's maintenance entirely manual, and it is spread across
four files. This script is the missing half: it resolves the current archive
from the signed pool index and rewrites every place that names it, so a
rotation is one command instead of five coordinated edits.

    python3 scripts/termux/refresh_bionic_pin.py            # rewrite the pin
    python3 scripts/termux/refresh_bionic_pin.py --check    # report drift only

``--check`` exits non-zero when the committed pin no longer matches the pool
(or the snapshot), so it can be used as a scheduled job without touching
anything.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDEX = (
    "https://packages.termux.dev/apt/termux-main/dists/stable/main/"
    "binary-aarch64/Packages.gz"
)
POOL_BASE = "https://packages.termux.dev/apt/termux-main/"

# The four places that name the pinned interpreter. Each entry is
# (path, list of (pattern, replacement-template)) where the template is
# formatted with the resolved version.
LOCK_JSON = REPO_ROOT / "pm" / "lock.json"
PACKAGES_PY = REPO_ROOT / "pm" / "packages.py"
BUILD_WHEELS_PY = REPO_ROOT / "scripts" / "termux" / "build_wheels.py"
TERMUX_BUILD_SH = REPO_ROOT / "scripts" / "termux" / "termux_build.sh"
SNAPSHOT = REPO_ROOT / "tests" / "pm" / "fixtures" / "termux_main_python_bionic.json"

_VERSION = r"[0-9]+\.[0-9]+\.[0-9]+(?:[.-][0-9]+)?"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-bionic-pin-refresh"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def resolve_from_index(index_url: str) -> dict:
    """Return the python stanza for aarch64 from the pool index."""
    raw = _fetch(index_url)
    try:
        text = gzip.decompress(raw).decode("utf-8", "replace")
    except OSError:
        text = raw.decode("utf-8", "replace")
    for stanza in text.split("\n\n"):
        if not re.search(r"^Package: python$", stanza, re.M):
            continue
        fields = dict(re.findall(r"^([A-Za-z0-9-]+):[ \t]*(.*)$", stanza, re.M))
        if fields.get("Architecture") != "aarch64":
            continue
        version = fields["Version"]
        filename = fields["Filename"]
        return {
            "version": version,
            "filename": filename,
            "sha256": fields.get("SHA256", ""),
            "size": int(fields.get("Size", "0") or 0),
            "url": POOL_BASE + filename,
        }
    raise SystemExit("the pool index has no aarch64 python stanza")


def verify_digest(entry: dict) -> str:
    """Download the artifact and confirm the index digest.

    The index digest is the authority; downloading proves the artifact still
    matches it. Both are recorded so a later reader can tell which was used.
    """
    if not entry["sha256"]:
        return ""
    blob = _fetch(entry["url"])
    got = hashlib.sha256(blob).hexdigest()
    if got != entry["sha256"]:
        raise SystemExit(
            f"digest mismatch for {entry['filename']}: index says {entry['sha256']}, "
            f"artifact hashes to {got}"
        )
    return got


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def rewrite(entry: dict) -> list[str]:
    """Point every place that names the pinned interpreter at the new archive."""
    version = entry["version"]
    filename = entry["filename"].split("/")[-1]
    changed: list[str] = []

    # 1. pm/lock.json -- the artifact row the install path reads
    lock = _read(LOCK_JSON)
    new_lock, n = re.subn(
        r'("linux-arm64-bionic"\s*:\s*\{\s*"sha256"\s*:\s*")[0-9a-f]{64}("\s*,\s*"url"\s*:\s*")'
        r'[^"]*python_[^"]*_aarch64\.deb(")',
        lambda m: m.group(1) + entry["sha256"] + m.group(2) + entry["url"] + m.group(3),
        lock,
        count=1,
    )
    if n:
        _write(LOCK_JSON, new_lock)
        changed.append(f"pm/lock.json -> {filename}")

    # 2. pm/packages.py -- the class's manual bionic URL
    pkg = _read(PACKAGES_PY)
    new_pkg, n = re.subn(
        r'(_BIONIC_URL\s*=\s*\(\s*"[^"]*?"\s*\n\s*")python_[^"]*_aarch64\.deb(")',
        lambda m: m.group(1) + filename + m.group(2),
        pkg,
        count=1,
    )
    if n:
        _write(PACKAGES_PY, new_pkg)
        changed.append(f"pm/packages.py -> {filename}")

    # 3+4. the wheel TARGET_ENV in both build scripts
    for target in (BUILD_WHEELS_PY, TERMUX_BUILD_SH):
        text = _read(target)
        new_text, n = re.subn(
            r'("(?:implementation_version|python_full_version)":\s*")' + _VERSION + r'(")',
            lambda m: m.group(1) + version + m.group(2),
            text,
        )
        if n:
            _write(target, new_text)
            changed.append(f"{target.relative_to(REPO_ROOT)} -> {version}")

    # 5. the snapshot the offline contract test compares against
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "version": version,
        "filename": filename,
        "sha256": entry["sha256"],
        "size": entry["size"],
        "url": entry["url"],
    }
    _write(SNAPSHOT, json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    changed.append(f"{SNAPSHOT.relative_to(REPO_ROOT)} -> {version}")

    return changed


def committed() -> dict:
    """What the repository currently pins."""
    lock = json.loads(_read(LOCK_JSON))
    row = lock["packages"]["python"]["artifacts"]["linux-arm64-bionic"]
    pkg = _read(PACKAGES_PY)
    m = re.search(r'python_[^"]*_aarch64\.deb', pkg)
    env = dict(
        re.findall(r'"(python_full_version|implementation_version|python_version)":\s*"([^"]*)"',
                   _read(BUILD_WHEELS_PY))
    )
    shell = re.findall(r'"(?:python_full_version|implementation_version)":\s*"([^"]*)"',
                        _read(TERMUX_BUILD_SH))
    return {
        "lock_url": row["url"],
        "lock_sha256": row["sha256"],
        "packages_py_url": m.group(0) if m else "",
        "wheel_env": env,
        "shell_versions": sorted(set(shell)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=DEFAULT_INDEX, help="pool Packages.gz URL")
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero; change nothing")
    args = ap.parse_args()

    entry = resolve_from_index(args.index)
    digest = verify_digest(entry)
    if digest:
        entry["sha256"] = digest

    print(f"pool: {entry['version']}  {entry['filename']}  sha256={entry['sha256'][:16]}...")

    if args.check:
        state = committed()
        filename = entry["filename"].split("/")[-1]
        problems = []
        if state["lock_sha256"] != entry["sha256"]:
            problems.append(f"pm/lock.json pins {state['lock_sha256'][:16]}...")
        if state["packages_py_url"] and state["packages_py_url"] != filename:
            problems.append(f"pm/packages.py names {state['packages_py_url']}")
        for key, value in state["wheel_env"].items():
            if key == "python_version":
                continue
            if value != entry["version"]:
                problems.append(f"build_wheels TARGET_ENV[{key}]={value}")
        for value in state["shell_versions"]:
            if value != entry["version"]:
                problems.append(f"termux_build.sh TARGET_ENV={value}")
        if SNAPSHOT.exists():
            snap = json.loads(_read(SNAPSHOT))
            if snap.get("version") != entry["version"]:
                problems.append(f"snapshot={snap.get('version')}")
        if problems:
            print("DRIFT:")
            for p in problems:
                print("  -", p)
            return 1
        print("no drift: every place names the archive the pool serves")
        return 0

    for line in rewrite(entry):
        print("  rewrote", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
