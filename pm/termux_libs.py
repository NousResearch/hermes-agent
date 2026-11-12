"""Termux pool pins: the runtime-lib table and lock.json's bionic rows.

``scripts/termux/stage_runtime_libs.py`` merges their .debs into one flat
``lib/`` inside the sealed Termux payload, and ``scripts/ci/archive_inputs.py``
mirrors the same bytes into R2. Both are chained to the pin table that lives
beside this module.

The pool is rolling: rebuilding a package DELETES the previous archive from
``pool/main/``, so a pin rots silently until a payload build downloads it and
gets a 404. The library set is the payload's recursive DT_NEEDED closure, so a
pin may not simply track "newest" — a repin is a repair:

    hermes pm update --termux --check   # what the pool no longer serves
    hermes pm update --termux           # repin exactly those rows

Nothing here resolves versions. "Move to the newest" is a different job with a
different blast radius (a repin can land a rebuilt library under a moved
soname), so an alive pin is left alone even when the pool has moved on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from pm.store import hash_url
from pm.update import _get_text

POOL = "https://packages.termux.dev/apt/termux-main/"
POOL_HOST = urlsplit(POOL).hostname
INDEX_URL = f"{POOL}dists/stable/main/binary-aarch64/Packages"
POOL_PATH_PREFIX = "/apt/termux-main/pool/main/"

TABLE_NAME = "termux_runtime_libs.json"


class PinMismatch(RuntimeError):
    """The pool serves bytes the package index does not describe."""


def table_path() -> Path:
    return Path(__file__).resolve().parent / TABLE_NAME


def load_table(path: Optional[Path] = None) -> dict:
    return json.loads(Path(path or table_path()).read_text(encoding="utf-8-sig"))


def save_table(table: dict, path: Optional[Path] = None) -> None:
    Path(path or table_path()).write_text(
        json.dumps(table, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


# ---------------------------------------------------------------------------
# The pool's own index (pure parse + one cached GET)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolPackage:
    name: str
    version: str
    url: str
    sha256: str


_FIELD = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):[ ]?(.*)$")


def parse_index(text: str) -> dict[str, PoolPackage]:
    """Package -> the archive the pool currently serves.

    RFC822-ish stanzas: one field per line, indented continuation lines
    (long descriptions) ignored. A stanza without a filename or hash is not
    installable evidence and is dropped.
    """
    pool: dict[str, PoolPackage] = {}
    fields: dict[str, str] = {}
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            row = _package(fields)
            if row is not None:
                pool[row.name] = row
            fields = {}
        elif not line[0].isspace():
            match = _FIELD.match(line)
            if match:
                fields[match.group(1)] = match.group(2).strip()
    return pool


def _package(fields: dict[str, str]) -> Optional[PoolPackage]:
    name = fields.get("Package")
    filename = fields.get("Filename")
    sha256 = fields.get("SHA256")
    if not name or not filename or not sha256:
        return None
    return PoolPackage(name, fields.get("Version", ""), POOL + filename, sha256)


def index() -> dict[str, PoolPackage]:
    return parse_index(_get_text(INDEX_URL))


def package_of(url: str) -> Optional[str]:
    """The pool package a pinned URL names, or None for any other source."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != POOL_HOST:
        return None
    if not parsed.path.startswith(POOL_PATH_PREFIX):
        return None
    parts = parsed.path[len(POOL_PATH_PREFIX):].split("/")
    if len(parts) != 3 or not all(parts):
        return None
    return parts[1]


def _filename(url: str) -> str:
    return url.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Pins, and which of them the pool has retired
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pin:
    scope: str  # "library", "license", or "tool"
    name: str
    url: str
    version: str


@dataclass(frozen=True)
class Retired:
    pin: Pin
    # None: the pool no longer carries the package at all — a rename or a
    # drop, which needs a human, not a repin.
    replacement: Optional[PoolPackage]


def pins(table: dict, lockfile) -> list[Pin]:
    """Every pinned URL the Termux pool serves, from both authorities.

    The table owns the payload's libraries and its license package; the
    lockfile owns the cross-target tools whose bionic arm is a pool .deb
    (python, node, uv). Other lock sources — GitHub tarballs, OCI digests —
    are not the pool's business.
    """
    out = [
        Pin("library", name, row["url"], row.get("version", ""))
        for name, row in (table.get("libs") or {}).items()
    ]
    licenses = table.get("licenses")
    if licenses:
        out.append(Pin("license", "termux-licenses", licenses["url"], licenses.get("version", "")))
    for name in lockfile.names():
        for target, rows in lockfile.pinned_artifacts(name).items():
            for row in rows if isinstance(rows, list) else [rows]:
                if isinstance(row, dict) and package_of(row.get("url", "")):
                    out.append(Pin("tool", f"{name}@{target}", row["url"], lockfile.version(name) or ""))
    return out


def retired(table: dict, lockfile, pool: dict[str, PoolPackage]) -> list[Retired]:
    """Pins whose archive the pool has replaced (or dropped)."""
    out = []
    for pin in pins(table, lockfile):
        entry = pool.get(package_of(pin.url) or "")
        if entry is None or _filename(entry.url) != _filename(pin.url):
            out.append(Retired(pin, entry))
    return out


def repair(table: dict, lockfile, stale: list[Retired], *, verify=None) -> int:
    """Repin every retired row to the pool's current archive.

    The index's SHA256 is not taken on faith: the replacement's bytes are
    fetched and hashed first — the same evidence `pm lock` pins with. Rows
    the pool no longer carries are left exactly as they are.
    """
    verify = verify or hash_url
    applied = 0
    for entry in stale:
        if entry.replacement is None:
            continue
        served = verify(entry.replacement.url)
        if served != entry.replacement.sha256:
            raise PinMismatch(
                f"{package_of(entry.pin.url)}: the pool serves {served}, "
                f"the index declares {entry.replacement.sha256}"
            )
        _repoint(table, lockfile, entry)
        applied += 1
    return applied


def _repoint(table: dict, lockfile, entry: Retired) -> None:
    pin, replacement = entry.pin, entry.replacement
    if pin.scope == "tool":
        name, target = pin.name.split("@", 1)
        artifacts = lockfile.pinned_artifacts(name)
        rows = artifacts[target]
        for row in rows if isinstance(rows, list) else [rows]:
            if package_of(row.get("url", "")) == package_of(pin.url):
                row["url"], row["sha256"] = replacement.url, replacement.sha256
        lockfile.set_pin(name, lockfile.version(name), artifacts)
        return
    row = table["libs"][pin.name] if pin.scope == "library" else table["licenses"]
    row["url"], row["sha256"], row["version"] = (
        replacement.url,
        replacement.sha256,
        replacement.version,
    )
