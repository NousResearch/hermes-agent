"""Machine-local re-pins for archives a supplier retired under a current lock row.

``pm/lock.json`` ships inside a git checkout or a sealed payload. A repair
written there leaves ``hermes update`` a dirty tree to autostash (and a
conflict when upstream re-pins the same row), and a sealed payload cannot be
written at all. A re-pin therefore lives in machine state, keyed by the
sha256s of the lock row it replaces: once the shipped row moves, the entry
stops matching and the lockfile is authoritative again.
"""

from __future__ import annotations

import json
import logging
from pm import paths
from pm.downloader import DownloadError, DownloadTransportError

LOG = logging.getLogger(__name__)


def load() -> dict:
    try:
        data = json.loads(paths.repins_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def apply(table: dict, name: str, target: str, rows: list[dict]) -> list[dict]:
    """The re-pinned rows when ``rows`` is exactly the row the re-pin replaced.
    A malformed or non-matching entry is ignored, and a target the lock does
    not ship is never given one: the entry may only stand in for a real row."""
    package = table.get(name)
    entry = package.get(target) if isinstance(package, dict) else None
    if not rows or not isinstance(entry, dict) or entry.get("replaces") != [row["sha256"] for row in rows]:
        return rows
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts or not all(
        isinstance(row, dict) and isinstance(row.get("url"), str) and isinstance(row.get("sha256"), str)
        for row in artifacts
    ):
        return rows
    return [{"url": row["url"], "sha256": row["sha256"]} for row in artifacts]


def record(name: str, target: str, replaces: list[str], artifacts: list[dict]) -> dict:
    from pm.lock import _write

    table = load()
    if not isinstance(table.get(name), dict):
        table[name] = {}
    table[name][target] = {"replaces": replaces, "artifacts": artifacts}
    _write(paths.repins_path(), table)
    return table


def retired(exc: DownloadError, rows: list[dict]) -> bool:
    """A pinned url answered 404/410 and every other source refused outright:
    the archive is gone, not merely unreachable. One transient failure anywhere
    in the chain (a mirror that probed fine and then timed out) means the
    pinned bytes may still arrive, and an integrity failure is never a
    transport error, so neither is ever "repaired" into a different archive."""
    urls = {row["url"] for row in rows}
    chain = [failure for failure in (exc, *exc.failures) if isinstance(failure, DownloadTransportError)]
    return any(failure.url in urls and failure.status in (404, 410) for failure in chain) and all(
        failure.status in (401, 403, 404, 410) for failure in chain
    )


def repair(package, lockfile, version: str, target: str, exc: DownloadError) -> bool:
    """Re-pin a retired archive to the build the live index advertises now.

    The replacement keeps the pin's version line and target: the exact
    version for semver packages, the newest patch of the locked major.minor
    for ``version_style = "minor"`` (ffmpeg pins "9.0.1" but any 9.0.x
    satisfies it, and a rolling supplier stops building the old patch once
    the next one ships). The fact and store entry keep the locked label; the
    re-pinned urls carry the exact build. Returns True when ``lockfile`` now
    resolves fresh archives and the caller may retry once; any other outcome
    keeps the original failure.
    """
    from pm.update import best_in_minor, minor_of, pin_rows

    # The effective rows: a re-pin may itself retire. repin_locally keys the
    # new entry on the shipped rows, so it simply replaces the old one.
    current = lockfile.artifacts(package.name, target)
    if not retired(exc, current):
        return False
    exact = version
    try:
        if package.version_style == "minor":
            candidates = package.latest_versions(target, locked=version) or []
            exact = best_in_minor(candidates, minor_of(version)) or version
        urls = package.fetch_urls(exact, target)
        if not urls or urls == [row["url"] for row in current]:
            return False
        pinned = pin_rows(package, exact, urls, {row["url"]: row["sha256"] for row in current})
        lockfile.repin_locally(package.name, target, pinned)
    except Exception as error:  # any failure here must leave the original error standing
        LOG.warning("repair: %s %s %s has no usable replacement: %s", package.name, version, target, error)
        return False
    LOG.info("repair: %s %s re-pinned retired archives for %s to %s", package.name, version, target, exact)
    return True
