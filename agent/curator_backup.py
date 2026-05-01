"""Curator snapshot + rollback.

A pre-run snapshot of ``~/.hermes/skills/`` (excluding ``.curator_backups/``
itself) is taken before any mutating curator pass. Snapshots are tar.gz
files under ``~/.hermes/skills/.curator_backups/<utc-iso>/`` with a
companion ``manifest.json`` describing the snapshot (reason, time, size,
counted skill files). Rollback picks a snapshot, moves the current
``skills/`` tree aside into another snapshot so even the rollback itself
is undoable, then extracts the chosen snapshot into place.

The snapshot does NOT include:
  - ``.curator_backups/`` (would recurse)
  - ``.hub/`` (hub-installed skills — managed by the hub, not us)

It DOES include:
  - all SKILL.md files + their directories (``scripts/``, ``references/``,
    ``templates/``, ``assets/``)
  - ``.usage.json`` (usage telemetry — needed to rehydrate state cleanly)
  - ``.archive/`` (so rollback restores previously-archived skills too)
  - ``.curator_state`` (so rolling back also restores the last-run-at
    pointer — otherwise the curator would immediately re-fire on the next
    tick)
  - ``.bundled_manifest`` (so protection markers stay consistent)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


DEFAULT_KEEP = 5

# Entries under skills/ that should NEVER be rolled up into a snapshot.
# .hub/ is managed by the skills hub; rolling it back would break lockfile
# invariants. .curator_backups is the backup dir itself — recursion bomb.
_EXCLUDE_TOP_LEVEL = {".curator_backups", ".hub"}

# Snapshot id regex: UTC ISO with colons replaced by dashes so the filename
# is portable (Windows-safe). An optional ``-NN`` suffix handles two
# snapshots landing in the same wallclock second.
_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z(-\d{2})?$")


def _backups_dir() -> Path:
    return get_hermes_home() / "skills" / ".curator_backups"


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _utc_id(now: Optional[datetime] = None) -> str:
    """UTC ISO-ish filesystem-safe timestamp: ``2026-05-01T13-05-42Z``."""
    if now is None:
        now = datetime.now(timezone.utc)
    # isoformat → "2026-05-01T13:05:42.123456+00:00"; strip subseconds and tz.
    s = now.replace(microsecond=0).isoformat()
    if s.endswith("+00:00"):
        s = s[:-6]
    return s.replace(":", "-") + "Z"


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as e:
        logger.debug("Failed to load config for curator backup: %s", e)
        return {}
    if not isinstance(cfg, dict):
        return {}
    cur = cfg.get("curator") or {}
    if not isinstance(cur, dict):
        return {}
    bk = cur.get("backup") or {}
    return bk if isinstance(bk, dict) else {}


def is_enabled() -> bool:
    """Default ON — the whole point of the backup is safety by default."""
    return bool(_load_config().get("enabled", True))


def get_keep() -> int:
    cfg = _load_config()
    try:
        n = int(cfg.get("keep", DEFAULT_KEEP))
    except (TypeError, ValueError):
        n = DEFAULT_KEEP
    return max(1, n)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _count_skill_files(base: Path) -> int:
    try:
        return sum(1 for _ in base.rglob("SKILL.md"))
    except OSError:
        return 0


def _write_manifest(dest: Path, reason: str, archive_path: Path,
                    skills_counted: int) -> None:
    manifest = {
        "id": dest.name,
        "reason": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "skill_files": skills_counted,
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def snapshot_skills(reason: str = "manual") -> Optional[Path]:
    """Create a tar.gz snapshot of ``~/.hermes/skills/`` and prune old ones.

    Returns the snapshot directory path, or ``None`` if the snapshot was
    skipped (backup disabled, skills dir missing, or an IO error occurred —
    in which case we log at debug and return None so the curator never
    aborts a pass because of a backup failure).
    """
    if not is_enabled():
        logger.debug("Curator backup disabled by config; skipping snapshot")
        return None

    skills = _skills_dir()
    if not skills.exists():
        logger.debug("No ~/.hermes/skills/ directory — nothing to back up")
        return None

    backups = _backups_dir()
    try:
        backups.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("Failed to create backups dir %s: %s", backups, e)
        return None

    # Uniquify: if a snapshot with the same second already exists (can
    # happen if two curator runs fire in the same second), append a short
    # counter. Avoids clobbering and avoids timestamp collisions.
    base_id = _utc_id()
    snap_id = base_id
    counter = 1
    while (backups / snap_id).exists():
        snap_id = f"{base_id}-{counter:02d}"
        counter += 1

    dest = backups / snap_id
    try:
        dest.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        logger.debug("Failed to create snapshot dir %s: %s", dest, e)
        return None

    archive = dest / "skills.tar.gz"
    try:
        # Stream into the tarball — no tempdir copy needed.
        with tarfile.open(archive, "w:gz", compresslevel=6) as tf:
            for entry in sorted(skills.iterdir()):
                if entry.name in _EXCLUDE_TOP_LEVEL:
                    continue
                # arcname: store paths relative to skills/ so extraction
                # drops cleanly back into the skills dir.
                tf.add(str(entry), arcname=entry.name, recursive=True)
        _write_manifest(dest, reason, archive, _count_skill_files(skills))
    except (OSError, tarfile.TarError) as e:
        logger.debug("Curator snapshot failed: %s", e, exc_info=True)
        # Clean up partial snapshot
        try:
            shutil.rmtree(dest, ignore_errors=True)
        except OSError:
            pass
        return None

    _prune_old(keep=get_keep())
    logger.info("Curator snapshot created: %s (%s)", snap_id, reason)
    return dest


def _prune_old(keep: int) -> List[str]:
    """Delete regular snapshots beyond the newest *keep*. Returns deleted
    ids. Staging dirs (``.rollback-staging-*``) are implementation detail
    and pruned independently on every call."""
    backups = _backups_dir()
    if not backups.exists():
        return []
    entries: List[Tuple[str, Path]] = []
    stale_staging: List[Path] = []
    for child in backups.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith(".rollback-staging-"):
            # Staging dirs are only supposed to exist briefly during a
            # rollback. If we find one here (e.g. from a crashed rollback),
            # clean it up opportunistically.
            stale_staging.append(child)
            continue
        if _ID_RE.match(child.name):
            entries.append((child.name, child))
    # Newest first (lexicographic works because the id is UTC ISO).
    entries.sort(key=lambda t: t[0], reverse=True)
    deleted: List[str] = []
    for _, path in entries[keep:]:
        try:
            shutil.rmtree(path)
            deleted.append(path.name)
        except OSError as e:
            logger.debug("Failed to prune %s: %s", path, e)
    for path in stale_staging:
        try:
            shutil.rmtree(path)
        except OSError as e:
            logger.debug("Failed to clean stale staging dir %s: %s", path, e)
    return deleted


# ---------------------------------------------------------------------------
# List + rollback
# ---------------------------------------------------------------------------

def _read_manifest(snap_dir: Path) -> Dict[str, Any]:
    mf = snap_dir / "manifest.json"
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def list_backups() -> List[Dict[str, Any]]:
    """Return all restorable snapshots, newest first. Only entries with a
    real ``skills.tar.gz`` tarball are listed — transient
    ``.rollback-staging-*`` directories created mid-rollback are
    implementation detail and not shown."""
    backups = _backups_dir()
    if not backups.exists():
        return []
    out: List[Dict[str, Any]] = []
    for child in sorted(backups.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if not _ID_RE.match(child.name):
            continue
        if not (child / "skills.tar.gz").exists():
            continue
        mf = _read_manifest(child)
        mf.setdefault("id", child.name)
        mf.setdefault("path", str(child))
        if "archive_bytes" not in mf:
            arc = child / "skills.tar.gz"
            try:
                mf["archive_bytes"] = arc.stat().st_size
            except OSError:
                mf["archive_bytes"] = 0
        out.append(mf)
    return out


def _resolve_backup(backup_id: Optional[str]) -> Optional[Path]:
    """Return the path of the requested backup, or the newest one if
    *backup_id* is None. Returns None if no match."""
    backups = _backups_dir()
    if not backups.exists():
        return None
    if backup_id:
        target = backups / backup_id
        if (
            target.is_dir()
            and _ID_RE.match(backup_id)
            and (target / "skills.tar.gz").exists()
        ):
            return target
        return None
    candidates = [
        c for c in sorted(backups.iterdir(), reverse=True)
        if c.is_dir() and _ID_RE.match(c.name) and (c / "skills.tar.gz").exists()
    ]
    return candidates[0] if candidates else None


def rollback(backup_id: Optional[str] = None) -> Tuple[bool, str, Optional[Path]]:
    """Restore ``~/.hermes/skills/`` from a snapshot.

    Strategy:
      1. Resolve the target snapshot (explicit id or newest regular).
      2. Take a safety snapshot of the CURRENT skills tree under
         ``.curator_backups/pre-rollback-<ts>/`` so the rollback itself is
         undoable.
      3. Move all current top-level entries (except ``.curator_backups``
         and ``.hub``) into a tempdir.
      4. Extract the chosen snapshot into ``~/.hermes/skills/``.
      5. On failure during 4, move the tempdir contents back (best-effort)
         and return failure.

    Returns ``(ok, message, snapshot_path)``.
    """
    target = _resolve_backup(backup_id)
    if target is None:
        return (
            False,
            f"no matching backup found"
            + (f" for id '{backup_id}'" if backup_id else "")
            + " (use `hermes curator rollback --list` to see available snapshots)",
            None,
        )
    archive = target / "skills.tar.gz"
    if not archive.exists():
        return (False, f"snapshot {target.name} has no skills.tar.gz — corrupted?", None)

    skills = _skills_dir()
    skills.mkdir(parents=True, exist_ok=True)
    backups = _backups_dir()
    backups.mkdir(parents=True, exist_ok=True)

    # Step 2: safety snapshot of current state FIRST. If this fails we bail
    # out before touching anything — otherwise a failed extract could leave
    # the user with no skills.
    try:
        snapshot_skills(reason=f"pre-rollback to {target.name}")
    except Exception as e:
        return (False, f"pre-rollback safety snapshot failed: {e}", None)

    # Additionally move current entries into an internal staging dir so
    # the extract happens into an empty skills tree (predictable result).
    # This dir is implementation detail — not listed as a restorable
    # backup. The safety snapshot above is the user-facing undo handle.
    staged = backups / f".rollback-staging-{_utc_id()}"
    try:
        staged.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        return (False, f"failed to create staging dir: {e}", None)

    moved: List[Tuple[Path, Path]] = []
    try:
        for entry in list(skills.iterdir()):
            if entry.name in _EXCLUDE_TOP_LEVEL:
                continue
            dest = staged / entry.name
            shutil.move(str(entry), str(dest))
            moved.append((entry, dest))
    except OSError as e:
        # Best-effort rollback of the move
        for orig, dest in moved:
            try:
                shutil.move(str(dest), str(orig))
            except OSError:
                pass
        try:
            shutil.rmtree(staged, ignore_errors=True)
        except OSError:
            pass
        return (False, f"failed to stage current skills: {e}", None)

    # Step 4: extract the snapshot into skills/
    try:
        with tarfile.open(archive, "r:gz") as tf:
            # Python 3.12+ supports filter='data' for safer extraction.
            # Fall back to the unfiltered call for older interpreters but
            # still reject absolute paths and .. components defensively.
            for member in tf.getmembers():
                name = member.name
                if name.startswith("/") or ".." in Path(name).parts:
                    raise tarfile.TarError(
                        f"refusing to extract unsafe path: {name!r}"
                    )
            try:
                tf.extractall(str(skills), filter="data")  # type: ignore[call-arg]
            except TypeError:
                # Python < 3.12 — no filter kwarg
                tf.extractall(str(skills))
    except (OSError, tarfile.TarError) as e:
        # Best-effort recover: move staged contents back
        for orig, dest in moved:
            try:
                shutil.move(str(dest), str(orig))
            except OSError:
                pass
        try:
            shutil.rmtree(staged, ignore_errors=True)
        except OSError:
            pass
        return (False, f"snapshot extract failed (state restored): {e}", None)

    # Extract succeeded — the staging dir has served its purpose. The
    # user's undo handle is the safety snapshot tarball we took earlier.
    try:
        shutil.rmtree(staged, ignore_errors=True)
    except OSError:
        pass

    logger.info("Curator rollback: restored from %s", target.name)
    return (True, f"restored from snapshot {target.name}", target)


# ---------------------------------------------------------------------------
# Human-readable summary for CLI
# ---------------------------------------------------------------------------

def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def summarize_backups() -> str:
    rows = list_backups()
    if not rows:
        return "No curator snapshots yet."
    lines = [f"{'id':<24}  {'reason':<40}  {'skills':>6}  {'size':>8}"]
    lines.append("─" * len(lines[0]))
    for r in rows:
        lines.append(
            f"{r.get('id','?'):<24}  "
            f"{(r.get('reason','?') or '?')[:40]:<40}  "
            f"{r.get('skill_files', 0):>6}  "
            f"{format_size(int(r.get('archive_bytes', 0))):>8}"
        )
    return "\n".join(lines)
