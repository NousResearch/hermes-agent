"""Curator snapshot + rollback. Before any mutating curator pass, ``~/.hermes/skills/`` is tar.gz'd under
``~/.hermes/skills/.curator_backups/<utc-iso>/`` with a ``manifest.json``. Rollback first snapshots the CURRENT tree (so it is
itself undoable), then extracts the chosen snapshot into place. Excluded: ``.curator_backups/``, ``.hub/`` (hub-managed), ``.git/``.
Included: skill dirs, ``.usage.json``, ``.archive/``, ``.curator_state`` (so rollback also restores last-run-at and the curator
doesn't re-fire), ``.bundled_manifest``, ``.curator_suppressed``. Each snapshot also copies ``~/.hermes/cron/jobs.json`` as
``cron-jobs.json``: the consolidation pass rewrites cron ``skills``/``skill`` references in place, so rollback restores those two
fields (only) — the rest is live state."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import tarfile
from datetime import datetime, timezone
from itertools import chain, count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from agent.skill_utils import is_excluded_skill_path
from agent.curator import _read_config_section
from hermes_cli.sizefmt import format_bytes

logger = logging.getLogger(__name__)

DEFAULT_KEEP = 5

# Never rolled into a snapshot: .hub/ is owned by the skills hub (rolling it back breaks lockfile invariants); .curator_backups
# is the backup dir itself; .git is repository metadata — rolling it back breaks git tracking, and snapshots that include it grow
# with the full history (once backups are committed back, each snapshot contains the prior ones: 38MB of skills inflated to 24GB
# in weeks). The tar filter in ``snapshot_skills`` applies the same set to nested paths, so a nested ``.git`` is skipped too.
# See #91449.
_EXCLUDE_TOP_LEVEL = {".curator_backups", ".hub", ".git"}

# Snapshot id: UTC ISO with colons replaced by dashes (Windows-safe filename); optional ``-NN`` suffix for same-second snapshots.
_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z(-\d{2})?$")

CRON_JOBS_FILENAME = "cron-jobs.json"
_ARCHIVE_NAME = "skills.tar.gz"
_STAGING_PREFIX = ".rollback-staging-"


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _backups_dir() -> Path:
    return _skills_dir() / ".curator_backups"


def _jobs_list(parsed: Any) -> Optional[list]:
    """jobs.json is ``{"jobs": [...], "updated_at": ...}``; also accept a bare list for forward compat. None otherwise."""
    parsed = parsed.get("jobs") if isinstance(parsed, dict) else parsed
    return parsed if isinstance(parsed, list) else None


def _backup_cron_jobs_into(dest: Path) -> Dict[str, Any]:
    """Copy the live ``~/.hermes/cron/jobs.json`` into ``dest`` as ``cron-jobs.json``. Never raises: a missing/unreadable
    file yields ``backed_up=False`` plus a reason, and the snapshot proceeds."""
    src = get_hermes_home() / "cron" / "jobs.json"
    info: Dict[str, Any] = {"backed_up": False, "jobs_count": 0}
    if not src.exists():
        return {**info, "reason": "no cron/jobs.json present"}
    try:
        # utf-8-sig, same dialect as cron/jobs.load_jobs: a Windows-editor BOM would otherwise break json.loads
        # AND be written into the backup.
        raw = src.read_text(encoding="utf-8-sig")
    except OSError as e:
        logger.debug("Failed to read cron/jobs.json for backup: %s", e)
        return {**info, "reason": f"read error: {e}"}
    try:  # jobs_count is a diagnostic only — an unparseable file is still stored raw.
        info["jobs_count"] = len(_jobs_list(json.loads(raw)) or [])
    except (json.JSONDecodeError, TypeError):
        info["parse_warning"] = "jobs.json was not valid JSON at snapshot time"
    try:
        (dest / CRON_JOBS_FILENAME).write_text(raw, encoding="utf-8")
    except OSError as e:
        logger.debug("Failed to write cron backup file: %s", e)
        return {**info, "reason": f"write error: {e}"}
    return {**info, "backed_up": True}


def _utc_id(now: Optional[datetime] = None) -> str:
    """UTC ISO-ish filesystem-safe timestamp: ``2026-05-01T13-05-42Z``."""
    s = (datetime.now(timezone.utc) if now is None else now).replace(microsecond=0).isoformat()
    return s.removesuffix("+00:00").replace(":", "-") + "Z"


def _load_config() -> Dict[str, Any]:
    return _read_config_section("curator", "backup", label="curator backup", log=logger)


def is_enabled() -> bool:
    """Default ON — the whole point of the backup is safety by default."""
    return bool(_load_config().get("enabled", True))


def get_keep() -> int:
    try:
        return max(1, int(_load_config().get("keep", DEFAULT_KEEP)))
    except (TypeError, ValueError):
        return DEFAULT_KEEP


# --- Snapshot ---
def _count_skill_files(base: Path) -> int:
    try:
        return sum(1 for p in base.rglob("SKILL.md") if not is_excluded_skill_path(p))
    except OSError:
        return 0


def _write_manifest(dest: Path, reason: str, archive_path: Path, skills_counted: int, cron_info: Dict[str, Any]) -> None:
    cron_jobs: Dict[str, Any] = {"backed_up": bool(cron_info.get("backed_up", False)), "jobs_count": int(cron_info.get("jobs_count", 0))}
    if not cron_info.get("backed_up"):
        cron_jobs["reason"] = cron_info.get("reason", "not captured")
    if cron_info.get("parse_warning"):
        cron_jobs["parse_warning"] = cron_info["parse_warning"]
    manifest = {"id": dest.name, "reason": reason, "created_at": datetime.now(timezone.utc).isoformat(), "archive": archive_path.name,
                "archive_bytes": archive_path.stat().st_size, "skill_files": skills_counted, "cron_jobs": cron_jobs}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _mkdir(path: Path, what: str, *, exist_ok: bool) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=exist_ok)
        return True
    except OSError as e:
        logger.debug("Failed to create %s %s: %s", what, path, e)
        return False


def snapshot_skills(reason: str = "manual", *, protect_ids: Optional[Set[str]] = None) -> Optional[Path]:
    """Create a tar.gz snapshot of ``~/.hermes/skills/`` and prune old ones. Returns the snapshot dir, or None when
    skipped (disabled, skills dir missing, IO error) — logged at debug so the curator never aborts a pass over a
    backup failure. ``protect_ids`` survive the prune step (rollback protects its target)."""
    if not is_enabled():
        logger.debug("Curator backup disabled by config; skipping snapshot")
        return None
    skills, backups = _skills_dir(), _backups_dir()
    if not skills.exists():
        logger.debug("No ~/.hermes/skills/ directory — nothing to back up")
        return None
    if not _mkdir(backups, "backups dir", exist_ok=True):
        return None

    base_id = _utc_id()  # Two curator runs in the same second must not clobber each other: -NN suffix.
    snap_id = next(i for i in chain([base_id], (f"{base_id}-{n:02d}" for n in count(1))) if not (backups / i).exists())
    dest = backups / snap_id
    if not _mkdir(dest, "snapshot dir", exist_ok=False):
        return None

    archive = dest / _ARCHIVE_NAME
    try:
        with tarfile.open(archive, "w:gz", compresslevel=6) as tf:
            for entry in sorted(skills.iterdir()):
                if entry.name not in _EXCLUDE_TOP_LEVEL:
                    # arcname relative to skills/ so extraction drops back in cleanly; the filter excludes nested _EXCLUDE_TOP_LEVEL paths too.
                    tf.add(str(entry), arcname=entry.name, recursive=True,
                           filter=lambda ti: None if any(p in _EXCLUDE_TOP_LEVEL for p in Path(ti.name).parts) else ti)
        # Cron capture is additive and never fails the snapshot; the manifest records whether it happened so rollback can say "no cron data".
        _write_manifest(dest, reason, archive, _count_skill_files(skills), _backup_cron_jobs_into(dest))
    except (OSError, tarfile.TarError) as e:
        logger.debug("Curator snapshot failed: %s", e, exc_info=True)
        shutil.rmtree(dest, ignore_errors=True)  # clean up partial snapshot
        return None

    _prune_old(keep=get_keep(), protect=protect_ids)
    logger.info("Curator snapshot created: %s (%s)", snap_id, reason)
    return dest


def _prune_old(keep: int, protect: Optional[Set[str]] = None) -> List[str]:
    """Delete regular snapshots beyond the newest *keep*; returns deleted ids. Ids in *protect* are never deleted —
    rollback() uses this so the mandatory pre-rollback safety snapshot cannot evict the snapshot being restored.
    Stale ``.rollback-staging-*`` dirs (crashed rollback) are cleaned up on every call."""
    protect = protect or set()
    backups = _backups_dir()
    if not backups.exists():
        return []
    dirs = [c for c in backups.iterdir() if c.is_dir()]
    # Newest first (lexicographic works because the id is UTC ISO).
    entries = sorted((c for c in dirs if _ID_RE.match(c.name)), key=lambda c: c.name, reverse=True)
    doomed = [(p, "prune") for p in entries[keep:] if p.name not in protect]
    doomed += [(p, "clean stale staging dir") for p in dirs if p.name.startswith(_STAGING_PREFIX)]
    deleted: List[str] = []
    for path, what in doomed:
        try:
            shutil.rmtree(path)
            if what == "prune":
                deleted.append(path.name)
        except OSError as e:
            logger.debug("Failed to %s %s: %s", what, path, e)
    return deleted


# --- List + rollback ---
def _read_manifest(snap_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _is_restorable(child: Path) -> bool:
    """A real snapshot dir with a tarball (excludes ``.rollback-staging-*``)."""
    return bool(child.is_dir() and _ID_RE.match(child.name) and (child / _ARCHIVE_NAME).exists())


def _restorable_snapshots() -> List[Path]:
    """Restorable snapshot dirs, newest first."""
    backups = _backups_dir()
    return [c for c in sorted(backups.iterdir(), reverse=True) if _is_restorable(c)] if backups.exists() else []


def list_backups() -> List[Dict[str, Any]]:
    """All restorable snapshots (manifest dicts), newest first."""
    out: List[Dict[str, Any]] = []
    for child in _restorable_snapshots():
        mf = {"id": child.name, "path": str(child), **_read_manifest(child)}
        try:
            mf.setdefault("archive_bytes", (child / _ARCHIVE_NAME).stat().st_size)
        except OSError:
            mf.setdefault("archive_bytes", 0)
        out.append(mf)
    return out


def _resolve_backup(backup_id: Optional[str]) -> Optional[Path]:
    """Path of the requested backup (newest if *backup_id* is None); None if no match."""
    if backup_id:
        target = _backups_dir() / backup_id
        return target if _ID_RE.match(backup_id) and _is_restorable(target) else None
    return next(iter(_restorable_snapshots()), None)


def _restore_cron_skill_links(snapshot_dir: Path) -> Dict[str, Any]:
    """Reconcile backed-up cron skill links into the live ``cron/jobs.json``. Only ``skills``/``skill`` are restored,
    and only on jobs that still exist live (by ``id``) — everything else is live state. Backup-only jobs are skipped
    and reported; live-only jobs untouched. Never raises; writes through ``cron.jobs`` under the scheduler's lock so
    we don't race tick()."""
    report: Dict[str, Any] = {"attempted": False, "restored": [], "skipped_missing": [], "unchanged": 0, "error": None}
    backup_file = snapshot_dir / CRON_JOBS_FILENAME
    if not backup_file.exists():
        return {**report, "error": f"snapshot has no {CRON_JOBS_FILENAME}"}
    try:
        backup_jobs = _jobs_list(json.loads(backup_file.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as e:
        return {**report, "error": f"failed to load backed-up jobs: {e}"}
    if backup_jobs is None:
        return {**report, "error": "backed-up cron-jobs.json has no jobs list"}

    # Backed-up skill state keyed by job id (legacy single + modern list field).
    backup_by_id: Dict[str, Dict[str, Any]] = {
        job["id"]: {"skills": job.get("skills"), "skill": job.get("skill"), "name": job.get("name") or job["id"]}
        for job in backup_jobs if isinstance(job, dict) and isinstance(job.get("id"), str) and job.get("id")
    }
    if not backup_by_id:
        return {**report, "attempted": True}  # we tried but there was nothing to do
    try:
        from cron.jobs import load_jobs, save_jobs, _jobs_lock
    except ImportError as e:
        return {**report, "error": f"cron module unavailable: {e}"}

    report["attempted"] = True
    try:
        with _jobs_lock():
            live_jobs = load_jobs()
            changed, live_ids = False, set()
            for live in live_jobs:
                jid = live.get("id") if isinstance(live, dict) else None
                if not isinstance(jid, str) or not jid:
                    continue
                live_ids.add(jid)
                backup = backup_by_id.get(jid)
                if backup is None:  # live job didn't exist at snapshot time
                    continue
                cur = {"skills": live.get("skills"), "skill": live.get("skill")}
                bkp = {"skills": backup.get("skills"), "skill": backup.get("skill")}
                if cur == bkp:
                    report["unchanged"] += 1
                    continue
                for key, value in bkp.items():  # Restore, preserving absence (don't add a key the backup lacked).
                    if value is None:
                        live.pop(key, None)
                    else:
                        live[key] = value
                report["restored"].append({"job_id": jid, "job_name": backup.get("name") or jid, "from": cur, "to": bkp})
                changed = True

            # Jobs in backup but not live = user deleted them after the snapshot.
            report["skipped_missing"] = [{"job_id": jid, "job_name": b.get("name") or jid}
                                         for jid, b in backup_by_id.items() if jid not in live_ids]
            if changed:
                save_jobs(live_jobs)
    except Exception as e:  # noqa: BLE001 — rollback must not die mid-restore
        logger.debug("Cron skill-link restore failed: %s", e, exc_info=True)
        report["error"] = f"restore failed mid-flight: {e}"
    return report


def _remove_entry(entry: Path) -> None:
    if entry.is_dir() and not entry.is_symlink():
        shutil.rmtree(entry)
    elif entry.exists() or entry.is_symlink():
        entry.unlink()


def _restore_excluded_subtrees(staged: Path, skills: Path) -> None:
    """Move excluded entries (nested ``.git``/``.hub``/...) from *staged* back under *skills* after a successful extract.
    Snapshots never contain these, so the staged copy of the live tree is the only source. ``.git`` may be a dir or a file
    (submodule / worktree ``gitdir:`` pointer) — both are moved. Best-effort and conditional: an entry is carried only when
    its parent skill dir was restored and nothing sits at the target. If the target snapshot predates the skill, the entry
    is dropped with the staging dir rather than left orphaned; the safety snapshot excludes these paths too, so not undoable."""
    for dirpath, dirnames, filenames in os.walk(staged):
        for src in [Path(dirpath) / n for n in (*dirnames, *filenames) if n in _EXCLUDE_TOP_LEVEL]:
            dest = skills / src.relative_to(staged)
            if dest.parent.is_dir() and not dest.exists():
                try:
                    shutil.move(str(src), str(dest))
                except OSError as e:
                    logger.debug("Could not restore excluded entry %s: %s", src, e)
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_TOP_LEVEL]


def _unstage(moved: List[Tuple[Path, Path]]) -> List[str]:
    """Move staged entries back to their original paths; returns names that could not be restored. ``shutil.move``
    moves *into* an existing destination dir, so partial-extract debris would bury the real skill
    (``skills/foo/foo/``) — clear each original path first. The staged copy is authoritative."""
    failed: List[str] = []
    for orig, dest in moved:
        try:
            _remove_entry(orig)
            shutil.move(str(dest), str(orig))
        except OSError:
            failed.append(orig.name)
    return failed


def _cron_summary(cron_report: Dict[str, Any]) -> Optional[str]:
    if not cron_report.get("attempted"):
        return None
    if cron_report.get("error"):
        return f"cron links: error — {cron_report['error']}"
    # (attempted with nothing matched — empty snapshot or no overlapping ids — says nothing)
    parts = [f"{n} {label}" for n, label in (
        (len(cron_report.get("restored") or []), "job(s) had skill links restored"),
        (len(cron_report.get("skipped_missing") or []), "backed-up job(s) no longer exist (skipped)"),
        (cron_report.get("unchanged", 0), "already matched"),
    ) if n]
    return "cron links: " + ", ".join(parts) if parts else None


def rollback(backup_id: Optional[str] = None) -> Tuple[bool, str, Optional[Path]]:
    """Restore ``~/.hermes/skills/`` from a snapshot (explicit id or newest): safety-snapshot the CURRENT tree; stage
    current top-level entries; extract; on failure move staged entries back. Returns ``(ok, message, snapshot_path)``."""
    target = _resolve_backup(backup_id)
    if target is None:
        return (False, "no matching backup found" + (f" for id '{backup_id}'" if backup_id else "")
                + " (use `hermes curator rollback --list` to see available snapshots)", None)
    archive = target / _ARCHIVE_NAME
    if not archive.exists():
        return (False, f"snapshot {target.name} has no skills.tar.gz — corrupted?", None)

    skills, backups = _skills_dir(), _backups_dir()
    backups.mkdir(parents=True, exist_ok=True)  # parents=True also creates skills/

    # Safety snapshot FIRST; bail if it fails, else a failed extract could leave the user with no skills. Protect the target from its prune.
    try:
        safety_snapshot = snapshot_skills(reason=f"pre-rollback to {target.name}", protect_ids={target.name})
    except Exception as e:
        return (False, f"pre-rollback safety snapshot failed: {e}", None)
    if safety_snapshot is None:
        return (False, "pre-rollback safety snapshot failed; backups may be disabled "
                "or unavailable, and current skills were not changed", None)

    # Stage current entries so the extract lands in an empty tree; the safety snapshot above (not staging) is the user-facing undo handle.
    staged = backups / f"{_STAGING_PREFIX}{_utc_id()}"
    try:
        staged.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        return (False, f"failed to create staging dir: {e}", None)

    moved: List[Tuple[Path, Path]] = []
    try:
        for entry in list(skills.iterdir()):
            if entry.name not in _EXCLUDE_TOP_LEVEL:
                shutil.move(str(entry), str(staged / entry.name))
                moved.append((entry, staged / entry.name))
    except OSError as e:
        _unstage(moved)
        shutil.rmtree(staged, ignore_errors=True)
        return (False, f"failed to stage current skills: {e}", None)

    try:
        with tarfile.open(archive, "r:gz") as tf:
            # Reject absolute paths and ".." defensively; Python 3.12+ also gets filter='data', older interpreters fall back unfiltered.
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise tarfile.TarError(f"refusing to extract unsafe path: {member.name!r}")
            try:
                tf.extractall(str(skills), filter="data")  # type: ignore[call-arg]
            except TypeError:
                tf.extractall(str(skills))  # Python < 3.12 — no filter kwarg
    except (OSError, tarfile.TarError) as e:
        # A partial extract can leave entries the original tree never had; drop those first or the "restored" tree is skills + a slice of snapshot.
        keep = _EXCLUDE_TOP_LEVEL | {orig.name for orig, _ in moved}
        for entry in [e for e in skills.iterdir() if e.name not in keep]:
            with contextlib.suppress(OSError):
                _remove_entry(entry)
        unrestored = _unstage(moved)
        if unrestored:  # Don't claim a clean restore; keep the staging dir for hand recovery.
            return (False, f"snapshot extract failed: {e} - could not restore "
                    f"{', '.join(sorted(unrestored))}; staged copies kept at {staged}", None)
        shutil.rmtree(staged, ignore_errors=True)
        return (False, f"snapshot extract failed (state restored): {e}", None)

    # Snapshots never contain excluded subtrees (nested ``.git``, ``.hub``, ...), so carry them over from the staged live tree
    # (top-level ``.git`` is never staged). Then staging is done; the undo handle is the safety snapshot.
    _restore_excluded_subtrees(staged, skills)
    shutil.rmtree(staged, ignore_errors=True)

    # Cron reconciliation failures don't fail the rollback — the skills tree (the main guarantee) is already restored.
    cron_report = _restore_cron_skill_links(target)
    logger.info("Curator rollback: restored from %s (cron_report=%s)", target.name, cron_report)
    return (True, "; ".join(filter(None, [f"restored from snapshot {target.name}", _cron_summary(cron_report)])), target)


# --- Human-readable summary for CLI ---
def summarize_backups() -> str:
    rows = list_backups()
    if not rows:
        return "No curator snapshots yet."
    header = f"{'id':<24}  {'reason':<40}  {'skills':>6}  {'size':>8}"
    return "\n".join([header, "─" * len(header)] + [
        f"{r.get('id','?'):<24}  {(r.get('reason','?') or '?')[:40]:<40}  "
        f"{r.get('skill_files', 0):>6}  {format_bytes(int(r.get('archive_bytes', 0))):>8}" for r in rows])
