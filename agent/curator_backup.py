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

Alongside the skills tarball, each snapshot also captures a copy of
``~/.hermes/cron/jobs.json`` as ``cron-jobs.json`` when it exists. Cron
jobs reference skills by name in their ``skills``/``skill`` fields; the
curator's consolidation pass rewrites those in place via
``cron.jobs.rewrite_skill_refs()``. Without capturing the pre-run state,
rolling back the skills tree would leave cron jobs pointing at the
umbrella skills even though the narrow skills they were originally
configured with have been restored. We store the whole jobs.json for
fidelity but rollback only touches the ``skills``/``skill`` fields — the
rest (schedule, next_run_at, enabled, prompt, etc.) is live state and
we leave it alone.
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


def _cron_jobs_file() -> Path:
    """Source path for the live cron jobs store (``~/.hermes/cron/jobs.json``)."""
    return get_hermes_home() / "cron" / "jobs.json"


CRON_JOBS_FILENAME = "cron-jobs.json"


def _backup_cron_jobs_into(dest: Path) -> Dict[str, Any]:
    """Copy the live cron jobs.json into ``dest`` as ``cron-jobs.json``.

    Returns a small dict describing what was captured so the caller can
    fold it into the manifest. Never raises — if the cron file is missing
    or unreadable, the return dict has ``backed_up=False`` and the reason,
    and the snapshot proceeds without cron data (the snapshot is still
    useful for rolling back skills).
    """
    src = _cron_jobs_file()
    info: Dict[str, Any] = {"backed_up": False, "jobs_count": 0}
    if not src.exists():
        info["reason"] = "no cron/jobs.json present"
        return info
    try:
        raw = src.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug("Failed to read cron/jobs.json for backup: %s", e)
        info["reason"] = f"read error: {e}"
        return info
    # Count jobs as a nice diagnostic — but don't fail the snapshot if the
    # file is unparseable; just store the raw text and let rollback deal
    # with it (or not, if it's corrupted). jobs.json wraps the list as
    # `{"jobs": [...], "updated_at": ...}` — we count via that shape, and
    # fall back to bare-list shape just in case the format ever changes.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            inner = parsed.get("jobs")
            if isinstance(inner, list):
                info["jobs_count"] = len(inner)
        elif isinstance(parsed, list):
            info["jobs_count"] = len(parsed)
    except (json.JSONDecodeError, TypeError):
        info["jobs_count"] = 0
        info["parse_warning"] = "jobs.json was not valid JSON at snapshot time"
    try:
        (dest / CRON_JOBS_FILENAME).write_text(raw, encoding="utf-8")
    except OSError as e:
        logger.debug("Failed to write cron backup file: %s", e)
        info["reason"] = f"write error: {e}"
        return info
    info["backed_up"] = True
    return info


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
                    skills_counted: int,
                    cron_info: Optional[Dict[str, Any]] = None) -> None:
    manifest = {
        "id": dest.name,
        "reason": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "skill_files": skills_counted,
    }
    if cron_info is not None:
        manifest["cron_jobs"] = {
            "backed_up": bool(cron_info.get("backed_up", False)),
            "jobs_count": int(cron_info.get("jobs_count", 0)),
        }
        if not cron_info.get("backed_up"):
            manifest["cron_jobs"]["reason"] = cron_info.get("reason", "not captured")
        if cron_info.get("parse_warning"):
            manifest["cron_jobs"]["parse_warning"] = cron_info["parse_warning"]
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
        # Capture cron/jobs.json alongside the tarball. Never fails the
        # snapshot — the skills side is the core guarantee; cron is
        # additive. We still record in the manifest whether it was
        # captured so rollback can surface "no cron data in this snapshot".
        cron_info = _backup_cron_jobs_into(dest)
        _write_manifest(dest, reason, archive,
                        _count_skill_files(skills),
                        cron_info=cron_info)
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


def _restore_cron_skill_links(snapshot_dir: Path) -> Dict[str, Any]:
    """Reconcile backed-up cron skill links into the live ``cron/jobs.json``.

    We do NOT overwrite the whole cron file. Only the ``skills`` and
    ``skill`` fields are restored, and only on jobs that still exist in the
    current file (matched by ``id``). Everything else about the job —
    schedule, next_run_at, last_run_at, enabled, prompt, workdir, hooks —
    is live state that the user/scheduler has modified since the snapshot;
    overwriting it would regress unrelated cron activity.

    Rules:
    - Jobs present in backup AND live, with differing skills → skills restored.
    - Jobs present in backup AND live, with matching skills → no-op.
    - Jobs present in backup but gone from live (user deleted the job
      after the snapshot) → skipped, noted in the return report.
    - Jobs present in live but not in backup (user created a new cron
      job after the snapshot) → left untouched.

    Never raises; failures are captured in the return dict. Writes through
    ``cron.jobs`` to pick up the same lock + atomic-write path that tick()
    uses, so we don't race the scheduler.
    """
    report: Dict[str, Any] = {
        "attempted": False,
        "restored": [],
        "skipped_missing": [],
        "unchanged": 0,
        "error": None,
    }
    backup_file = snapshot_dir / CRON_JOBS_FILENAME
    if not backup_file.exists():
        report["error"] = f"snapshot has no {CRON_JOBS_FILENAME}"
        return report

    try:
        backup_text = backup_file.read_text(encoding="utf-8")
        backup_parsed = json.loads(backup_text)
    except (OSError, json.JSONDecodeError) as e:
        report["error"] = f"failed to load backed-up jobs: {e}"
        return report
    # jobs.json on disk is `{"jobs": [...], "updated_at": ...}`; accept both
    # that shape and a bare list for forward compat.
    if isinstance(backup_parsed, dict):
        backup_jobs = backup_parsed.get("jobs")
    elif isinstance(backup_parsed, list):
        backup_jobs = backup_parsed
    else:
        backup_jobs = None
    if not isinstance(backup_jobs, list):
        report["error"] = "backed-up cron-jobs.json has no jobs list"
        return report

    # Build a lookup of the backed-up skill state keyed by job id.
    # We only need the two skill-ish fields (legacy single and modern list).
    backup_by_id: Dict[str, Dict[str, Any]] = {}
    for job in backup_jobs:
        if not isinstance(job, dict):
            continue
        jid = job.get("id")
        if not isinstance(jid, str) or not jid:
            continue
        backup_by_id[jid] = {
            "skills": job.get("skills"),
            "skill": job.get("skill"),
            "name": job.get("name") or jid,
        }

    if not backup_by_id:
        report["attempted"] = True  # we tried but there was nothing to do
        return report

    # Load and rewrite the live jobs under the scheduler's lock.
    try:
        from cron.jobs import load_jobs, save_jobs, _jobs_file_lock
    except ImportError as e:
        report["error"] = f"cron module unavailable: {e}"
        return report

    report["attempted"] = True
    try:
        with _jobs_file_lock:
            live_jobs = load_jobs()
            changed = False

            live_ids = set()
            for live in live_jobs:
                if not isinstance(live, dict):
                    continue
                jid = live.get("id")
                if not isinstance(jid, str) or not jid:
                    continue
                live_ids.add(jid)

                backup = backup_by_id.get(jid)
                if backup is None:
                    continue  # live job didn't exist at snapshot time

                cur_skills = live.get("skills")
                cur_skill = live.get("skill")
                bkp_skills = backup.get("skills")
                bkp_skill = backup.get("skill")

                if cur_skills == bkp_skills and cur_skill == bkp_skill:
                    report["unchanged"] += 1
                    continue

                # Restore. Preserve absence (don't force the key to appear
                # if the backup didn't have it either).
                if bkp_skills is None:
                    live.pop("skills", None)
                else:
                    live["skills"] = bkp_skills
                if bkp_skill is None:
                    live.pop("skill", None)
                else:
                    live["skill"] = bkp_skill

                report["restored"].append({
                    "job_id": jid,
                    "job_name": backup.get("name") or jid,
                    "from": {"skills": cur_skills, "skill": cur_skill},
                    "to": {"skills": bkp_skills, "skill": bkp_skill},
                })
                changed = True

            # Jobs in backup but not in live = user deleted them after snapshot
            for jid, backup in backup_by_id.items():
                if jid not in live_ids:
                    report["skipped_missing"].append({
                        "job_id": jid,
                        "job_name": backup.get("name") or jid,
                    })

            if changed:
                save_jobs(live_jobs)
    except Exception as e:  # noqa: BLE001 — rollback must not die mid-restore
        logger.debug("Cron skill-link restore failed: %s", e, exc_info=True)
        report["error"] = f"restore failed mid-flight: {e}"

    return report



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

    # Reconcile cron skill-links. Surgical: only the skills/skill fields
    # on jobs matched by id. Everything else in jobs.json is live state
    # (schedule, next_run_at, enabled, prompt, etc.) and we leave it
    # alone. Failures here don't fail the overall rollback — the skills
    # tree is already restored, which is the main guarantee.
    cron_report = _restore_cron_skill_links(target)

    summary_bits = [f"restored from snapshot {target.name}"]
    if cron_report.get("attempted"):
        restored_n = len(cron_report.get("restored") or [])
        skipped_n = len(cron_report.get("skipped_missing") or [])
        if cron_report.get("error"):
            summary_bits.append(f"cron links: error — {cron_report['error']}")
        elif restored_n == 0 and skipped_n == 0 and cron_report.get("unchanged", 0) == 0:
            # Attempted but nothing matched — empty snapshot or no overlapping ids.
            pass
        else:
            parts = []
            if restored_n:
                parts.append(f"{restored_n} job(s) had skill links restored")
            if skipped_n:
                parts.append(f"{skipped_n} backed-up job(s) no longer exist (skipped)")
            if cron_report.get("unchanged"):
                parts.append(f"{cron_report['unchanged']} already matched")
            summary_bits.append("cron links: " + ", ".join(parts))

    logger.info("Curator rollback: restored from %s (cron_report=%s)",
                target.name, cron_report)
    return (True, "; ".join(summary_bits), target)


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
