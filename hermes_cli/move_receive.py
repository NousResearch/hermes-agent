"""Receive a move bundle into a RUNNING install: resumable upload, preview, merge, undo.

The flow a Desktop "Move to Hermes Cloud" run drives over REST (``hermes_cli.web_routers.move``):

1. ``begin``   — declare the bundle's size and sha256; get an upload id (resumable by digest).
2. ``chunk``   — append base64 chunks at explicit offsets; a repeated chunk is a no-op.
3. ``finish``  — verify the digest and the manifest, store the preview.
4. ``apply``   — snapshot what will change, then merge piece by piece, journaling each step.
5. ``undo``    — restore the snapshot of one applied move.

Nothing here needs the gateway stopped. Each piece goes through the store a live gateway already
shares: sessions through the revision-fenced ``import`` mutation (new ids only, never an
overwrite), cron through the jobs lock (imported jobs arrive paused), memory through the memory
file lock, config through a deep merge where the target's own values win, skills as new
directories (an existing same-named skill is kept and the incoming one is reported).

``apply(..., replace={...})`` names the parts where the bundle wins instead: memory and persona are
overwritten, same-named skills and jobs are swapped for the incoming ones, the target's other jobs
are removed, its visible conversations are archived (never deleted), and config keys the bundle
carries win. What the hosting platform set up stays: credentials are never in a bundle, and the
config sections in :data:`_PLATFORM_CONFIG_KEYS` keep the target's values in every mode. Undo
restores all of it.

Every state change lands in ``<home>/moves/<id>/state.json`` so a crash mid-apply is visible and
resumable rather than silent: ``apply`` skips steps already recorded as done.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.move_bundle import MANIFEST_NAME, read_manifest
from hermes_cli.move_sessions import visible_session_ids

# Hard ceilings. A profile bundle without media is a few MB; history pushes it to tens.
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_CHUNK_BYTES = 4 * 1024 * 1024
_MOVES_DIR = "moves"
_STATE = "state.json"
_BUNDLE = "bundle.zip"
_SNAPSHOT = "snapshot"
_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
STEPS = ("config", "soul", "memory", "skills", "cron", "sessions")
# Config the receiving install's platform owns (Hermes Cloud seeds the model, the Nous tool gateway
# and the Chronos scheduler): a move never changes these, because the keys that would make the
# source's choices work never travel with it.
_PLATFORM_CONFIG_KEYS = frozenset({
    "model", "providers", "fallback_providers", "fallback", "auxiliary", "nous", "auth", "cron", "web",
    "image_gen", "tts", "stt", "max_concurrent_sessions", "max_live_sessions",
})

_lock = threading.Lock()


class MoveError(ValueError):
    """A refusal the client shows as-is. ``status`` is the HTTP status the router answers."""

    def __init__(self, message: str, status: int = 400, code: str = "move_refused"):
        super().__init__(message)
        self.status = status
        self.code = code


# --------------------------------------------------------------------------- state


def _moves_root(home: Path) -> Path:
    return home / _MOVES_DIR


def _move_dir(home: Path, move_id: str) -> Path:
    if not _ID_RE.match(move_id or ""):
        raise MoveError("Unknown move.", 404, "not_found")
    path = _moves_root(home) / move_id
    if not path.is_dir():
        raise MoveError("Unknown move.", 404, "not_found")
    return path


def _read_state(move_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((move_dir / _STATE).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise MoveError("This move's record is unreadable.", 409, "state_unreadable") from None


def _write_state(move_dir: Path, state: dict[str, Any]) -> None:
    from utils import atomic_json_write
    state["updated_at"] = time.time()
    atomic_json_write(move_dir / _STATE, state, mode=0o600)


def _public(state: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "status", "bytes", "received", "sha256", "created_at", "updated_at", "preview",
            "steps", "result", "error", "applied_at", "undone_at", "replace")
    return {k: state.get(k) for k in keys if k in state}


def list_moves(home: Path) -> list[dict[str, Any]]:
    root = _moves_root(home)
    out = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if path.is_dir() and _ID_RE.match(path.name):
            try:
                out.append(_public(_read_state(path)))
            except MoveError:
                continue
    return sorted(out, key=lambda s: s.get("created_at") or 0, reverse=True)


def get_move(home: Path, move_id: str) -> dict[str, Any]:
    return _public(_read_state(_move_dir(home, move_id)))


# --------------------------------------------------------------------------- upload


def begin(home: Path, *, size: int, sha256: str) -> dict[str, Any]:
    """Start (or resume) an upload of a bundle identified by its digest."""
    if not isinstance(size, int) or size <= 0:
        raise MoveError("The bundle is empty.")
    if size > MAX_BUNDLE_BYTES:
        raise MoveError(f"The bundle is {size // (1024 * 1024)} MB; the limit is "
                        f"{MAX_BUNDLE_BYTES // (1024 * 1024)} MB. Leave chat history out and try again.",
                        413, "too_large")
    sha256 = (sha256 or "").lower()
    if not _SHA_RE.match(sha256):
        raise MoveError("The bundle digest is malformed.")
    with _lock:
        for existing in list_moves(home):
            if existing.get("sha256") == sha256 and existing.get("status") in {"uploading", "ready"}:
                return existing
        from hermes_constants import mkdir_under_hermes_home
        move_id = uuid.uuid4().hex
        move_dir = _moves_root(home) / move_id
        mkdir_under_hermes_home(move_dir)
        os.chmod(move_dir, 0o700)
        (move_dir / _BUNDLE).touch(mode=0o600)
        state = {"id": move_id, "status": "uploading", "bytes": size, "received": 0, "sha256": sha256,
                 "created_at": time.time()}
        _write_state(move_dir, state)
        return _public(state)


def chunk(home: Path, move_id: str, *, offset: int, data_b64: str) -> dict[str, Any]:
    """Write one chunk at ``offset``. A chunk that ends at or before ``received`` is a replay."""
    move_dir = _move_dir(home, move_id)
    try:
        data = base64.b64decode(data_b64 or "", validate=True)
    except ValueError:
        raise MoveError("A chunk was not valid base64.") from None
    if not data or len(data) > MAX_CHUNK_BYTES:
        raise MoveError("A chunk was empty or too large.")
    with _lock:
        state = _read_state(move_dir)
        if state["status"] != "uploading":
            raise MoveError("This upload is already complete.", 409, "not_uploading")
        received = int(state["received"])
        if not isinstance(offset, int) or offset < 0 or offset > received:
            raise MoveError(f"Chunk offset {offset} skips ahead of {received} received bytes.", 409,
                            "offset_gap")
        end = offset + len(data)
        if end > state["bytes"]:
            raise MoveError("The chunk runs past the declared bundle size.")
        if end > received:
            with (move_dir / _BUNDLE).open("r+b") as fh:
                fh.seek(offset)
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            state["received"] = end
            _write_state(move_dir, state)
        return _public(state)


def finish(home: Path, move_id: str) -> dict[str, Any]:
    """Verify the uploaded bytes and build the preview the user confirms."""
    move_dir = _move_dir(home, move_id)
    with _lock:
        state = _read_state(move_dir)
        if state["status"] == "ready":
            return _public(state)
        if state["status"] != "uploading" or state["received"] != state["bytes"]:
            raise MoveError(f"Received {state['received']} of {state['bytes']} bytes.", 409, "incomplete")
        bundle = move_dir / _BUNDLE
        digest = hashlib.sha256()
        with bundle.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != state["sha256"]:
            state.update(status="uploading", received=0)
            bundle.write_bytes(b"")
            _write_state(move_dir, state)
            raise MoveError("The bundle arrived damaged. Sending it again.", 422, "digest_mismatch")
        try:
            with zipfile.ZipFile(bundle) as zf:
                if zf.testzip() is not None:
                    raise MoveError("The bundle is a damaged zip.", 422, "bad_zip")
                manifest = read_manifest(zf)
                state["preview"] = preview(home, zf, manifest)
        except zipfile.BadZipFile:
            raise MoveError("The upload is not a zip file.", 422, "bad_zip") from None
        except MoveError:
            raise
        except ValueError as exc:
            raise MoveError(str(exc), 422, "bad_manifest") from None
        state["status"] = "ready"
        _write_state(move_dir, state)
        return _public(state)


# --------------------------------------------------------------------------- preview


def _target_skill_names(home: Path) -> set[str]:
    root = home / "skills"
    return {p.parent.relative_to(root).as_posix() for p in root.rglob("SKILL.md")} if root.is_dir() else set()


def _target_cron_ids(home: Path) -> set[str]:
    try:
        data = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return set()
    jobs = data.get("jobs") if isinstance(data, dict) else data
    return {str(j.get("id")) for j in jobs if isinstance(j, dict)} if isinstance(jobs, list) else set()


def preview(home: Path, zf: zipfile.ZipFile, manifest: dict[str, Any]) -> dict[str, Any]:
    """What applying this bundle would do here, computed against the target's current state."""
    inv = manifest.get("inventory") or {}
    names = set(zf.namelist())
    existing_skills = _target_skill_names(home)
    incoming_skills = sorted({n.split("/SKILL.md")[0].removeprefix("skills/")
                              for n in names if n.startswith("skills/") and n.endswith("/SKILL.md")})
    existing_jobs = _target_cron_ids(home)
    jobs = [j for j in inv.get("cron_jobs") or [] if isinstance(j, dict)]
    soul_here = (home / "SOUL.md").read_text(encoding="utf-8-sig") if (home / "SOUL.md").is_file() else ""
    from hermes_cli.default_soul import DEFAULT_SOUL_MD, is_legacy_template_soul
    soul_customized = bool(soul_here.strip()) and soul_here.strip() != DEFAULT_SOUL_MD.strip() \
        and not is_legacy_template_soul(soul_here)
    return {
        "source": manifest.get("source") or {},
        "created_at": manifest.get("created_at"),
        "skills": {"add": [s for s in incoming_skills if s not in existing_skills],
                   "keep_existing": [s for s in incoming_skills if s in existing_skills],
                   "left_behind": inv.get("skills_skipped") or []},
        "memory": inv.get("memory_entries") or {},
        "soul": {"incoming": "SOUL.md" in names,
                 "action": "keep_existing" if soul_customized else ("replace" if "SOUL.md" in names else "none")},
        "cron": {"add": [j for j in jobs if str(j.get("id")) not in existing_jobs],
                 "keep_existing": [j for j in jobs if str(j.get("id")) in existing_jobs]},
        "sessions": sum(1 for n in names if n.startswith("sessions/") and n.endswith(".json")),
        "sessions_left_behind": inv.get("sessions_skipped") or 0,
        "config_keys": inv.get("config_keys") or [],
        "mcp_servers": inv.get("mcp_servers") or [],
        "keys_to_reenter": inv.get("keys_to_reenter") or [],
        # What a replace would take away here, so the confirmation can name it.
        "target": {"skills": len(existing_skills), "cron_jobs": len(existing_jobs),
                   "conversations": len(visible_session_ids(home)), "soul_customized": soul_customized},
    }


# --------------------------------------------------------------------------- snapshot / undo


_SNAPSHOT_PATHS = ("config.yaml", "SOUL.md", "memories/MEMORY.md", "memories/USER.md", "cron/jobs.json")


def _snapshot(home: Path, move_dir: Path) -> dict[str, Any]:
    """Copy every file apply can rewrite; record which ones did not exist yet."""
    snap = move_dir / _SNAPSHOT
    snap.mkdir(exist_ok=True)
    record: dict[str, Any] = {"files": {}, "added_skills": [], "replaced_skills": [], "added_sessions": [],
                              "archived_sessions": [], "added_jobs": []}
    for rel in _SNAPSHOT_PATHS:
        src = home / rel
        if src.is_file():
            dst = snap / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            record["files"][rel] = "existed"
        else:
            record["files"][rel] = "absent"
    return record


def undo(home: Path, move_id: str, *, sessions) -> dict[str, Any]:
    """Put back every file the move rewrote, every skill it swapped and every conversation it
    archived, and remove what it added. ``sessions`` writes through the store's owner
    (:mod:`hermes_cli.move_sessions`)."""
    move_dir = _move_dir(home, move_id)
    with _lock:
        state = _read_state(move_dir)
        if state.get("status") not in {"applied", "partial"}:
            raise MoveError("Only an applied move can be undone.", 409, "not_applied")
        record = state.get("undo") or {}
        problems: list[str] = []
        for rel, how in (record.get("files") or {}).items():
            try:
                _restore_file(home, move_dir, rel, how)
            except OSError as exc:
                problems.append(f"{rel}: {exc}")
        skills_root = (home / "skills").resolve()
        for rel in [*(record.get("added_skills") or []), *(record.get("replaced_skills") or [])]:
            path = home / "skills" / rel
            if path.is_dir() and path.resolve().is_relative_to(skills_root):
                shutil.rmtree(path, ignore_errors=True)
        for rel in record.get("replaced_skills") or []:
            kept = move_dir / _SNAPSHOT / "skills" / rel
            if kept.is_dir():
                (home / "skills" / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(kept), home / "skills" / rel)
        removed = _each(record.get("added_sessions") or [], sessions.delete, "remove", problems)
        restored = _each(record.get("archived_sessions") or [], lambda sid: sessions.set_archived(sid, False),
                         "unarchive", problems)
        state.update(status="undone", undone_at=time.time(),
                     result={**(state.get("result") or {}), "undo_problems": problems,
                             "sessions_removed": removed, "sessions_unarchived": restored})
        _write_state(move_dir, state)
        return _public(state)


def _restore_file(home: Path, move_dir: Path, rel: str, how: str) -> None:
    target = home / rel
    if rel == "cron/jobs.json":
        # The live ticker shares this file: restore under its lock, as a wholesale rewrite.
        from agent.runtime_scope import bind_profile_scope
        from cron import jobs as cron_jobs
        kept = json.loads((move_dir / _SNAPSHOT / rel).read_text(encoding="utf-8")) if how == "existed" else {}
        jobs = kept.get("jobs") if isinstance(kept, dict) else kept
        with bind_profile_scope(home, hydrate_secrets=False), cron_jobs.use_cron_store(home):
            cron_jobs.save_jobs(jobs if isinstance(jobs, list) else [], replace=True)
        return
    if how == "existed":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(move_dir / _SNAPSHOT / rel, target)
    elif target.exists():
        target.unlink()


def _each(ids: list[str], write: Callable[[str], None], verb: str, problems: list[str]) -> int:
    done = 0
    for sid in ids:
        try:
            write(sid)
            done += 1
        except Exception as exc:  # health: allow BLE001 -- one row must not stop the rest of an undo
            problems.append(f"could not {verb} conversation {sid}: {exc}")
    return done


# --------------------------------------------------------------------------- apply


def _deep_merge_missing(target: Any, incoming: Any) -> Any:
    """``incoming`` fills keys ``target`` does not have; the target's own values always win."""
    if isinstance(target, dict) and isinstance(incoming, dict):
        out = dict(target)
        for key, value in incoming.items():
            out[key] = _deep_merge_missing(target[key], value) if key in target else value
        return out
    return target


def _apply_config(home: Path, zf: zipfile.ZipFile, ctx: dict[str, Any]) -> dict[str, Any]:
    import hermes_yaml as yaml
    from hermes_cli.config import require_readable_config_before_write, save_config
    from agent.runtime_scope import bind_profile_scope
    try:
        incoming = yaml.safe_load(zf.read("config.yaml")) or {}  # config-reader: ok — the bundle member, not this home's config
    except KeyError:
        return {"keys_added": []}
    if not isinstance(incoming, dict):
        return {"keys_added": []}
    replace = "config" in ctx["replace"]
    with bind_profile_scope(home, hydrate_secrets=False):
        existing = require_readable_config_before_write(home / "config.yaml")
        # Replace: the bundle's settings win, section by section. Merge: the target's always win.
        merged = _deep_merge_missing(incoming, existing) if replace else _deep_merge_missing(existing, incoming)
        # What the hosting platform set up (model, Nous tool gateway, scheduler) stays as it is in
        # both modes: the credentials that would make the source's choices work never travel.
        for key in _PLATFORM_CONFIG_KEYS:
            if key in existing:
                merged[key] = existing[key]
            else:
                merged.pop(key, None)
        changed = sorted(k for k in incoming if k not in _PLATFORM_CONFIG_KEYS and merged.get(k) != existing.get(k))
        save_config(merged)
        return {"keys_added" if not replace else "keys_changed": changed}


def _apply_soul(home: Path, zf: zipfile.ZipFile, ctx: dict[str, Any]) -> dict[str, Any]:
    action = ctx["preview"]["soul"]["action"]
    if "soul" in ctx["replace"] and ctx["preview"]["soul"]["incoming"]:
        action = "replace"
    if action != "replace":
        return {"action": action}
    from utils import atomic_write_text
    atomic_write_text(home / "SOUL.md", zf.read("SOUL.md").decode("utf-8"), preserve_mode=True)
    return {"action": "replace"}


def _apply_memory(home: Path, zf: zipfile.ZipFile, ctx: dict[str, Any]) -> dict[str, Any]:
    """Merge: union of entries, target first, deduplicated. Replace: the incoming entries only.
    Over the char limit is allowed on load and reported, never truncated: the user decides what to
    prune."""
    from tools.memory_tool_store import ENTRY_DELIMITER, MemoryStore
    from utils import atomic_write_text
    replace = "memory" in ctx["replace"]
    added: dict[str, int] = {}
    for name in ("MEMORY.md", "USER.md"):
        try:
            incoming = MemoryStore._parse_entries(zf.read(f"memories/{name}").decode("utf-8"))
        except KeyError:
            incoming = None
        path = home / "memories" / name
        if incoming is None and not (replace and path.is_file()):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with MemoryStore._file_lock(path):
            raw, ok = MemoryStore._read_raw_checked(path)
            if not ok and not replace:
                raise MoveError(f"{name} on this agent is unreadable; nothing was merged into it.", 409)
            current = [] if replace else MemoryStore._parse_entries(raw)
            new = [e for e in incoming or [] if e not in current]
            if new or replace:
                atomic_write_text(path, ENTRY_DELIMITER.join(current + new), tmp_prefix=".mem_")
            added[name] = len(new)
    return {"entries_added" if not replace else "entries_now": added}


def _apply_skills(home: Path, zf: zipfile.ZipFile, ctx: dict[str, Any]) -> dict[str, Any]:
    """New skills land as new directories. Replace also swaps each same-named skill for the
    incoming copy (the target's is kept in the move's snapshot for undo); other skills stay."""
    root = (home / "skills").resolve()
    skills = ctx["preview"]["skills"]
    swapping = set(skills["keep_existing"]) if "skills" in ctx["replace"] else set()
    written, swapped = [], []
    for skill in sorted(set(skills["add"]) | swapping):
        dest = root / skill
        if not dest.resolve().is_relative_to(root):
            continue
        if dest.exists() and skill not in swapping:
            continue
        staging = _stage_skill(root, zf, skill)
        try:
            if dest.exists():
                kept = ctx["move_dir"] / _SNAPSHOT / "skills" / skill
                kept.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), kept)
                ctx["undo"]["replaced_skills"].append(skill)
                swapped.append(skill)
            else:
                ctx["undo"]["added_skills"].append(skill)
                written.append(skill)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, dest)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return {"added": written, "replaced": swapped}


def _stage_skill(root: Path, zf: zipfile.ZipFile, skill: str) -> Path:
    from hermes_cli.archive_safe import normalize_archive_parts
    staging = root / f".move-{uuid.uuid4().hex[:8]}"
    prefix = f"skills/{skill}/"
    for member in zf.namelist():
        if not member.startswith(prefix) or member.endswith("/"):
            continue
        out = staging.joinpath(*normalize_archive_parts(member[len(prefix):]))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(zf.read(member))
    return staging


def _paused(job: dict[str, Any], now: str) -> dict[str, Any]:
    meta = job.pop("_move", {}) or {}
    needs = meta.get("needs_this_machine") or []
    job.update(enabled=False, state="paused", paused_at=now, next_run_at=None,
               paused_reason=("Moved from another computer. Check its script and folder paths, "
                              "then resume it." if needs else
                              "Moved from another computer. Resume it when you are ready."),
               repeat={"times": (job.get("repeat") or {}).get("times"), "completed": 0}
               if isinstance(job.get("repeat"), dict) else job.get("repeat"))
    return job


def _apply_cron(home: Path, zf: zipfile.ZipFile, ctx: dict[str, Any]) -> dict[str, Any]:
    """Merge: incoming jobs whose id is new here are added. Replace: the job list becomes the
    incoming one. Moved jobs always arrive paused."""
    try:
        jobs = json.loads(zf.read("cron/jobs.json")).get("jobs") or []
    except (KeyError, ValueError, AttributeError):
        jobs = []
    from agent.runtime_scope import bind_profile_scope
    from cron import jobs as cron_jobs
    replace = "cron" in ctx["replace"]
    add_ids = {str(j.get("id")) for j in ctx["preview"]["cron"]["add"]}
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with bind_profile_scope(home, hydrate_secrets=False):
        with cron_jobs.use_cron_store(home), cron_jobs._jobs_lock():
            current = cron_jobs.load_jobs()
            incoming = [_paused(j, now) for j in jobs if isinstance(j, dict)
                        and (replace or str(j.get("id")) in add_ids)]
            if replace:
                removed = [str(j.get("id")) for j in current]
                cron_jobs._save_jobs_unlocked(incoming, removed_ids=removed)
            else:
                have = {str(j.get("id")) for j in current}
                incoming = [j for j in incoming if str(j.get("id")) not in have]
                removed = []
                cron_jobs._save_jobs_unlocked(current + incoming)
    added = [str(j["id"]) for j in incoming]
    ctx["undo"]["added_jobs"] = added
    return {"added": added, "removed": removed, "paused": True}


def _taken_titles(home: Path) -> set[str]:
    """Titles already used here: titles are UNIQUE in ``sessions``, and one clash aborts an import."""
    import sqlite3
    from hermes_cli.sqlite_safe_read import connect_tracked
    db_path = home / "state.db"
    if not db_path.is_file():
        return set()
    try:
        con = connect_tracked(f"file:{db_path}?mode=ro", tracking_path=db_path, uri=True)
        try:
            return {row[0] for row in con.execute("SELECT title FROM sessions WHERE title IS NOT NULL")}
        finally:
            con.close()
    except sqlite3.Error:
        return set()


def _free_title(title: Any, taken: set[str]) -> Any:
    if not isinstance(title, str) or title not in taken:
        return title
    # The store's own lineage numbering ("Taxes" -> "Taxes #2"), so a moved chat reads like a sibling.
    n = 2
    while f"{title} #{n}" in taken:
        n += 1
    return f"{title} #{n}"


def _apply_sessions(home: Path, zf: zipfile.ZipFile, ctx: dict[str, Any]) -> dict[str, Any]:
    """Merge: incoming conversations are added beside the target's. Replace: the target's visible
    conversations are archived first (kept, never deleted; undo brings them back)."""
    sessions = ctx["sessions"]
    archived = 0
    if "sessions" in ctx["replace"]:
        done = set(ctx["undo"]["archived_sessions"])
        for sid in visible_session_ids(home):
            if sid not in done:
                sessions.set_archived(sid, True)
                ctx["undo"]["archived_sessions"].append(sid)
                archived += 1
    imported = skipped = failed = 0
    errors: list[str] = []
    taken = _taken_titles(home)
    for member in sorted(n for n in zf.namelist() if n.startswith("sessions/") and n.endswith(".json")):
        try:
            session = json.loads(zf.read(member))
        except ValueError:
            failed += 1
            continue
        session["title"] = _free_title(session.get("title"), taken)
        if isinstance(session.get("title"), str):
            taken.add(session["title"])
        try:
            result = sessions.import_session(session)
        except Exception as exc:  # health: allow BLE001 -- one bad session must not abort the rest
            failed += 1
            if len(errors) < 5:
                errors.append(f"{session.get('title') or session.get('id')}: {exc}")
            continue
        imported += int(result.get("imported") or 0)
        skipped += int(result.get("skipped") or 0)
        ctx["undo"]["added_sessions"].extend(result.get("imported_ids") or [])
    return {"imported": imported, "already_here": skipped, "failed": failed, "errors": errors,
            "archived": archived}


_APPLIERS: dict[str, Callable[[Path, zipfile.ZipFile, dict[str, Any]], dict[str, Any]]] = {
    "config": _apply_config, "soul": _apply_soul, "memory": _apply_memory,
    "skills": _apply_skills, "cron": _apply_cron, "sessions": _apply_sessions,
}


def apply(home: Path, move_id: str, *, sessions, include: Optional[set[str]] = None,
          replace: Optional[set[str]] = None) -> dict[str, Any]:
    """Apply a verified bundle: each part in ``include`` (default: all), merged, or replaced when it
    is also in ``replace``. ``sessions`` writes conversations through the store's owner
    (:mod:`hermes_cli.move_sessions`). Resumable: steps already recorded as done are skipped, and
    the include/replace choice is fixed by the first apply so a retry cannot change its meaning."""
    unknown = (set(include or ()) | set(replace or ())) - set(STEPS)
    if unknown:
        raise MoveError(f"Unknown part(s): {', '.join(sorted(unknown))}.", 400, "bad_parts")
    move_dir = _move_dir(home, move_id)
    with _lock:
        state = _read_state(move_dir)
        if state.get("status") in {"applied", "undone"}:
            return _public(state)
        if state.get("status") not in {"ready", "applying", "partial"}:
            raise MoveError("Finish the upload before applying it.", 409, "not_ready")
        if "undo" not in state:
            state["undo"] = _snapshot(home, move_dir)
            state["include"] = sorted(include) if include is not None else list(STEPS)
            state["replace"] = sorted(set(replace or ()) & set(state["include"]))
        state["status"] = "applying"
        state.setdefault("steps", {})
        _write_state(move_dir, state)
    ctx = {"preview": state["preview"], "undo": state["undo"], "sessions": sessions, "move_dir": move_dir,
           "replace": set(state.get("replace") or ())}
    chosen = [s for s in STEPS if s in set(state.get("include") or STEPS)]
    with zipfile.ZipFile(move_dir / _BUNDLE) as zf:
        for step in chosen:
            if (state["steps"].get(step) or {}).get("status") == "done":
                continue
            try:
                detail = _APPLIERS[step](home, zf, ctx)
                state["steps"][step] = {"status": "done", **detail}
            except Exception as exc:  # health: allow BLE001 -- recorded per step, the rest still run
                state["steps"][step] = {"status": "failed", "error": str(exc)[:500]}
            with _lock:
                _write_state(move_dir, state)
    failed = [s for s in chosen if state["steps"].get(s, {}).get("status") == "failed"]
    state.update(status="partial" if failed else "applied", applied_at=time.time(),
                 result={"failed_steps": failed})
    with _lock:
        _write_state(move_dir, state)
    # A partial move keeps its bundle so a retry resumes the failed steps.
    if not failed:
        try:
            (move_dir / _BUNDLE).unlink()
        except OSError:
            pass
    return _public(state)


def discard(home: Path, move_id: str) -> dict[str, Any]:
    """Drop an upload that was never applied."""
    move_dir = _move_dir(home, move_id)
    state = _read_state(move_dir)
    if state.get("status") in {"applying", "applied", "partial"}:
        raise MoveError("An applied move is kept so it can be undone.", 409, "applied")
    shutil.rmtree(move_dir, ignore_errors=True)
    return {"id": move_id, "status": "discarded"}


__all__ = ["MANIFEST_NAME", "MAX_BUNDLE_BYTES", "MAX_CHUNK_BYTES", "STEPS", "MoveError", "apply", "begin",
           "chunk", "discard", "finish", "get_move", "list_moves", "undo"]
