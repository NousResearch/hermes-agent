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
resumable rather than silent: ``apply`` skips steps already recorded as done, and each destructive
change is journaled there BEFORE it happens, so a resumed step knows what it already took away.
One apply or undo runs at a time per move, across processes (``<move>/.lock``).
"""

from __future__ import annotations

import base64
import contextlib
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
# What a bundle may expand to: a small zip can declare gigabytes, and finish() reads every member.
MAX_MEMBERS = 200_000
MAX_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
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


@contextlib.contextmanager
def _held(move_dir: Path):
    """The move's exclusive lock for a whole apply or undo, shared by the CLI and ``hermes serve``
    (a process-local lock cannot see the other). The OS drops it when its holder dies, so an
    ``applying`` state with no holder is an interrupted apply, free to resume."""
    from pm.filesystem import lock_fd
    fd = os.open(move_dir / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not lock_fd(fd, wait=False):
            raise MoveError("This move is already being applied.", 409, "busy")
        yield
    finally:
        os.close(fd)


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
            # A partial or interrupted apply of the same bundle resumes rather than starting over.
            if existing.get("sha256") == sha256 and existing.get("status") in {"uploading", "ready", "applying",
                                                                               "partial"}:
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
        if state["status"] in {"ready", "applying", "partial"}:
            return _public(state)
        if state["status"] != "uploading" or state["received"] != state["bytes"]:
            raise MoveError(f"Received {state['received']} of {state['bytes']} bytes.", 409, "incomplete")
    # Hashing and reading every member take a while; the shared lock also serves other uploads.
    bundle = move_dir / _BUNDLE
    digest = hashlib.sha256()
    with bundle.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != state["sha256"]:
        with _lock:
            state.update(status="uploading", received=0)
            bundle.write_bytes(b"")
            _write_state(move_dir, state)
        raise MoveError("The bundle arrived damaged. Sending it again.", 422, "digest_mismatch")
    try:
        with zipfile.ZipFile(bundle) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_EXPANDED_BYTES \
                    or any(i.file_size > MAX_BUNDLE_BYTES for i in infos):
                raise MoveError("The bundle unpacks to far more than a move bundle holds.", 413, "too_large")
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
    with _lock:
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


# The files each step can rewrite. Merged cron is undone by removing the added jobs, not by putting
# the file back: the ticker keeps rewriting it for the target's own running jobs.
_STEP_FILES = {"config": ("config.yaml",), "soul": ("SOUL.md",),
               "memory": ("memories/MEMORY.md", "memories/USER.md"), "cron": ("cron/jobs.json",)}


def _sha(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _step_files(step: str, replace: set[str]) -> tuple[str, ...]:
    return () if step == "cron" and "cron" not in replace else _STEP_FILES.get(step, ())


def _snapshot(home: Path, move_dir: Path, record: dict[str, Any], rels: tuple[str, ...]) -> None:
    """Copy the files a step is about to rewrite, or note that they do not exist yet. A resumed
    step keeps the first copy: by then the file may already hold the incoming version."""
    for rel in rels:
        if rel in record["files"]:
            continue
        src = home / rel
        if src.is_file():
            dst = move_dir / _SNAPSHOT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        record["files"][rel] = "existed" if src.is_file() else "absent"


def _record_written(home: Path, move_dir: Path, record: dict[str, Any], rels: tuple[str, ...]) -> None:
    """Remember what the move wrote, so undo can tell a file changed since from one it may put
    back. A file the step left as it was has nothing to undo."""
    for rel in rels:
        after = _sha(home / rel)
        if after == (_sha(move_dir / _SNAPSHOT / rel) if record["files"].get(rel) == "existed" else None):
            record["files"].pop(rel, None)
            record["written"].pop(rel, None)
        else:
            record["written"][rel] = after


def undo(home: Path, move_id: str, *, sessions) -> dict[str, Any]:
    """Put back every file the move rewrote, every skill it swapped and every conversation it
    archived, and remove what it added. ``sessions`` writes through the store's owner
    (:mod:`hermes_cli.move_sessions`). Each finished piece leaves the record, so an undo that hit
    problems (status ``undo_partial``) can be run again and only redoes the rest. A file changed
    since the move is left as it is and reported, never overwritten."""
    move_dir = _move_dir(home, move_id)
    with _held(move_dir):
        state = _read_state(move_dir)
        if state.get("status") not in {"applied", "partial", "undo_partial"}:
            raise MoveError("Only an applied move can be undone.", 409, "not_applied")
        record = state.get("undo") or {}
        problems: list[str] = []

        def save() -> None:
            _write_state(move_dir, state)

        _undo_files(home, move_dir, record, problems, save)
        _undo_skills(home, move_dir, record, problems, save)
        _undo_jobs(home, record, problems)
        removed = _each(record, "added_sessions", sessions.delete, "remove", problems)
        restored = _each(record, "archived_sessions", lambda sid: sessions.set_archived(sid, False),
                         "unarchive", problems)
        state.update(status="undo_partial" if problems else "undone",
                     result={**(state.get("result") or {}), "undo_problems": problems,
                             "sessions_removed": removed, "sessions_unarchived": restored})
        if not problems:
            state["undone_at"] = time.time()
        save()
        return _public(state)


def _undo_files(home: Path, move_dir: Path, record: dict[str, Any], problems: list[str], save) -> None:
    written = record.get("written") or {}
    for rel, how in list((record.get("files") or {}).items()):
        if _sha(home / rel) != written.get(rel):
            problems.append(f"changed since the move: {rel}")
            continue
        try:
            _restore_file(home, move_dir, rel, how)
        except (OSError, ValueError) as exc:
            problems.append(f"{rel}: {exc}")
            continue
        del record["files"][rel]
        save()


def _undo_skills(home: Path, move_dir: Path, record: dict[str, Any], problems: list[str], save) -> None:
    root = (home / "skills").resolve()
    for rel in list(record.get("added_skills") or []):
        path = home / "skills" / rel
        if path.resolve().is_relative_to(root):
            shutil.rmtree(path, ignore_errors=True)
        record["added_skills"].remove(rel)
        save()
    for rel in list(record.get("replaced_skills") or []):
        kept, path = move_dir / _SNAPSHOT / "skills" / rel, home / "skills" / rel
        if not kept.is_dir():
            problems.append(f"skills/{rel}: the copy kept for undo is missing")
            continue
        if path.resolve().is_relative_to(root):
            shutil.rmtree(path, ignore_errors=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(kept), path)
        record["replaced_skills"].remove(rel)
        save()


def _undo_jobs(home: Path, record: dict[str, Any], problems: list[str]) -> None:
    added = set(record.get("added_jobs") or [])
    if not added:
        return
    from agent.runtime_scope import bind_profile_scope
    from cron import jobs as cron_jobs
    try:
        with bind_profile_scope(home, hydrate_secrets=False), cron_jobs.use_cron_store(home), cron_jobs._jobs_lock():
            kept = [j for j in cron_jobs.load_jobs() if str(j.get("id")) not in added]
            cron_jobs._save_jobs_unlocked(kept, removed_ids=added)
    except OSError as exc:
        problems.append(f"cron/jobs.json: {exc}")
        return
    record["added_jobs"] = []


def _restore_file(home: Path, move_dir: Path, rel: str, how: str) -> None:
    target = home / rel
    if rel == "cron/jobs.json":
        # The live ticker shares this file: restore under its lock, as a wholesale rewrite.
        from agent.runtime_scope import bind_profile_scope
        from cron import jobs as cron_jobs
        kept = json.loads((move_dir / _SNAPSHOT / rel).read_text(encoding="utf-8-sig")) if how == "existed" else {}
        jobs = kept.get("jobs") if isinstance(kept, dict) else kept
        with bind_profile_scope(home, hydrate_secrets=False), cron_jobs.use_cron_store(home):
            cron_jobs.save_jobs(jobs if isinstance(jobs, list) else [], replace=True)
        return
    if how == "existed":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(move_dir / _SNAPSHOT / rel, target)
    elif target.exists():
        target.unlink()


def _each(record: dict[str, Any], key: str, write: Callable[[str], None], verb: str, problems: list[str]) -> int:
    """Write each id in ``record[key]``; the ones that failed stay there for the next undo."""
    ids, left = record.get(key) or [], []
    for sid in ids:
        try:
            write(sid)
        except Exception as exc:  # health: allow BLE001 -- one row must not stop the rest of an undo
            problems.append(f"could not {verb} conversation {sid}: {exc}")
            left.append(sid)
    record[key] = left
    return len(ids) - len(left)


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
    incoming copy (the target's is kept in the move's snapshot for undo); other skills stay.

    Journaled before each move on disk, so a resumed step tells its own work from the target's:
    a journaled skill with a kept copy was already swapped, one without is still the original."""
    root = (home / "skills").resolve()
    skills, undo_rec = ctx["preview"]["skills"], ctx["undo"]
    swapping = set(skills["keep_existing"]) if "skills" in ctx["replace"] else set()
    written, swapped = [], []
    for skill in sorted(set(skills["add"]) | swapping):
        dest, kept = root / skill, ctx["move_dir"] / _SNAPSHOT / "skills" / skill
        if not dest.resolve().is_relative_to(root):
            continue
        ours = skill in undo_rec["added_skills"] or skill in undo_rec["replaced_skills"]
        if skill in swapping and dest.exists() and not kept.exists():
            if skill not in undo_rec["replaced_skills"]:
                undo_rec["replaced_skills"].append(skill)
            ctx["journal"]()
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), kept)
        elif dest.exists() and not ours:
            continue
        elif not ours:
            undo_rec["added_skills"].append(skill)
            ctx["journal"]()
        if not dest.exists():
            staging = _stage_skill(root, zf, skill)
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, dest)
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
        (swapped if skill in undo_rec["replaced_skills"] else written).append(skill)
    return {"added": written, "replaced": swapped}


def _stage_skill(root: Path, zf: zipfile.ZipFile, skill: str) -> Path:
    from hermes_cli.archive_safe import normalize_archive_parts
    staging = root / f".move-{uuid.uuid4().hex[:8]}"
    prefix = f"skills/{skill}/"
    for info in zf.infolist():
        member = info.filename
        if not member.startswith(prefix) or member.endswith("/"):
            continue
        out = staging.joinpath(*normalize_archive_parts(member[len(prefix):]))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(zf.read(info))
        # A skill's scripts must still run; only the execute bit travels, never setuid or setgid.
        if (info.external_attr >> 16) & 0o111:
            out.chmod(0o755)
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
                # Undone by putting the snapshot of jobs.json back.
                removed = [str(j.get("id")) for j in current]
                cron_jobs._save_jobs_unlocked(incoming, removed_ids=removed)
            else:
                have = {str(j.get("id")) for j in current}
                incoming = [j for j in incoming if str(j.get("id")) not in have]
                removed = []
                # Journaled first: a resumed step finds these jobs already here and adds nothing.
                ctx["undo"]["added_jobs"] = sorted(set(ctx["undo"]["added_jobs"]) | {str(j["id"]) for j in incoming})
                ctx["journal"]()
                if incoming:  # nothing new leaves the scheduler's file untouched, timestamp included
                    cron_jobs._save_jobs_unlocked(current + incoming)
    added = [str(j["id"]) for j in incoming]
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


def _import_one(home: Path, ctx: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    """Import one conversation. The store fences a deleted id for good (a row restored beside the
    fence lists fine and refuses every turn), and undoing an earlier move deleted exactly these ids,
    so a fenced one arrives under an id of its own, fixed per move so a retry stays idempotent.

    The fence is read from the store itself, not from the refusal: a live owner answers only
    ``admission_conflict``, which other refusals share. The id is journaled before the import, and
    one already here that the move did not import is the target's own, never undone."""
    from hermes_cli.move_sessions import stored_status
    added = ctx["undo"]["added_sessions"]
    status = stored_status(home, session["id"])
    if status == "retired":
        session = {**session, "id": f"{session['id']}-m{ctx['move_dir'].name[:8]}"}
        status = stored_status(home, session["id"])
    if status == "present":
        ours = session["id"] in added
        return {"imported": int(ours), "skipped": int(not ours)}
    if session["id"] not in added:
        added.append(session["id"])
        ctx["journal"]()
    return ctx["sessions"].import_session(session)


def _apply_sessions(home: Path, zf: zipfile.ZipFile, ctx: dict[str, Any]) -> dict[str, Any]:
    """Merge: incoming conversations are added beside the target's. Replace: the target's visible
    conversations are archived first (kept, never deleted; undo brings them back)."""
    sessions = ctx["sessions"]
    archived = 0
    imported = skipped = failed = 0
    if "sessions" in ctx["replace"]:
        # A resumed step must not archive what it imported itself.
        done = set(ctx["undo"]["archived_sessions"]) | set(ctx["undo"]["added_sessions"])
        for sid in visible_session_ids(home):
            if sid not in done:
                ctx["undo"]["archived_sessions"].append(sid)
                ctx["journal"]()
                sessions.set_archived(sid, True)
                archived += 1
    errors: list[str] = []
    taken = _taken_titles(home)
    for member in sorted(n for n in zf.namelist() if n.startswith("sessions/") and n.endswith(".json")):
        try:
            session = json.loads(zf.read(member))
        except ValueError:
            failed += 1
            errors.append(f"{member}: not a readable conversation")
            continue
        session["title"] = _free_title(session.get("title"), taken)
        if isinstance(session.get("title"), str):
            taken.add(session["title"])
        try:
            result = _import_one(home, ctx, session)
        except Exception as exc:  # health: allow BLE001 -- one bad session must not abort the rest
            failed += 1
            errors.append(f"{session.get('title') or session.get('id')}: {exc}")
            continue
        imported += int(result.get("imported") or 0)
        skipped += int(result.get("skipped") or 0)
    return {"imported": imported, "already_here": skipped, "failed": failed, "errors": errors[:5],
            "archived": archived}


_APPLIERS: dict[str, Callable[[Path, zipfile.ZipFile, dict[str, Any]], dict[str, Any]]] = {
    "config": _apply_config, "soul": _apply_soul, "memory": _apply_memory,
    "skills": _apply_skills, "cron": _apply_cron, "sessions": _apply_sessions,
}


def _run_step(home: Path, zf: zipfile.ZipFile, ctx: dict[str, Any], step: str) -> dict[str, Any]:
    """One step: snapshot its files (journaled before anything changes), run it, record what it
    wrote. A step with items that failed is ``partial``; a retry re-runs it, skipping done items."""
    record, rels = ctx["undo"], _step_files(step, ctx["replace"])
    _snapshot(home, ctx["move_dir"], record, rels)
    ctx["journal"]()
    try:
        detail = _APPLIERS[step](home, zf, ctx)
        out = {"status": "partial" if detail.get("failed") else "done", **detail}
    except Exception as exc:  # health: allow BLE001 -- recorded per step, the rest still run
        out = {"status": "failed", "error": str(exc)[:500]}
    _record_written(home, ctx["move_dir"], record, rels)
    return out


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
    with _held(move_dir):
        state = _read_state(move_dir)
        if state.get("status") in {"applied", "undone", "undo_partial"}:
            return _public(state)
        if state.get("status") not in {"ready", "applying", "partial"}:
            raise MoveError("Finish the upload before applying it.", 409, "not_ready")
        if "undo" not in state:
            state["undo"] = {"files": {}, "written": {}, "added_skills": [], "replaced_skills": [],
                             "added_sessions": [], "archived_sessions": [], "added_jobs": []}
            state["include"] = sorted(include) if include is not None else list(STEPS)
            state["replace"] = sorted(set(replace or ()) & set(state["include"]))
            # Before any step: an earlier step can seed a later one's file (saving config writes a
            # default SOUL.md), and undo must put back the file as it was before the move.
            for step in state["include"]:
                _snapshot(home, move_dir, state["undo"], _step_files(step, set(state["replace"])))
        state["status"] = "applying"
        state.setdefault("steps", {})
        _write_state(move_dir, state)
        ctx = {"preview": state["preview"], "undo": state["undo"], "sessions": sessions, "move_dir": move_dir,
               "replace": set(state.get("replace") or ()), "journal": lambda: _write_state(move_dir, state)}
        chosen = [s for s in STEPS if s in set(state.get("include") or STEPS)]
        with zipfile.ZipFile(move_dir / _BUNDLE) as zf:
            for step in chosen:
                if (state["steps"].get(step) or {}).get("status") != "done":
                    state["steps"][step] = _run_step(home, zf, ctx, step)
                    _write_state(move_dir, state)
        unfinished = [s for s in chosen if state["steps"][s]["status"] != "done"]
        failed = [s for s in unfinished if state["steps"][s]["status"] == "failed"]
        state.update(status="partial" if unfinished else "applied", applied_at=time.time(),
                     result={"failed_steps": failed, "partial_steps": [s for s in unfinished if s not in failed]})
        _write_state(move_dir, state)
        # A partial move keeps its bundle so a retry resumes what did not finish.
        if not unfinished:
            with contextlib.suppress(OSError):
                (move_dir / _BUNDLE).unlink()
    return _public(state)


def discard(home: Path, move_id: str) -> dict[str, Any]:
    """Drop an upload that was never applied."""
    move_dir = _move_dir(home, move_id)
    state = _read_state(move_dir)
    if state.get("status") in {"applying", "applied", "partial", "undo_partial"}:
        raise MoveError("An applied move is kept so it can be undone.", 409, "applied")
    shutil.rmtree(move_dir, ignore_errors=True)
    return {"id": move_id, "status": "discarded"}


__all__ = ["MANIFEST_NAME", "MAX_BUNDLE_BYTES", "MAX_CHUNK_BYTES", "STEPS", "MoveError", "apply", "begin",
           "chunk", "discard", "finish", "get_move", "list_moves", "undo"]
