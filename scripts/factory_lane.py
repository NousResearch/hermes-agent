"""factory_lane.py — registry core cross-agent (HER-54).

Journal per-lane JSONL append-only en `registry/lanes/<KEY>.jsonl`, verrous
par répertoire atomique en `registry/locks/<KEY>/owner.json`, protégés par
`fcntl.flock` sur un fichier de verrou dédié. Python stdlib uniquement.
"""

import argparse
import calendar
import contextlib
import errno
import fcntl
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Constantes de contrat
# --------------------------------------------------------------------------

KNOWN_PREFIXES = frozenset({"HER", "SCA", "IMP", "TL", "JYI"})
_TICKET_KEY_RE = re.compile(
    r"^(" + "|".join(sorted(KNOWN_PREFIXES)) + r")-[1-9][0-9]*\Z"
)
_ADHOC_KEY_RE = re.compile(r"^adhoc-[a-z0-9]+(?:-[a-z0-9]+)*\Z")

ALLOWED_MANUAL_EVENTS = frozenset({
    "build_started", "checkpoint", "pr_opened", "ci_green", "review_done",
    "ready_to_merge", "go_jean", "merged", "deployed", "smoke_ok",
    "memory_written", "lane_abandoned",
})

STATUS_LADDER = {
    "lane_claimed": "claimed",
    "build_started": "in_progress",
    "pr_opened": "pr_open",
    "lane_closed": "closed",
}

ALLOWED_HANDOFF_STATUS = frozenset({
    "ready_to_merge", "blocked", "in_progress", "needs_review", "done",
})

MAX_EVIDENCE_LEN = 200
MAX_NEXT_STEP_LEN = 1000
WORKTREE_ACTIVE_SECONDS = 24 * 3600
DEFAULT_TTL_HOURS = 72.0

_SECRET_PATTERN = re.compile(
    r"(token|password|passwd|secret|api[_-]?key|credential)", re.IGNORECASE
)

# --------------------------------------------------------------------------
# Constantes context pack (HER-55 Scope A)
# --------------------------------------------------------------------------

CONTEXT_PACK_MAX_LINES = 300
UNMAPPED_BRIEF_MAX_LINES = 20
LINEAR_CACHE_STALE_HOURS = 24.0
UNMAPPED_REPO_MARKER = "UNMAPPED-REPO"
# Motifs de répertoire bloqués en défense en profondeur, même si allow-listés
# par erreur dans context-map.json. `_SECRET_PATTERN` couvre déjà
# secret/token/credential/password (insensible à la casse).
_VAULT_BLOCKED_DIR_PREFIXES = ("transcripts/", "Legal/", "Clients/")


class RegistryError(Exception):
    """Erreur de validation ou d'état attendue — provoque un exit non nul."""


# --------------------------------------------------------------------------
# Validation clé / chemins
# --------------------------------------------------------------------------

def validate_key(key):
    if not isinstance(key, str) or not key:
        raise RegistryError("key must be a non-empty string")
    if _TICKET_KEY_RE.match(key) or _ADHOC_KEY_RE.match(key):
        return
    raise RegistryError(f"invalid lane key: {key!r}")


_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)

# Alias système connus (macOS monte `/var`, `/tmp`, `/etc` comme symlinks vers
# `/private/...`). Ce ne sont pas des symlinks créés par un attaquant dans un
# chemin utilisateur : ils sont fixes, racine du système, et la résolution
# réelle est vérifiée avant de les tolérer.
_KNOWN_SYSTEM_ALIAS_ROOTS = frozenset({"/var", "/tmp", "/etc"})


def _is_known_system_alias(ancestor):
    """True seulement pour un alias macOS top-level résolvant vers /private/<nom>."""
    ancestor_str = str(ancestor)
    if ancestor_str not in _KNOWN_SYSTEM_ALIAS_ROOTS:
        return False
    try:
        resolved = os.path.realpath(ancestor_str)
    except OSError:
        return False
    return resolved == f"/private{ancestor_str}"


def _reject_symlink_ancestors(path):
    for ancestor in Path(path).parents:
        if ancestor.exists() and ancestor.is_symlink():
            if _is_known_system_alias(ancestor):
                continue
            raise RegistryError(f"path ancestor must not be a symlink: {ancestor}")


def _open_secure(path, flags, mode=0o600):
    """os.open with O_NOFOLLOW when available; refuses to traverse a symlink."""
    try:
        return os.open(str(path), flags | _NOFOLLOW_FLAG, mode)
    except OSError as exc:
        if _NOFOLLOW_FLAG and exc.errno in (errno.ELOOP, errno.EMLINK):
            raise RegistryError(f"refusing to follow symlink: {path}") from exc
        raise


def _reject_symlink(path, label):
    if Path(path).is_symlink():
        raise RegistryError(f"{label} must not be a symlink: {path}")


def _safe_registry_root(path_str):
    root = Path(path_str)
    _reject_symlink_ancestors(root)
    if root.is_symlink():
        raise RegistryError(f"registry path must not be a symlink: {root}")
    if root.exists():
        if not root.is_dir():
            raise RegistryError(f"registry path is not a directory: {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_subdir(root, name):
    p = root / name
    if p.is_symlink():
        raise RegistryError(f"{name} directory must not be a symlink: {p}")
    if p.exists():
        if not p.is_dir():
            raise RegistryError(f"{name} path is not a directory: {p}")
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _validate_metadata_field(value, label):
    if len(value) > MAX_EVIDENCE_LEN:
        raise RegistryError(f"{label} too long")
    if "\n" in value or "\r" in value:
        raise RegistryError(f"{label} must not contain newlines")
    if _SECRET_PATTERN.search(value):
        raise RegistryError(f"{label} looks like a secret and was rejected")


def _validate_evidence(evidence):
    _validate_metadata_field(evidence, "evidence")


# --------------------------------------------------------------------------
# I/O bas niveau : verrou, owner.json, journal JSONL
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _locked(lock_path):
    fd = _open_secure(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _read_json(path):
    fd = _open_secure(path, os.O_RDONLY)
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        return json.load(f)


def _fsync_dir(dir_path):
    """fsync du répertoire (no-follow) pour rendre un os.replace durable."""
    fd = _open_secure(dir_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parse_jsonl_lines(text, source):
    """Parse fail-closed : toute ligne corrompue lève, aucune ligne n'est ignorée."""
    events = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RegistryError(
                f"corrupt journal {source} at line {lineno}: {exc}"
            ) from exc
    return events


def _read_all_events(lane_file):
    """Lit le journal via ouverture secure no-follow + flock partagé (cohérent
    avec le flock exclusif utilisé pendant l'append)."""
    fd = _open_secure(lane_file, os.O_RDONLY)
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            text = f.read()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return _parse_jsonl_lines(text, lane_file)


def _same_event(a, b):
    keys = (set(a.keys()) | set(b.keys())) - {"ts"}
    return all(a.get(k) == b.get(k) for k in keys)


# --------------------------------------------------------------------------
# I/O durci dirfd/openat — écriture claim/admit fail-closed contre un swap
# d'ancêtre symlink (registry/locks, registry/lanes) survenant APRÈS la
# validation Path. `O_NOFOLLOW` sur la feuille seule ne suffit pas : chaque
# composant du chemin est ré-ouvert via openat `O_NOFOLLOW|O_DIRECTORY` depuis
# un fd du registry root, donc un ancêtre swappé fait échouer l'openat
# (fail-closed) et aucune écriture ne peut sortir du registry.
# --------------------------------------------------------------------------

_ODIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_REPLACE_SUPPORTS_DIR_FD = os.replace in os.supports_dir_fd
_SYMLINK_ERRNOS = (errno.ELOOP, errno.ENOTDIR, errno.EMLINK)


def _openat_subdir(parent_fd, name, create):
    """openat d'un sous-répertoire via `O_NOFOLLOW|O_DIRECTORY`. Lève
    RegistryError si `name` est un symlink (swap d'ancêtre). Propage
    FileNotFoundError si `name` est simplement absent et `create` est faux."""
    if create:
        with contextlib.suppress(FileExistsError):
            os.mkdir(name, 0o700, dir_fd=parent_fd)
    try:
        return os.open(
            name, os.O_RDONLY | _NOFOLLOW_FLAG | _ODIRECTORY_FLAG, dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in _SYMLINK_ERRNOS:
            raise RegistryError(
                f"refusing symlinked registry directory: {name!r}"
            ) from exc
        raise


def _open_dir_chain(root_path, parts, create=False):
    """Descend depuis le registry root via openat `O_NOFOLLOW` à chaque composant.

    Retourne un fd du dernier répertoire (à fermer par l'appelant) ; ferme tous
    les fds intermédiaires. Ré-ouvrir cette chaîne à chaque écriture re-valide
    tous les ancêtres et referme la fenêtre TOCTOU du swap symlink."""
    fd = _open_secure(root_path, os.O_RDONLY | _ODIRECTORY_FLAG)
    try:
        for part in parts:
            nxt = _openat_subdir(fd, part, create)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_json_at(parent_fd, name):
    fd = os.open(name, os.O_RDONLY | _NOFOLLOW_FLAG, dir_fd=parent_fd)
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_replace_at(parent_fd, parent_path, tmp_name, final_name):
    """Rename atomique tmp -> final à l'intérieur de `parent_fd`.

    Sur les plateformes exposant renameat(dir_fd) (Linux), l'opération est
    entièrement relative au fd. Sinon (macOS), on re-valide que le chemin textuel
    désigne toujours l'inode ouvert (anti-swap) avant un rename textuel ; comme
    le tmp vit dans le vrai répertoire (créé via dir_fd), un swap tardif ferait
    de toute façon échouer le rename (source introuvable) — fail-closed."""
    if _REPLACE_SUPPORTS_DIR_FD:
        os.replace(tmp_name, final_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        return
    st_fd = os.fstat(parent_fd)
    try:
        st_path = os.stat(parent_path, follow_symlinks=False)
    except OSError as exc:
        raise RegistryError(
            f"registry path vanished during write: {parent_path}"
        ) from exc
    if (st_fd.st_dev, st_fd.st_ino) != (st_path.st_dev, st_path.st_ino):
        raise RegistryError(f"registry ancestor swapped during write: {parent_path}")
    os.replace(
        os.path.join(parent_path, tmp_name), os.path.join(parent_path, final_name),
    )


def _write_json_at(parent_fd, parent_path, name, obj, mode=0o600):
    """Écrit `obj` (JSON) sous `name` dans `parent_fd`, atomiquement."""
    tmp_name = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    fd = os.open(
        tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW_FLAG, mode,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace_at(parent_fd, parent_path, tmp_name, name)
        os.fsync(parent_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name, dir_fd=parent_fd)
        raise


def _write_text_at(parent_fd, parent_path, name, content, mode=0o600):
    """Write text atomically inside an already verified directory fd."""
    tmp_name = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    fd = os.open(
        tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW_FLAG, mode,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace_at(parent_fd, parent_path, tmp_name, name)
        os.fsync(parent_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name, dir_fd=parent_fd)
        raise


def _append_event_at(parent_fd, name, source_label, key, event_name, extra=None):
    payload = {"ts": time.time(), "key": key, "event": event_name}
    if extra:
        payload.update(extra)
    fd = os.open(
        name, os.O_CREAT | os.O_RDWR | _NOFOLLOW_FLAG, 0o600, dir_fd=parent_fd,
    )
    with os.fdopen(fd, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            os.fchmod(f.fileno(), 0o600)
            events = _parse_jsonl_lines(f.read(), source_label)
            last = events[-1] if events else None
            if last is not None and _same_event(last, payload):
                return
            f.seek(0, os.SEEK_END)
            f.write(json.dumps(payload, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _dirfd_flock(root_path, parts, name):
    """flock exclusif sur `name` sous la chaîne dirfd `parts` (créée sûrement)."""
    parent_fd = _open_dir_chain(root_path, parts, create=True)
    try:
        fd = os.open(
            name, os.O_CREAT | os.O_RDWR | _NOFOLLOW_FLAG, 0o600, dir_fd=parent_fd,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    finally:
        os.close(parent_fd)


def _read_owner_via_chain(root_path, key):
    """owner.json de la lane `key` via chaîne dirfd, ou None si absent."""
    try:
        key_fd = _open_dir_chain(root_path, ("locks", key), create=False)
    except FileNotFoundError:
        return None
    try:
        return _read_json_at(key_fd, "owner.json")
    except FileNotFoundError:
        return None
    finally:
        os.close(key_fd)


def _write_owner_via_chain(root_path, key, owner):
    key_fd = _open_dir_chain(root_path, ("locks", key), create=True)
    try:
        parent_path = os.path.join(root_path, "locks", key)
        _write_json_at(key_fd, parent_path, "owner.json", owner)
    finally:
        os.close(key_fd)


def _unlink_owner_via_chain(root_path, key):
    try:
        key_fd = _open_dir_chain(root_path, ("locks", key), create=False)
    except FileNotFoundError:
        return
    try:
        with contextlib.suppress(FileNotFoundError):
            os.unlink("owner.json", dir_fd=key_fd)
        os.fsync(key_fd)
    finally:
        os.close(key_fd)


def _append_event_via_chain(root_path, key, event_name, extra=None):
    lanes_fd = _open_dir_chain(root_path, ("lanes",), create=True)
    try:
        _append_event_at(
            lanes_fd, f"{key}.jsonl",
            os.path.join(root_path, "lanes", f"{key}.jsonl"),
            key, event_name, extra=extra,
        )
    finally:
        os.close(lanes_fd)


def _compute_status(events):
    status = "unclaimed"
    for e in events:
        name = e.get("event")
        if name in STATUS_LADDER:
            status = STATUS_LADDER[name]
    return status


# --------------------------------------------------------------------------
# Identité de processus / activité worktree (pour la réclamation de lock)
# --------------------------------------------------------------------------

def _get_process_start_time(pid):
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def _get_process_state_char(pid):
    try:
        result = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out[:1] if out else None


def determine_process_state(owner):
    """Retourne 'alive' | 'zombie' | 'reused' | 'not_found'."""
    pid = owner.get("pid")
    if not isinstance(pid, int):
        return "not_found"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "not_found"
    except PermissionError:
        pass
    except OSError:
        return "not_found"

    if _get_process_state_char(pid) == "Z":
        return "zombie"

    current_start = _get_process_start_time(pid)
    recorded_start = owner.get("process_start_time")
    # Sans empreinte de départ enregistrée, la réutilisation de PID est
    # indémontrable : un process vivant reste 'alive' (jamais 'reused'), sinon
    # un owner vivant sans baseline deviendrait faussement réclamable.
    if recorded_start is None:
        return "alive"
    if current_start is not None and recorded_start != current_start:
        return "reused"
    return "alive"


_WORKTREE_SCAN_MAX_ENTRIES = 5000
_WORKTREE_SCAN_MAX_DEPTH = 8


def _get_worktree_last_active(path):
    """Dernière activité du worktree, avec refus conservateur d'un scan partiel.

    Ne lit aucun contenu et ne suit aucun symlink. Si la profondeur ou le
    nombre maximal d'entrées empêche un scan complet, retourne l'instant
    courant : le worktree est alors considéré actif et le reclaim est refusé.
    """
    if not path:
        return None
    root = Path(path)
    try:
        last = os.stat(root, follow_symlinks=False).st_mtime
    except OSError:
        return None

    remaining = [_WORKTREE_SCAN_MAX_ENTRIES]
    scan_complete = [True]

    def _scan(dir_path, depth):
        nonlocal last
        if depth > _WORKTREE_SCAN_MAX_DEPTH:
            scan_complete[0] = False
            return
        try:
            with os.scandir(dir_path) as it:
                for entry in it:
                    if remaining[0] <= 0:
                        scan_complete[0] = False
                        return
                    remaining[0] -= 1
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        scan_complete[0] = False
                        continue
                    if st.st_mtime > last:
                        last = st.st_mtime
                    if entry.is_dir(follow_symlinks=False):
                        _scan(entry.path, depth + 1)
        except OSError:
            scan_complete[0] = False

    _scan(root, 0)
    return time.time() if not scan_complete[0] else last


def evaluate_reclaim(now, owner, process_state, worktree_last_active):
    """Fonction pure : décide si un lock détenu par `owner` est réclamable.

    Réclamable seulement si le processus n'est PAS vivant, le TTL basé sur
    heartbeat est expiré, ET le worktree n'a pas été touché récemment.
    """
    ttl_seconds = owner["ttl_hours"] * 3600
    heartbeat_age = now - owner["heartbeat_at"]
    ttl_expired = heartbeat_age > ttl_seconds
    worktree_active = (
        worktree_last_active is not None
        and (now - worktree_last_active) < WORKTREE_ACTIVE_SECONDS
    )
    process_stale = process_state in ("zombie", "reused", "not_found")
    reclaimable = process_stale and ttl_expired and not worktree_active
    return {
        "reclaimable": reclaimable,
        "process_stale": process_stale,
        "ttl_expired": ttl_expired,
        "worktree_active": worktree_active,
    }


def _canonical_worktree(worktree):
    return os.path.realpath(str(worktree))


def _resolve_owner_identity(owner_pid, owner_start_time):
    """Valide une identité de process parent transportée par l'appelant.

    Le hard gate/gateway lance `factory_lane.py` en subprocess éphémère : si on
    persistait `os.getpid()`, l'owner porterait un PID mort dès la fin de la
    commande et deviendrait immédiatement réclamable. L'appelant transporte donc
    l'identité de l'agent parent de longue durée via `--owner-pid`
    (+ `--owner-start-time` optionnel, anti-usurpation).

    Retourne `(pid, start_time)` validés, ou `(None, None)` si aucune identité
    n'est transportée (mode CLI autonome — l'appelant EST le process propriétaire).
    Lève `RegistryError` pour un PID absent/mort ou une empreinte incohérente :
    on ne crée jamais un owner déjà mort.
    """
    if owner_pid is None:
        return None, None
    if not isinstance(owner_pid, int) or isinstance(owner_pid, bool) or owner_pid <= 0:
        raise RegistryError(f"invalid owner-pid: {owner_pid!r}")
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError as exc:
        raise RegistryError(f"owner-pid {owner_pid} is not a live process") from exc
    except PermissionError:
        pass  # vivant mais appartenant à un autre utilisateur
    except OSError as exc:
        raise RegistryError(f"owner-pid {owner_pid} is not usable: {exc}") from exc

    actual_start = _get_process_start_time(owner_pid)
    if owner_start_time is not None:
        if actual_start is None or owner_start_time != actual_start:
            raise RegistryError("owner-start-time does not match owner-pid start time")
        return owner_pid, owner_start_time
    return owner_pid, actual_start


def _build_owner(agent, session, worktree, ttl_hours, now, profile=None,
                 gateway_session_key=None, owner_pid=None, owner_start_time=None):
    if owner_pid is not None:
        pid = owner_pid
        start_time = owner_start_time
    else:
        pid = os.getpid()
        start_time = _get_process_start_time(pid)
    owner = {
        "host": socket.gethostname(),
        "agent": agent,
        "session_id": session,
        "pid": pid,
        "process_start_time": start_time,
        "started_at": now,
        "heartbeat_at": now,
        "ttl_hours": ttl_hours,
        "worktree": _canonical_worktree(worktree),
    }
    if profile:
        owner["profile"] = profile
    if gateway_session_key:
        owner["gateway_session_key"] = gateway_session_key
    return owner


def _is_same_session(owner, agent, session):
    return owner.get("agent") == agent and owner.get("session_id") == session


def _iter_owner_files(locks_root):
    if not locks_root.exists() or locks_root.is_symlink():
        return []
    try:
        entries = sorted(locks_root.iterdir())
    except OSError:
        return []
    owner_files = []
    for lock_dir in entries:
        if lock_dir.is_symlink() or not lock_dir.is_dir():
            continue
        owner_file = lock_dir / "owner.json"
        if owner_file.exists() and not owner_file.is_symlink():
            owner_files.append((lock_dir.name, owner_file))
    return owner_files


def _find_worktree_claim(locks_root, worktree_real):
    for key, owner_file in _iter_owner_files(locks_root):
        owner = _read_json(owner_file)
        claimed = owner.get("worktree")
        if claimed and _canonical_worktree(claimed) == worktree_real:
            return key, owner, owner_file
    return None


def _can_reclaim_owner(now, owner):
    process_state = determine_process_state(owner)
    worktree_last_active = _get_worktree_last_active(owner.get("worktree"))
    verdict = evaluate_reclaim(
        now=now,
        owner=owner,
        process_state=process_state,
        worktree_last_active=worktree_last_active,
    )
    return verdict["reclaimable"]


def _git_dirty(repo):
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=str(repo), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


# --------------------------------------------------------------------------
# Commandes
# --------------------------------------------------------------------------

def cmd_preflight(root, key, as_json):
    validate_key(key)
    lanes_dir = _safe_subdir(root, "lanes")
    locks_root = _safe_subdir(root, "locks")
    lane_file = lanes_dir / f"{key}.jsonl"

    if lane_file.exists():
        events = _read_all_events(lane_file)
        status = _compute_status(events)
    else:
        status = "unclaimed"

    owner_file = locks_root / key / "owner.json"
    payload = {"key": key, "status": status, "active": owner_file.exists()}
    if as_json:
        print(json.dumps(payload))
    else:
        print(f"{key}: {status}")
    return 0


def _validate_ttl_hours(ttl_hours):
    if not isinstance(ttl_hours, (int, float)) or isinstance(ttl_hours, bool):
        raise RegistryError(f"invalid ttl-hours: {ttl_hours!r}")
    if not math.isfinite(ttl_hours) or ttl_hours <= 0:
        raise RegistryError(f"invalid ttl-hours: {ttl_hours!r}")


def _claim_under_gate(root, key, agent, session, worktree, reclaim, ttl_hours,
                      profile=None, gateway_session_key=None,
                      owner_pid=None, owner_start_time=None):
    validate_key(key)
    _validate_ttl_hours(ttl_hours)
    # Un `gateway_session_key` ressemblant à un secret (token/password/api_key…)
    # ne doit jamais être persisté en clair dans owner.json — rejet fail-closed.
    if gateway_session_key is not None:
        _validate_metadata_field(gateway_session_key, "gateway_session_key")
    resolved_pid, resolved_start = _resolve_owner_identity(owner_pid, owner_start_time)

    # Validation Path-level (crée + refuse un symlink d'emblée) ; les écritures
    # elles-mêmes passent ensuite par les chaînes dirfd fail-closed ci-dessous.
    _safe_subdir(root, "lanes")
    locks_root = _safe_subdir(root, "locks")
    worktree_real = _canonical_worktree(worktree)

    lock_dir = locks_root / key
    if lock_dir.is_symlink():
        raise RegistryError(f"lock directory must not be a symlink: {lock_dir}")
    lock_dir.mkdir(parents=True, exist_ok=True)

    root_path = str(root)
    # Verrou machine-wide (préflight->claim TOCTOU) + verrou par-lane, tous deux
    # ouverts via chaîne dirfd O_NOFOLLOW ; toute écriture (owner.json, journal)
    # ré-ouvre sa chaîne et échoue fermé si un ancêtre a été swappé en symlink.
    with _dirfd_flock(root_path, (), ".worktree-admission.lock"), \
            _dirfd_flock(root_path, ("locks", key), ".lock"):
        now = time.time()
        owner = _read_owner_via_chain(root_path, key)
        if owner is not None:
            if _is_same_session(owner, agent, session):
                current_wt = owner.get("worktree")
                if not (current_wt and _canonical_worktree(current_wt) == worktree_real):
                    # Rebind vers un worktree différent : ne jamais réécrire
                    # l'owner vers un worktree déjà détenu par une autre lane
                    # (sinon deux owners pour le même worktree).
                    rebind_conflict = _find_worktree_claim(locks_root, worktree_real)
                    if rebind_conflict is not None and rebind_conflict[0] != key:
                        rc_owner = rebind_conflict[1]
                        raise RegistryError(
                            f"worktree already claimed by {rebind_conflict[0]} "
                            f"{rc_owner.get('agent')}/{rc_owner.get('session_id')}"
                        )
                    owner["worktree"] = worktree_real
                owner["heartbeat_at"] = now
                _write_owner_via_chain(root_path, key, owner)
                return 0

            if not reclaim:
                raise RegistryError(
                    f"lane {key} already claimed by "
                    f"{owner.get('agent')}/{owner.get('session_id')}"
                )

            if not _can_reclaim_owner(now, owner):
                raise RegistryError(
                    f"lane {key} owner still active, refusing --reclaim"
                )

            previous_agent = owner.get("agent")
            previous_session = owner.get("session_id")
            new_owner = _build_owner(
                agent, session, worktree_real, ttl_hours, now,
                profile=profile, gateway_session_key=gateway_session_key,
                owner_pid=resolved_pid, owner_start_time=resolved_start,
            )
            _write_owner_via_chain(root_path, key, new_owner)
            _append_event_via_chain(root_path, key, "lock_reclaimed", extra={
                "previous_agent": previous_agent,
                "previous_session": previous_session,
            })
            return 0

        conflict = _find_worktree_claim(locks_root, worktree_real)
        if conflict is not None:
            conflict_key, conflict_owner, _conflict_owner_file = conflict
            if conflict_key != key:
                if not reclaim:
                    raise RegistryError(
                        f"worktree already claimed by {conflict_key} "
                        f"{conflict_owner.get('agent')}/{conflict_owner.get('session_id')}"
                    )
                if not _can_reclaim_owner(now, conflict_owner):
                    raise RegistryError(
                        f"worktree already claimed by active owner {conflict_key}"
                    )
                _unlink_owner_via_chain(root_path, conflict_key)

        new_owner = _build_owner(
            agent, session, worktree_real, ttl_hours, now,
            profile=profile, gateway_session_key=gateway_session_key,
            owner_pid=resolved_pid, owner_start_time=resolved_start,
        )
        _write_owner_via_chain(root_path, key, new_owner)
        _append_event_via_chain(root_path, key, "lane_claimed")
        return 0


def cmd_claim(root, key, agent, session, worktree, reclaim, ttl_hours,
              profile=None, gateway_session_key=None,
              owner_pid=None, owner_start_time=None):
    return _claim_under_gate(
        root, key, agent, session, worktree, reclaim, ttl_hours,
        profile=profile, gateway_session_key=gateway_session_key,
        owner_pid=owner_pid, owner_start_time=owner_start_time,
    )


def _lane_prefix(key):
    return key.split("-", 1)[0]


def _validate_profile_domain(key, profile, domain_prefixes):
    if not profile or not domain_prefixes:
        return
    allowed = {part.strip() for part in domain_prefixes.split(",") if part.strip()}
    if allowed and _lane_prefix(key) not in allowed:
        raise RegistryError(f"profile {profile} cannot own lane {key}")


def cmd_admit(root, key, mode, hard, agent, session, worktree, ttl_hours,
              as_json=False, profile=None, gateway_session_key=None,
              domain_prefixes=None, owner_pid=None, owner_start_time=None):
    validate_key(key)
    _validate_ttl_hours(ttl_hours)
    locks_root = _safe_subdir(root, "locks")
    _safe_subdir(root, "lanes")
    worktree_real = _canonical_worktree(worktree)
    gate_file = root / ".worktree-admission.lock"

    with _locked(gate_file):
        conflict = _find_worktree_claim(locks_root, worktree_real)
        if mode == "reviewer":
            payload = {
                "key": key,
                "worktree": worktree_real,
                "decision": "reviewer_allowed",
            }
            if conflict:
                payload["owner_key"] = conflict[0]
                payload["owner_agent"] = conflict[1].get("agent")
                payload["owner_session"] = conflict[1].get("session_id")
            if as_json:
                print(json.dumps(payload, sort_keys=True))
            return 0

        _validate_profile_domain(key, profile, domain_prefixes)
        if hard and conflict is None and _git_dirty(worktree_real):
            raise RegistryError(f"dirty ownerless worktree: {worktree_real}")

    return _claim_under_gate(
        root, key, agent, session, worktree_real, False, ttl_hours,
        profile=profile, gateway_session_key=gateway_session_key,
        owner_pid=owner_pid, owner_start_time=owner_start_time,
    )


# --------------------------------------------------------------------------
# Hard gate pré-mutation (lecture seule) — appelé par le hook `pre_tool_call`
# --------------------------------------------------------------------------

def evaluate_admission_guard(root, worktree_real, agent, session,
                             profile=None, domain_prefixes=None):
    """Décision de gate pré-mutation, en LECTURE SEULE (aucun owner écrit, aucun
    PID éphémère persisté).

    Retourne `(allowed: bool, reason: str | None)`.

    - Worktree non revendiqué -> `(True, None)` : advisory fail-open (le gate
      n'agit que sur des lanes réellement admises).
    - Profil métier borné (`profile` + `domain_prefixes`) et lane possédant le
      worktree hors domaine -> `(False, ...)` : refus automatique, indépendant de
      la session et de la liveness.
    - Autre session détenant le worktree avec un process VIVANT -> `(False, ...)`
      : anti double-occupation (un seul gagnant par worktree).
    - Sinon -> `(True, None)`.
    """
    match = _find_claim_for_worktree(root, worktree_real)
    if match is None:
        return True, None
    key, owner = match

    if profile and domain_prefixes:
        allowed = {p.strip() for p in domain_prefixes.split(",") if p.strip()}
        if allowed and _lane_prefix(key) not in allowed:
            return False, (
                f"profile {profile} cannot mutate lane {key} "
                f"(out of domain {sorted(allowed)})"
            )

    if not _is_same_session(owner, agent, session):
        if determine_process_state(owner) == "alive":
            return False, (
                f"worktree owned by {key} "
                f"{owner.get('agent', '?')}/{owner.get('session_id', '?')}"
            )

    return True, None


def cmd_guard(root, repo, agent, session, profile=None, domain_prefixes=None,
              as_json=False):
    """Hard gate CLI (debug/canary). Le vrai flux runtime importe
    `evaluate_admission_guard` depuis `factory_admission_hook.py`."""
    worktree_real = _git_toplevel_or_none(repo) or os.path.realpath(repo)
    allowed, reason = evaluate_admission_guard(
        root, worktree_real, agent, session,
        profile=profile, domain_prefixes=domain_prefixes,
    )
    if as_json:
        print(json.dumps(
            {"allowed": allowed, "reason": reason, "worktree": worktree_real},
            sort_keys=True,
        ))
    if not allowed:
        if not as_json:
            print(f"BLOCKED: {reason}", file=sys.stderr)
        return 1
    return 0


def cmd_event(root, key, event_name, evidence, pr, commit=None, ci=None, deploy=None):
    validate_key(key)
    _safe_subdir(root, "lanes")
    _safe_subdir(root, "locks")

    if event_name not in ALLOWED_MANUAL_EVENTS:
        raise RegistryError(f"unknown event: {event_name!r}")
    if evidence is not None:
        _validate_evidence(evidence)
    if commit is not None:
        _validate_metadata_field(commit, "commit")
    if ci is not None:
        _validate_metadata_field(ci, "ci")
    if deploy is not None:
        _validate_metadata_field(deploy, "deploy")
    if pr is not None:
        _validate_metadata_field(pr, "pr")
    if event_name == "pr_opened" and not pr:
        raise RegistryError("pr_opened requires --pr")

    root_path = str(root)
    with _dirfd_flock(root_path, ("locks", key), ".lock"):
        owner = _read_owner_via_chain(root_path, key)
        if owner is None:
            raise RegistryError(f"no active claim for lane {key}")
        owner["heartbeat_at"] = time.time()
        _write_owner_via_chain(root_path, key, owner)

        extra = {}
        if evidence is not None:
            extra["evidence"] = evidence
        if pr is not None:
            extra["pr"] = pr
        if commit is not None:
            extra["commit"] = commit
        if ci is not None:
            extra["ci"] = ci
        if deploy is not None:
            extra["deploy"] = deploy
        _append_event_via_chain(root_path, key, event_name, extra=extra)
    return 0


def cmd_handoff(root, key, status, next_step, evidence):
    validate_key(key)
    if status not in ALLOWED_HANDOFF_STATUS:
        raise RegistryError(f"unknown handoff status: {status!r}")
    if "\n" in next_step or "\r" in next_step:
        raise RegistryError("next-step must not contain newlines")
    if len(next_step) > MAX_NEXT_STEP_LEN:
        raise RegistryError("next-step too long")
    if _SECRET_PATTERN.search(next_step):
        raise RegistryError("next-step looks like a secret and was rejected")
    if evidence is not None:
        _validate_evidence(evidence)

    _safe_subdir(root, "lanes")
    _safe_subdir(root, "locks")
    _safe_subdir(root, "handoffs")

    root_path = str(root)
    with _dirfd_flock(root_path, ("locks", key), ".lock"):
        if _read_owner_via_chain(root_path, key) is None:
            raise RegistryError(f"no active claim for lane {key}")

        now = time.time()
        ts_ms = int(now * 1000)
        date_suffix = time.strftime("%Y-%m-%d", time.gmtime(now))
        handoff_name = f"{key}-{date_suffix}.md"
        lines = [
            f"# Handoff {key}",
            f"Status: {status}",
            f"Next step: {next_step}",
        ]
        if evidence is not None:
            lines.append(f"Evidence: {evidence}")
        lines.append(f"Timestamp: {ts_ms}")
        content = "\n".join(lines) + "\n"

        handoffs_fd = _open_dir_chain(root_path, ("handoffs",), create=True)
        try:
            _write_text_at(
                handoffs_fd, os.path.join(root_path, "handoffs"), handoff_name, content,
            )
        finally:
            os.close(handoffs_fd)

        _append_event_via_chain(root_path, key, "handoff", extra={
            "status": status, "handoff_file": handoff_name,
        })
    return 0


def cmd_render(root):
    lanes_dir = _safe_subdir(root, "lanes")
    locks_root = _safe_subdir(root, "locks")

    entries = []
    for lane_file in sorted(lanes_dir.glob("*.jsonl")):
        key = lane_file.stem
        events = _read_all_events(lane_file)
        owner_file = locks_root / key / "owner.json"
        if not owner_file.exists():
            continue
        entries.append((key, _compute_status(events)))

    lines = ["# LANES", ""]
    if not entries:
        lines.append("_No active lanes._")
    else:
        lines.append("| Key | Status |")
        lines.append("|-----|--------|")
        for key, status in entries:
            lines.append(f"| {key} | {status} |")
    content = "\n".join(lines) + "\n"

    root_path = str(root)
    root_fd = _open_dir_chain(root_path, (), create=False)
    try:
        _write_text_at(root_fd, root_path, "LANES.md", content, mode=0o644)
    finally:
        os.close(root_fd)
    return 0


def cmd_close(root, key):
    validate_key(key)
    _safe_subdir(root, "lanes")
    _safe_subdir(root, "locks")

    root_path = str(root)
    with _dirfd_flock(root_path, ("locks", key), ".lock"):
        if _read_owner_via_chain(root_path, key) is None:
            raise RegistryError(f"no active claim for lane {key}")
        _append_event_via_chain(root_path, key, "lane_closed")
        _unlink_owner_via_chain(root_path, key)
    return 0


# --------------------------------------------------------------------------
# Context pack (HER-55 Scope A) — Hermes -> Claude Code
# --------------------------------------------------------------------------

def _lane_status(root, key):
    """Statut de lane courant, sans effet de bord (aucune création de dossier)."""
    lane_file = root / "lanes" / f"{key}.jsonl"
    if not lane_file.exists():
        return "unclaimed"
    try:
        events = _read_all_events(lane_file)
    except RegistryError:
        return "unclaimed"
    return _compute_status(events)


def _load_context_map(path):
    """Charge `context-map.json`. Toute anomalie (absent, symlink, corrompu)
    dégrade silencieusement vers "aucun repo mappé", jamais une exception :
    un repo mal mappé doit se comporter comme un repo inconnu, pas planter."""
    path = Path(path)
    try:
        if path.is_symlink() or not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    repos = data.get("repos")
    if not isinstance(repos, dict):
        return {}
    return repos


def _is_blocked_vault_path(relpath):
    """Défense en profondeur : blocage même si `allow_files` liste le chemin."""
    if not isinstance(relpath, str) or not relpath:
        return True
    posix = relpath.replace(os.sep, "/").lstrip("/")
    basename = posix.rsplit("/", 1)[-1]
    if basename == ".env" or basename.startswith(".env"):
        return True
    if _SECRET_PATTERN.search(posix):
        return True
    return any(
        posix == prefix.rstrip("/") or posix.startswith(prefix)
        for prefix in _VAULT_BLOCKED_DIR_PREFIXES
    )


def _read_vault_file_safely(vault_root, relpath):
    """Lit un fichier du vault sans jamais suivre un symlink ni sortir de
    `vault_root`. Retourne None (jamais une exception) sur toute anomalie."""
    if not isinstance(relpath, str) or not relpath or relpath.startswith("/"):
        return None
    target = vault_root / relpath
    try:
        _reject_symlink_ancestors(target)
        if target.is_symlink():
            return None
        resolved = target.resolve(strict=True)
        resolved.relative_to(vault_root)
    except (OSError, ValueError, RegistryError):
        return None
    try:
        fd = _open_secure(resolved, os.O_RDONLY)
    except (OSError, RegistryError):
        return None
    try:
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _filter_markdown_sections(content, allow_sections):
    """Ne garde que les sections `## Titre` explicitement autorisées.

    Liste vide -> aucune restriction (fichier entier)."""
    if not allow_sections:
        return content
    allowed = set(allow_sections)
    kept = []
    keep = False
    for line in content.splitlines():
        if line.startswith("## "):
            keep = line in allowed
        if keep:
            kept.append(line)
    return "\n".join(kept)


def _linear_cache_banner(root, key):
    """Bannière missing/stale sans jamais planter ; None si cache frais."""
    cache_path = root / "linear-cache" / f"{key}.json"
    try:
        if cache_path.is_symlink() or not cache_path.exists():
            return f"> Linear cache MISSING for {key}."
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"> Linear cache MISSING or unreadable for {key}."

    cached_at = data.get("cached_at")
    if not isinstance(cached_at, (int, float)) or isinstance(cached_at, bool):
        return f"> Linear cache STALE for {key} (no cached_at)."

    age_hours = (time.time() - cached_at) / 3600
    if age_hours > LINEAR_CACHE_STALE_HOURS:
        return f"> Linear cache STALE for {key} ({age_hours:.1f}h old)."

    title = data.get("title")
    if isinstance(title, str) and title:
        return f"Linear: {key} — {title}"
    return f"Linear cache: fresh for {key}."


def _render_unmapped_brief(key, repo_real, status):
    lines = [
        f"# Context Claude Code — {key}",
        UNMAPPED_REPO_MARKER,
        f"Repo: {repo_real}",
        f"Lane status: {status}",
    ]
    return "\n".join(lines) + "\n"


def _render_mapped_pack(key, repo_real, status, root, vault_root, entry):
    allow_files = entry.get("allow_files") or []
    allow_sections = entry.get("allow_sections") or []

    lines = [
        f"# Context Claude Code — {key}",
        f"Repo: {repo_real}",
        f"Lane status: {status}",
    ]
    banner = _linear_cache_banner(root, key)
    if banner:
        lines.append(banner)
    lines.append("")

    for relpath in allow_files:
        if _is_blocked_vault_path(relpath):
            continue
        raw = _read_vault_file_safely(vault_root, relpath)
        if raw is None:
            continue
        filtered = _filter_markdown_sections(raw, allow_sections)
        if not filtered.strip():
            continue
        lines.append(f"## Fichier: {relpath}")
        lines.extend(filtered.splitlines())
        lines.append("")

    content = "\n".join(lines).rstrip("\n") + "\n"
    body_lines = content.splitlines()
    if len(body_lines) > CONTEXT_PACK_MAX_LINES:
        body_lines = body_lines[: CONTEXT_PACK_MAX_LINES - 1] + ["_(pack tronqué à 300 lignes)_"]
        content = "\n".join(body_lines) + "\n"
    return content


def _write_context_output(out_path, content):
    out_path = Path(out_path)
    _reject_symlink_ancestors(out_path)
    if out_path.is_symlink():
        raise RegistryError(f"context output must not be a symlink: {out_path}")
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise RegistryError(f"context output directory must not be a symlink: {parent}")

    fd, tmp_path = tempfile.mkstemp(dir=str(parent), prefix=".context-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, str(out_path))
        _fsync_dir(parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def _path_is_within(path, root):
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def cmd_context(root, key, repo, vault, context_map_arg, out_arg):
    validate_key(key)
    repo_real = os.path.realpath(repo)
    registry_output_root = root / "contexts"
    repo_output_root = Path(repo_real) / ".factory"
    out_path = Path(out_arg) if out_arg else registry_output_root / f"{key}.md"
    output_parent_real = Path(os.path.realpath(str(out_path.parent)))
    allowed_output_roots = (
        Path(os.path.realpath(str(registry_output_root))),
        Path(os.path.realpath(str(repo_output_root))),
    )
    if not any(
        _path_is_within(output_parent_real, allowed_root)
        for allowed_root in allowed_output_roots
    ):
        raise RegistryError(
            "context output must stay inside registry contexts or repo .factory"
        )

    context_map_path = Path(context_map_arg) if context_map_arg else root / "context-map.json"
    repos_map = _load_context_map(context_map_path)
    entry = repos_map.get(repo_real)
    status = _lane_status(root, key)

    if entry is None:
        # Repo inconnu : aucun accès au vault, pas même `os.path.realpath`.
        content = _render_unmapped_brief(key, repo_real, status)
        _write_context_output(out_path, content)
        return 0

    vault_root = Path(os.path.realpath(vault))
    content = _render_mapped_pack(key, repo_real, status, root, vault_root, entry)
    _write_context_output(out_path, content)
    return 0


# --------------------------------------------------------------------------
# Capture handoff (HER-55 Scope B) — Claude Code -> Hermes
# --------------------------------------------------------------------------

MAX_GIT_STATUS_ENTRIES = 5000
MAX_SUMMARY_TEXT_LEN = 1000
MAX_SUMMARY_BYTES = 64 * 1024
MAX_SUMMARY_NODES = 256
MAX_SUMMARY_DEPTH = 8
MAX_SUMMARY_CONTAINER_ITEMS = 128
MAX_HANDOFF_BYTES = 1024 * 1024
SUMMARY_ALLOWED_KEYS = frozenset({
    "tests", "decisions", "blockers", "next_step", "pr", "ci",
})


def _run_git(repo_real, *args, timeout=10):
    try:
        return subprocess.run(
            ["git", *args], cwd=str(repo_real), capture_output=True, text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RegistryError(f"git {' '.join(args)} failed: {exc}") from exc


def _git_toplevel(repo_path):
    result = _run_git(repo_path, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise RegistryError(f"--repo is not a git worktree: {repo_path}")
    return os.path.realpath(result.stdout.strip())


def _git_current_branch(repo_real):
    result = _run_git(repo_real, "rev-parse", "--abbrev-ref", "HEAD")
    if result.returncode != 0:
        raise RegistryError("failed to determine current branch")
    return result.stdout.strip()


def _git_head_commit(repo_real):
    result = _run_git(repo_real, "rev-parse", "HEAD")
    if result.returncode != 0:
        raise RegistryError("failed to determine HEAD commit")
    return result.stdout.strip()


def _git_ref_exists(repo_real, ref):
    result = _run_git(repo_real, "rev-parse", "--verify", "--quiet", ref)
    return result.returncode == 0


def _git_base_commit(repo_real, branch):
    """merge-base avec `main` si elle existe et diffère de `branch`, sinon
    commit racine (premier commit de l'historique)."""
    if branch and branch != "main" and _git_ref_exists(repo_real, "main"):
        result = _run_git(repo_real, "merge-base", branch, "main")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    result = _run_git(repo_real, "rev-list", "--max-parents=0", "HEAD")
    if result.returncode == 0:
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if lines:
            return lines[0]
    raise RegistryError("unable to determine base commit")


def _git_status_entries(repo_real):
    """Chemins + codes de statut seulement — jamais de contenu ni de diff.

    Le format NUL de porcelain v1 préserve les espaces, guillemets et la
    sous-chaîne littérale ` -> `. Pour rename/copy, Git place la destination
    dans le premier record et la source dans le record NUL suivant."""
    result = _run_git(repo_real, "status", "--porcelain=v1", "-z")
    if result.returncode != 0:
        raise RegistryError("failed to read git status")
    records = result.stdout.split("\0")
    entries = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(entries) >= MAX_GIT_STATUS_ENTRIES:
            raise RegistryError("too many git status entries to capture safely")
        if len(record) < 3:
            raise RegistryError("malformed git status record")
        code = record[:2]
        path = record[3:] if record[2:3] == " " else record[2:]
        if "R" in code or "C" in code:
            if index >= len(records) or not records[index]:
                raise RegistryError("malformed git rename/copy status record")
            index += 1  # source path; destination is the first path in -z mode
        entries.append({"path": path, "status": code})
    return entries


def _validate_summary_text(value, label):
    if len(value) > MAX_SUMMARY_TEXT_LEN:
        raise RegistryError(f"summary field {label} too long")
    if "\n" in value or "\r" in value:
        raise RegistryError(f"summary field {label} must not contain newlines")
    if _SECRET_PATTERN.search(value):
        raise RegistryError(f"summary field {label} looks like a secret and was rejected")


def _validate_summary_value(label, value, depth=0, budget=None):
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > MAX_SUMMARY_NODES:
        raise RegistryError("summary has too many nodes")
    if depth > MAX_SUMMARY_DEPTH:
        raise RegistryError("summary is nested too deeply")
    if isinstance(value, str):
        _validate_summary_text(value, label)
        return value
    if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        if len(value) > MAX_SUMMARY_CONTAINER_ITEMS:
            raise RegistryError(f"summary field {label} has too many items")
        return [
            _validate_summary_value(f"{label}[]", item, depth + 1, budget)
            for item in value
        ]
    if isinstance(value, dict):
        if len(value) > MAX_SUMMARY_CONTAINER_ITEMS:
            raise RegistryError(f"summary field {label} has too many entries")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RegistryError(f"summary field {label} has a non-string key")
            _validate_summary_text(key, f"{label}.key")
            result[key] = _validate_summary_value(
                f"{label}.{key}", item, depth + 1, budget
            )
        return result
    raise RegistryError(f"summary field {label} has an unsupported type")


def _load_summary(summary_path):
    path = Path(summary_path)
    if path.is_symlink():
        raise RegistryError(f"summary path must not be a symlink: {path}")
    if not path.exists() or not path.is_file():
        raise RegistryError(f"summary path is not a regular file: {path}")

    fd = _open_secure(path, os.O_RDONLY)
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        if os.fstat(f.fileno()).st_size > MAX_SUMMARY_BYTES:
            raise RegistryError("summary JSON is too large")
        raw = f.read(MAX_SUMMARY_BYTES + 1)
    if len(raw.encode("utf-8")) > MAX_SUMMARY_BYTES:
        raise RegistryError("summary JSON is too large")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"summary is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError("summary must be a JSON object")

    unknown = set(data.keys()) - SUMMARY_ALLOWED_KEYS
    if unknown:
        raise RegistryError(f"summary has unsupported keys: {sorted(unknown)}")
    budget = [0]
    return {
        k: _validate_summary_value(k, v, depth=0, budget=budget)
        for k, v in data.items()
    }


def _serialize_handoff(payload):
    content = json.dumps(payload, sort_keys=True) + "\n"
    if len(content.encode("utf-8")) > MAX_HANDOFF_BYTES:
        raise RegistryError("handoff JSON is too large")
    return content


def cmd_capture_handoff(root, key, repo, summary_path):
    validate_key(key)
    _safe_subdir(root, "lanes")
    _safe_subdir(root, "locks")
    _safe_subdir(root, "handoffs")

    summary_fields = {}
    if summary_path is not None:
        summary_fields = _load_summary(summary_path)

    root_path = str(root)
    with _dirfd_flock(root_path, ("locks", key), ".lock"):
        owner = _read_owner_via_chain(root_path, key)
        if owner is None:
            raise RegistryError(f"no active claim for lane {key}")

        repo_real = _git_toplevel(repo)
        claimed_worktree = owner.get("worktree")
        if claimed_worktree and os.path.realpath(claimed_worktree) != repo_real:
            raise RegistryError(
                f"--repo {repo_real} does not match claimed worktree {claimed_worktree}"
            )

        branch = _git_current_branch(repo_real)
        head_sha = _git_head_commit(repo_real)
        base_commit = _git_base_commit(repo_real, branch)
        git_status = _git_status_entries(repo_real)
        files_changed = sorted({entry["path"] for entry in git_status if entry["path"]})

        now = time.time()
        date_suffix = time.strftime("%Y-%m-%d", time.gmtime(now))
        handoff_name = f"{key}-{date_suffix}.handoff.json"

        payload = {
            "issue": key,
            "agent": owner.get("agent"),
            "session_id": owner.get("session_id"),
            "repo": repo_real,
            "worktree": repo_real,
            "branch": branch,
            "base_commit": base_commit,
            "head_commit": head_sha,
            "git_status": git_status,
            "files_changed": files_changed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        payload.update(summary_fields)

        content = _serialize_handoff(payload)
        handoffs_fd = _open_dir_chain(root_path, ("handoffs",), create=True)
        try:
            _write_text_at(
                handoffs_fd, os.path.join(root_path, "handoffs"), handoff_name, content,
            )
        finally:
            os.close(handoffs_fd)

        _append_event_via_chain(root_path, key, "handoff_captured", extra={
            "handoff_file": handoff_name,
        })
    return 0


# --------------------------------------------------------------------------
# Réconciliation Claude -> Hermes + hooks Claude Code (HER-55 Scope B suite)
# --------------------------------------------------------------------------

RECONCILE_STALE_HOURS = 24.0


def _git_toplevel_or_none(repo_path, timeout=5):
    """Comme `_git_toplevel` mais ne lève jamais : `None` sur tout échec
    (binaire git absent, timeout, pas un dépôt Git). Usage exclusif hooks
    fail-open — `reconcile` reste fail-closed via `_git_toplevel`."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    if not out:
        return None
    try:
        return os.path.realpath(out)
    except OSError:
        return None


def _find_claim_for_worktree(root, repo_real):
    """Cherche un `owner.json` actif dont le worktree realpath correspond à
    `repo_real`. Retourne `(key, owner)` ou `None`.

    Le gate runtime ne peut pas transformer un scan incomplet en ``ownerless``:
    un locks/owner remplacé par symlink, un fichier non régulier ou un échec de
    lecture est indiscernable d'un owner caché. Le scan descend donc depuis le
    fd du registry avec ``openat(..., O_NOFOLLOW)`` et propage ``RegistryError``
    au hook, qui refuse alors la mutation avant l'outil.
    """
    try:
        locks_fd = _open_dir_chain(str(root), ("locks",), create=False)
    except FileNotFoundError:
        return None
    except (OSError, RegistryError) as exc:
        raise RegistryError(f"registry lock scan failed: {exc}") from exc
    try:
        def assert_locks_path_stable():
            """Reject a textual locks/ path swapped after its no-follow open."""
            try:
                path_stat = os.stat(root / "locks", follow_symlinks=False)
            except OSError as exc:
                raise RegistryError(f"registry locks changed during owner scan: {exc}") from exc
            fd_stat = os.fstat(locks_fd)
            if (
                not stat.S_ISDIR(path_stat.st_mode)
                or (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise RegistryError("registry locks changed during owner scan")

        try:
            names = sorted(os.listdir(locks_fd))
        except OSError as exc:
            raise RegistryError(f"registry lock scan failed: {exc}") from exc
        assert_locks_path_stable()
        for key in names:
            try:
                validate_key(key)
                key_fd = _openat_subdir(locks_fd, key, create=False)
            except (OSError, RegistryError) as exc:
                raise RegistryError(f"registry lock scan failed for {key!r}: {exc}") from exc
            try:
                try:
                    owner_fd = os.open("owner.json", os.O_RDONLY | _NOFOLLOW_FLAG, dir_fd=key_fd)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise RegistryError(f"registry lock scan failed for {key!r}: {exc}") from exc
                try:
                    with os.fdopen(owner_fd, "r", encoding="utf-8") as f:
                        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                            raise RegistryError(f"owner record is not a regular file for {key!r}")
                        owner = json.load(f)
                except (OSError, ValueError, json.JSONDecodeError, RegistryError) as exc:
                    raise RegistryError(f"registry lock scan failed for {key!r}: {exc}") from exc
                if not isinstance(owner, dict):
                    raise RegistryError(f"registry lock scan failed for {key!r}: owner record is not an object")
                worktree = owner.get("worktree")
                if isinstance(worktree, str) and os.path.realpath(worktree) == repo_real:
                    assert_locks_path_stable()
                    return key, owner
            finally:
                os.close(key_fd)
        assert_locks_path_stable()
        return None
    finally:
        os.close(locks_fd)


def _load_handoff(handoff_path, handoffs_dir):
    path = Path(handoff_path)
    _reject_symlink_ancestors(path)
    if path.is_symlink():
        raise RegistryError(f"handoff path must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(Path(handoffs_dir).resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RegistryError("handoff path must stay inside registry/handoffs") from exc
    if not resolved.is_file():
        raise RegistryError(f"handoff path is not a regular file: {resolved}")

    fd = _open_secure(resolved, os.O_RDONLY)
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        if os.fstat(f.fileno()).st_size > MAX_HANDOFF_BYTES:
            raise RegistryError("handoff JSON is too large")
        raw = f.read(MAX_HANDOFF_BYTES + 1)
    if len(raw.encode("utf-8")) > MAX_HANDOFF_BYTES:
        raise RegistryError("handoff JSON is too large")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"handoff is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError("handoff must be a JSON object")
    return data


def _parse_handoff_timestamp(value):
    """epoch seconds (int/float) ou ISO8601 UTC `%Y-%m-%dT%H:%M:%SZ`.
    Toute autre forme -> `None` (traité comme périmé, jamais comme frais)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            return None
    return None


def _test_entry_has_artifact(entry, repo_real):
    """Une déclaration de test ne compte comme preuve que si elle est verte et
    pointe vers un artefact régulier sous `<repo>/.factory/test-artifacts/`.
    Les fichiers système, chemins externes et symlinks ne valent jamais preuve."""
    if not isinstance(entry, dict):
        return False
    if entry.get("result") != "pass":
        return False
    artifact_path = entry.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        return False
    try:
        p = Path(artifact_path)
        _reject_symlink_ancestors(p)
        if p.is_symlink() or not p.is_file():
            return False
        resolved = p.resolve(strict=True)
        allowed_root = (Path(repo_real) / ".factory" / "test-artifacts").resolve(strict=False)
        resolved.relative_to(allowed_root)
        return True
    except (OSError, ValueError, RegistryError):
        return False


def _evaluate_handoff(repo_real, owner, handoff):
    """Fonction pure (hors accès Git en lecture seule) : verdict fermé +
    raisons. Priorité conflicting > stale > partial > verified."""
    reasons = []

    if handoff.get("agent") != owner.get("agent"):
        reasons.append("agent_mismatch")
    if handoff.get("session_id") != owner.get("session_id"):
        reasons.append("session_mismatch")

    claimed_worktree = owner.get("worktree")
    handoff_worktree = handoff.get("worktree")
    worktree_ok = (
        isinstance(handoff_worktree, str)
        and bool(claimed_worktree)
        and os.path.realpath(handoff_worktree) == os.path.realpath(claimed_worktree)
    )
    if not worktree_ok:
        reasons.append("worktree_mismatch")

    try:
        actual_branch = _git_current_branch(repo_real)
    except RegistryError:
        actual_branch = None
    if actual_branch is None or handoff.get("branch") != actual_branch:
        reasons.append("branch_mismatch")

    try:
        actual_head = _git_head_commit(repo_real)
    except RegistryError:
        actual_head = None
    if actual_head is None or handoff.get("head_commit") != actual_head:
        reasons.append("head_commit_mismatch")

    conflicting = bool(reasons)

    ts = _parse_handoff_timestamp(handoff.get("timestamp"))
    stale = ts is None or (time.time() - ts) > (RECONCILE_STALE_HOURS * 3600)
    if stale:
        reasons.append("stale_timestamp")

    tests = handoff.get("tests")
    unproven_tests = isinstance(tests, list) and len(tests) > 0 and any(
        not _test_entry_has_artifact(entry, repo_real) for entry in tests
    )
    if unproven_tests:
        reasons.append("unproven_test_claim")

    if conflicting:
        verdict = "conflicting"
    elif stale:
        verdict = "stale"
    elif unproven_tests:
        verdict = "partial"
    else:
        verdict = "verified"

    return verdict, reasons


def cmd_reconcile(root, key, repo, handoff_path):
    validate_key(key)
    _safe_subdir(root, "lanes")
    _safe_subdir(root, "locks")
    handoffs_dir = _safe_subdir(root, "handoffs")

    handoff = _load_handoff(handoff_path, handoffs_dir)

    if handoff.get("issue") != key:
        raise RegistryError(
            f"handoff issue {handoff.get('issue')!r} does not match lane {key!r}"
        )

    repo_real = _git_toplevel(repo)

    handoff_repo = handoff.get("repo")
    if not isinstance(handoff_repo, str) or os.path.realpath(handoff_repo) != repo_real:
        raise RegistryError("handoff repo does not match --repo")

    handoff_canonical = handoff.get("canonical_repo")
    if handoff_canonical is not None:
        if not isinstance(handoff_canonical, str) or os.path.realpath(handoff_canonical) != repo_real:
            raise RegistryError("handoff canonical_repo does not match --repo")

    root_path = str(root)
    with _dirfd_flock(root_path, ("locks", key), ".lock"):
        owner = _read_owner_via_chain(root_path, key)
        if owner is None:
            raise RegistryError(f"no active claim for lane {key}")

        verdict, reasons = _evaluate_handoff(repo_real, owner, handoff)

        _append_event_via_chain(root_path, key, "reconciled", extra={
            "verdict": verdict,
            "reasons": reasons,
            "handoff_head_commit": handoff.get("head_commit"),
        })

    payload = {
        "key": key,
        "issue": key,
        "repo": repo_real,
        "verdict": verdict,
        "reasons": reasons,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def cmd_hook_session_start(root, repo, agent=None, session=None):
    """`SessionStart` : silencieux hors repo suivi/sans claim, brief court
    sinon. Fail-open strict : toute exception -> code 0 sans sortie."""
    try:
        repo_real = _git_toplevel_or_none(repo)
        if repo_real is None:
            return 0

        match = _find_claim_for_worktree(root, repo_real)
        if match is None:
            return 0
        key, owner = match

        status = _lane_status(root, key)
        if agent and session and not _is_same_session(owner, agent, session):
            print(
                "STOP: worktree already owned by "
                f"{key} {owner.get('agent', '?')}/{owner.get('session_id', '?')} "
                f"for {repo_real}"
            )
            return 0

        lines = [
            f"# Hermes — {key}",
            f"Repo: {repo_real}",
            f"Lane status: {status}",
            f"Agent: {owner.get('agent', '?')}",
        ]
        print("\n".join(lines[:UNMAPPED_BRIEF_MAX_LINES]))
        return 0
    except Exception:
        return 0


def cmd_hook_stop(root, repo, key, summary_path=None):
    """`Stop` : capture les faits Git objectifs (jamais transcript/contenu),
    silencieux hors repo suivi/sans claim. Fail-open strict."""
    try:
        validate_key(key)

        repo_real = _git_toplevel_or_none(repo)
        if repo_real is None:
            return 0

        root_path = str(root)
        owner = _read_owner_via_chain(root_path, key)
        if owner is None:
            return 0
        claimed_worktree = owner.get("worktree")
        if not claimed_worktree or os.path.realpath(claimed_worktree) != repo_real:
            return 0

        branch = _git_current_branch(repo_real)
        head_sha = _git_head_commit(repo_real)
        git_status = _git_status_entries(repo_real)
        files_changed = sorted({e["path"] for e in git_status if e["path"]})

        payload = {
            "issue": key,
            "agent": owner.get("agent"),
            "session_id": owner.get("session_id"),
            "repo": repo_real,
            "worktree": repo_real,
            "branch": branch,
            "head_commit": head_sha,
            "git_status": git_status,
            "files_changed": files_changed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
        }

        if summary_path is not None:
            try:
                payload.update(_load_summary(summary_path))
            except RegistryError:
                pass

        content = _serialize_handoff(payload)
        capture_name = f"{key}-stop-{time.time_ns()}.json"
        handoffs_fd = _open_dir_chain(root_path, ("handoffs",), create=True)
        try:
            _write_text_at(
                handoffs_fd, os.path.join(root_path, "handoffs"), capture_name, content,
            )
        finally:
            os.close(handoffs_fd)

        return 0
    except Exception:
        return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(prog="factory_lane")
    parser.add_argument("--registry", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    p_preflight = sub.add_parser("preflight")
    p_preflight.add_argument("key")
    p_preflight.add_argument("--json", action="store_true")

    p_claim = sub.add_parser("claim")
    p_claim.add_argument("key")
    p_claim.add_argument("--agent", required=True)
    p_claim.add_argument("--session", required=True)
    p_claim.add_argument("--worktree", required=True)
    p_claim.add_argument("--reclaim", action="store_true")
    p_claim.add_argument("--reclaim-worktree", action="store_true")
    p_claim.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    p_claim.add_argument("--profile")
    p_claim.add_argument("--gateway-session-key")
    p_claim.add_argument("--owner-pid", type=int)
    p_claim.add_argument("--owner-start-time")

    p_admit = sub.add_parser("admit")
    p_admit.add_argument("key")
    p_admit.add_argument("--mode", choices=("owner", "reviewer"), required=True)
    p_admit.add_argument("--hard", action="store_true")
    p_admit.add_argument("--agent", required=True)
    p_admit.add_argument("--session", required=True)
    p_admit.add_argument("--worktree", required=True)
    p_admit.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    p_admit.add_argument("--json", action="store_true")
    p_admit.add_argument("--profile")
    p_admit.add_argument("--gateway-session-key")
    p_admit.add_argument("--domain-prefixes")
    p_admit.add_argument("--owner-pid", type=int)
    p_admit.add_argument("--owner-start-time")

    p_guard = sub.add_parser("guard")
    p_guard.add_argument("--repo", required=True)
    p_guard.add_argument("--agent", required=True)
    p_guard.add_argument("--session", required=True)
    p_guard.add_argument("--profile")
    p_guard.add_argument("--domain-prefixes")
    p_guard.add_argument("--json", action="store_true")

    p_event = sub.add_parser("event")
    p_event.add_argument("key")
    p_event.add_argument("event_name")
    p_event.add_argument("--evidence")
    p_event.add_argument("--pr")
    p_event.add_argument("--commit")
    p_event.add_argument("--ci")
    p_event.add_argument("--deploy")

    sub.add_parser("render")

    p_handoff = sub.add_parser("handoff")
    p_handoff.add_argument("key")
    p_handoff.add_argument("--status", required=True)
    p_handoff.add_argument("--next-step", required=True)
    p_handoff.add_argument("--evidence")

    p_close = sub.add_parser("close")
    p_close.add_argument("key")

    p_context = sub.add_parser("context")
    p_context.add_argument("key")
    p_context.add_argument("--repo", required=True)
    p_context.add_argument("--vault", required=True)
    p_context.add_argument("--context-map")
    p_context.add_argument("--out")

    p_capture_handoff = sub.add_parser("capture-handoff")
    p_capture_handoff.add_argument("key")
    p_capture_handoff.add_argument("--repo", required=True)
    p_capture_handoff.add_argument("--summary")

    p_reconcile = sub.add_parser("reconcile")
    p_reconcile.add_argument("key")
    p_reconcile.add_argument("--repo", required=True)
    p_reconcile.add_argument("--handoff", required=True)

    p_hook_session_start = sub.add_parser("hook-session-start")
    p_hook_session_start.add_argument("--repo", required=True)
    p_hook_session_start.add_argument("--agent")
    p_hook_session_start.add_argument("--session")

    p_hook_stop = sub.add_parser("hook-stop")
    p_hook_stop.add_argument("--repo", required=True)
    p_hook_stop.add_argument("--key", required=True)
    p_hook_stop.add_argument("--summary")

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        root = _safe_registry_root(args.registry)
        if args.command == "preflight":
            return cmd_preflight(root, args.key, args.json)
        if args.command == "claim":
            return cmd_claim(
                root, args.key, args.agent, args.session, args.worktree,
                args.reclaim or args.reclaim_worktree, args.ttl_hours,
                profile=args.profile, gateway_session_key=args.gateway_session_key,
                owner_pid=args.owner_pid, owner_start_time=args.owner_start_time,
            )
        if args.command == "admit":
            return cmd_admit(
                root, args.key, args.mode, args.hard, args.agent, args.session,
                args.worktree, args.ttl_hours, as_json=args.json,
                profile=args.profile,
                gateway_session_key=args.gateway_session_key,
                domain_prefixes=args.domain_prefixes,
                owner_pid=args.owner_pid, owner_start_time=args.owner_start_time,
            )
        if args.command == "guard":
            return cmd_guard(
                root, args.repo, args.agent, args.session,
                profile=args.profile, domain_prefixes=args.domain_prefixes,
                as_json=args.json,
            )
        if args.command == "event":
            return cmd_event(
                root, args.key, args.event_name, args.evidence, args.pr,
                commit=args.commit, ci=args.ci, deploy=args.deploy,
            )
        if args.command == "render":
            return cmd_render(root)
        if args.command == "handoff":
            return cmd_handoff(root, args.key, args.status, args.next_step, args.evidence)
        if args.command == "close":
            return cmd_close(root, args.key)
        if args.command == "context":
            return cmd_context(
                root, args.key, args.repo, args.vault,
                args.context_map, args.out,
            )
        if args.command == "capture-handoff":
            return cmd_capture_handoff(root, args.key, args.repo, args.summary)
        if args.command == "reconcile":
            return cmd_reconcile(root, args.key, args.repo, args.handoff)
        if args.command == "hook-session-start":
            return cmd_hook_session_start(root, args.repo, args.agent, args.session)
        if args.command == "hook-stop":
            return cmd_hook_stop(root, args.repo, args.key, args.summary)
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 1
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # fail-closed sur toute erreur inattendue
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
