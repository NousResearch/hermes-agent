"""A2A persistence layer — file locking, context/peer/session persistence,
and routing utilities.

Extracted from ``adapter.py`` to keep the main adapter module focused on
HTTP transport, server infrastructure, and adapter lifecycle.  These are
module-level functions consumed by both the adapter class and the
``task_routing`` mixin.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Platform-conditional imports ────────────────────────────────────────
try:
    import contextlib
except ImportError:
    import contextlib2 as contextlib  # type: ignore[no-redef]

try:
    import fcntl  # type: ignore[import-not-found]
    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

try:
    import msvcrt  # type: ignore[import-not-found]
    _HAS_MSVCRT = True
except ImportError:
    msvcrt = None  # type: ignore[assignment]
    _HAS_MSVCRT = False

# ── Constants shared with adapter.py ───────────────────────────────────
_DEFAULT_PORT = 9900
_MAX_CONTEXT_PEERS = 4096

# ── Portable file-based locking for persistence transactions ────────────
_THREAD_FALLBACK_LOCK = threading.Lock()
_MSVCRT_RETRIES = 50
_MSVCRT_RETRY_DELAY = 0.01


def _retained_terminal_ids(
    records,
    terminal_states,
    *,
    max_terminal: int = 500,
) -> set[str]:
    """Select terminal history by completion time, then stable insertion order."""
    ranked: list[tuple[float, int, str]] = []
    for index, (task_id, record) in enumerate(records.items()):
        if record.get("state", "") not in terminal_states:
            continue
        timestamp = record.get("completed_at")
        if timestamp is None:
            timestamp = record.get("created_at", 0)
        try:
            sortable_timestamp = float(timestamp)
        except (TypeError, ValueError):
            sortable_timestamp = 0.0
        ranked.append((sortable_timestamp, index, task_id))
    ranked.sort()
    return {task_id for _timestamp, _index, task_id in ranked[-max_terminal:]}


def _durable_task_snapshot(
    records,
    terminal_states,
    *,
    max_terminal: int = 500,
) -> dict[str, dict]:
    """Keep every live task and only the newest bounded terminal history."""
    kept_terminal = _retained_terminal_ids(
        records,
        terminal_states,
        max_terminal=max_terminal,
    )
    snapshot: dict[str, dict] = {}
    for task_id, record in records.items():
        state = record.get("state", "")
        if state in terminal_states and task_id not in kept_terminal:
            continue
        snapshot[task_id] = {
            "task_id": record.get("task_id", task_id),
            "context_id": record.get("context_id", ""),
            "peer": record.get("peer", ""),
            "agent_slug": record.get("agent_slug", ""),
            "tenant": record.get("tenant", ""),
            "state": state,
            "reply": record.get("reply", ""),
            "created_at": record.get("created_at", 0),
            "created_iso": record.get("created_iso", ""),
            "completed_at": record.get("completed_at"),
            "push_url": record.get("push_url", ""),
            "push_config_id": record.get("push_config_id", ""),
        }
    return snapshot


# ── Durable terminal tombstones ─────────────────────────────────────────
# Bounded terminal retention evicts a terminal row from the ledger window.
# Without durable evidence of that eviction, eviction is indistinguishable
# from "never existed", so a stale second TaskStore can re-add its stale
# WORKING copy during merge and settle the task again with a different
# reply, repeating terminal side effects (PR #92494 review 5252598219,
# blocker 2).  A tombstone records the terminal fact out-of-band so the
# merge contract can tell terminal eviction apart from a genuinely new
# task: an id with a tombstone may only re-publish its identical terminal
# fact (idempotent, no side effects) and can never return to a live state
# or complete with a divergent reply.  The sidecar is bounded
# (_MAX_TOMBSTONES, 4x the terminal window): the reply BODY is never
# stored, only its sha256 fingerprint, which detects divergence exactly
# while keeping the file small.  The caller must hold the ledger lock.
_TOMBSTONES_SUFFIX = ".tombstones.json"
_MAX_TOMBSTONES = 2000


def _tombstone_path_for(ledger_path: Path) -> Path:
    """Sidecar holding eviction tombstones for *ledger_path* (same directory)."""
    return ledger_path.with_name(ledger_path.stem + _TOMBSTONES_SUFFIX)


def _task_tombstones_path() -> Path:
    return _tombstone_path_for(_task_ledger_path())


def _reply_fingerprint(reply: Any) -> str:
    """Stable fingerprint of a reply body for tombstone divergence checks."""
    text = reply if isinstance(reply, str) else str(reply)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tombstone_entry(task_id: str, record: dict) -> dict:
    """Compact durable evidence that *record* was terminal when evicted."""
    return {
        "task_id": record.get("task_id", task_id),
        "state": record.get("state", ""),
        "reply_sha256": _reply_fingerprint(record.get("reply", "")),
        "context_id": record.get("context_id", ""),
        "peer": record.get("peer", ""),
        "agent_slug": record.get("agent_slug", ""),
        "tenant": record.get("tenant", ""),
        "created_at": record.get("created_at", 0),
        "created_iso": record.get("created_iso", ""),
        "completed_at": record.get("completed_at"),
    }


def _load_tombstones(path: Path) -> "tuple[dict, Optional[str]]":
    """Load the tombstone sidecar; returns ``(tombstones, error)``.

    A missing file means "nothing evicted yet" -> ``({}, None)``.  A present
    but unreadable sidecar fails closed with an error: silently forgetting
    tombstones would reopen the resurrection hole.
    """
    if not path.exists():
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except Exception as exc:  # noqa: BLE001 — fail closed on any read/parse error
        return {}, f"tombstone store unreadable: {exc}"
    if not isinstance(loaded, dict):
        return {}, "tombstone store not a dict"
    return {tid: rec for tid, rec in loaded.items() if isinstance(rec, dict)}, None


def _evicted_terminal_tombstones(merged, snapshot, terminal_states) -> dict:
    """Tombstones for terminal records present in *merged* but evicted from *snapshot*."""
    evicted: dict = {}
    for task_id, record in merged.items():
        if record.get("state", "") in terminal_states and task_id not in snapshot:
            evicted[task_id] = _tombstone_entry(task_id, record)
    return evicted


def _write_tombstones(path: Path, tombstones: dict, *,
                      max_tombstones: int = _MAX_TOMBSTONES) -> None:
    """Atomically persist *tombstones*, keeping the newest ``max_tombstones``
    by completion time then insertion order (mirrors the ledger retention
    rule).  Unique temp file, 0o600, fsync, atomic replace."""
    ranked: list = []
    for index, (task_id, entry) in enumerate(tombstones.items()):
        timestamp = entry.get("completed_at")
        try:
            sortable_timestamp = float(timestamp)
        except (TypeError, ValueError):
            sortable_timestamp = 0.0
        ranked.append((sortable_timestamp, index, task_id))
    ranked.sort()
    kept = {task_id for _ts, _idx, task_id in ranked[-max_tombstones:]}
    bounded = {tid: rec for tid, rec in tombstones.items() if tid in kept}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        try:
            os.fchmod(tmp_fd, 0o600)
        except Exception:
            pass
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            tmp_fd = None
            json.dump(bounded, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(tmp_path, str(path))
    except BaseException:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _tombstone_gate(tombstones, task_id, candidate_state, candidate_reply,
                    candidate_context, candidate_peer, candidate_slug, candidate_tenant):
    """Terminal authority for an evicted id (review 5252598219, blocker 2).

    Returns ``("idempotent", record)`` when the candidate re-publishes the
    identical terminal fact, ``("error", {...})`` for ownership mismatches or
    any divergent/non-identical candidate, or ``("none", {})`` when the id has
    no tombstone (a genuinely new task; the caller proceeds with the normal
    merge).  The reply BODY is not retained — only its sha256 fingerprint —
    so divergence is detected exactly while the sidecar stays bounded.
    """
    tombstone = tombstones.get(task_id)
    if tombstone is None:
        return "none", {}
    for field_name, cand_val in (("context_id", candidate_context), ("peer", candidate_peer),
                                 ("agent_slug", candidate_slug), ("tenant", candidate_tenant)):
        if cand_val is not None and cand_val != tombstone.get(field_name, ""):
            return "error", {
                "durable_state": tombstone.get("state", ""),
                "error": f"ownership mismatch for {field_name}: {cand_val!r} != {tombstone.get(field_name, '')!r}",
            }
    tombstone_state = tombstone.get("state", "")
    if candidate_state == tombstone_state and _reply_fingerprint(candidate_reply) == tombstone.get("reply_sha256", ""):
        return "idempotent", {
            "task_id": tombstone.get("task_id", task_id),
            "context_id": tombstone.get("context_id", ""),
            "peer": tombstone.get("peer", ""),
            "agent_slug": tombstone.get("agent_slug", ""),
            "tenant": tombstone.get("tenant", ""),
            "state": tombstone_state,
            "reply": "",  # bounded tombstones keep the fingerprint, not the body
            "created_at": tombstone.get("created_at", 0),
            "created_iso": tombstone.get("created_iso", ""),
            "completed_at": tombstone.get("completed_at"),
            "push_url": "",
            "push_config_id": "",
        }
    return "error", {
        "durable_state": tombstone_state,
        "error": "terminal conflict: task tombstoned after terminal eviction",
    }


def _apply_tombstone_evictions(tombstone_path, new_tombstones, prior_tombstones=None,
                               max_tombstones=_MAX_TOMBSTONES):
    """Merge *new_tombstones* into the sidecar (loading it first when
    *prior_tombstones* is None).  Returns an error string, or None on
    success; never raises."""
    if not new_tombstones:
        return None
    if prior_tombstones is None:
        prior_tombstones, error = _load_tombstones(tombstone_path)
        if error is not None:
            return error
    updated = dict(prior_tombstones)
    updated.update(new_tombstones)
    try:
        _write_tombstones(tombstone_path, updated, max_tombstones=max_tombstones)
    except Exception as exc:
        return f"tombstone write failed: {exc}"
    return None


# ── File locking ────────────────────────────────────────────────────────

@contextlib.contextmanager
def _file_lock(lock_path: Path):
    """Portable advisory file lock: serialises concurrent load→merge→write
    transactions across threads AND processes.

    On Unix, ``fcntl.flock(LOCK_EX)`` blocks until the lock is acquired.
    On Windows, ``msvcrt.locking(LOCK_EX)`` retries with back-off until
    the lock is acquired.
    On platforms with neither (fallback), a threading lock provides
    within-process serialisation only.
    The lock is released when the context manager exits (even on exception).
    """
    if _HAS_FCNTL:
        yield from _file_lock_fcntl(lock_path)
    elif _HAS_MSVCRT:
        yield from _file_lock_msvcrt(lock_path)
    else:
        yield from _file_lock_thread_fallback(lock_path)


def _file_lock_fcntl(lock_path: Path):
    """Unix file lock via fcntl.flock(2)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


def _file_lock_msvcrt(lock_path: Path):
    """Windows file lock via msvcrt.locking.

    msvcrt.locking() locks exactly one byte (at offset 0).  It raises
    OSError on transient contention (errno EACCES or EAGAIN), so we
    retry with exponential-ish back-off up to _MSVCRT_RETRIES times
    before giving up — avoids false negatives when two gateways pulse
    at the same instant.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        for attempt in range(_MSVCRT_RETRIES):
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[union-attr]
                break
            except OSError:
                if attempt == _MSVCRT_RETRIES - 1:
                    raise
                time.sleep(_MSVCRT_RETRY_DELAY)
        yield
    finally:
        if fd is not None:
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[union-attr]
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


def _file_lock_thread_fallback(lock_path: Path):
    """Thread-only fallback for platforms with neither fcntl nor msvcrt.

    Provides within-process serialisation via a threading.Lock.  Does NOT
    protect against concurrent OS processes — acceptable for single-gateway
    deployments where only threads contend on the same data file.
    """
    with _THREAD_FALLBACK_LOCK:
        yield


# ── Context→peer persistence ───────────────────────────────────────────
_CONTEXT_PEERS_FILE = "a2a_context_peers.json"

# ── Context→origin-session persistence ─────────────────────────────────
_CONTEXT_SESSIONS_FILE = "a2a_context_sessions.json"

# ── Fan-out children persistence ────────────────────────────────────────
_FANOUT_CHILDREN_FILE = "a2a_fanout_children.json"


def _fanout_children_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / _FANOUT_CHILDREN_FILE


def _persist_fanout_children(data: dict) -> None:
    """Best-effort write-through of the fan-out children map (atomic)."""
    try:
        path = _fanout_children_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix="a2a_fanout_", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug("A2A: could not persist fan-out children", exc_info=True)


def _load_fanout_children() -> dict:
    """Load the persisted fan-out children map (empty dict on any failure)."""
    try:
        path = _fanout_children_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.debug("A2A: could not load persisted fan-out children", exc_info=True)
    return {}


def _merge_fanout_children(
    existing: Dict[str, Dict[str, str]], extra: Dict[str, Dict[str, str]],
    cap: int = _MAX_CONTEXT_PEERS,
) -> Dict[str, Dict[str, str]]:
    """Merge ``extra`` into ``existing``, bounded by *cap* (default _MAX_CONTEXT_PEERS)."""
    out = dict(existing)
    for parent, children in extra.items():
        if parent in out:
            out.pop(parent, None)
        elif len(out) >= cap:
            out.pop(next(iter(out)), None)
        out[parent] = dict(children)
    return out


_TASK_LEDGER_FILE = "a2a_task_ledger.json"


def _task_ledger_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / _TASK_LEDGER_FILE


def _reset_worker_session_vars() -> None:
    """Reset session-context vars bound on an HTTP worker thread."""
    try:
        from gateway.session_context import reset_session_vars
        reset_session_vars()
    except Exception:
        pass


def _context_peers_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / _CONTEXT_PEERS_FILE


def _persist_context_peers(peers: Dict[str, str]) -> None:
    """Best-effort write-through of the context→peer map to disk (atomic)."""
    try:
        path = _context_peers_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix="a2a_peers_", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(peers, fh, ensure_ascii=False)
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug("A2A: could not persist context peers", exc_info=True)


def _load_context_peers() -> Dict[str, str]:
    """Load the persisted context→peer map (empty dict on any failure)."""
    try:
        path = _context_peers_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
    except Exception:
        logger.debug("A2A: could not load persisted context peers", exc_info=True)
    return {}


def _merge_context_peers(peers: Dict[str, str], extra: Dict[str, str], cap: int = _MAX_CONTEXT_PEERS) -> Dict[str, str]:
    """Merge ``extra`` into ``peers``, bounded by *cap* (default _MAX_CONTEXT_PEERS)."""
    out = dict(peers)
    for cid, peer in extra.items():
        if cid in out:
            out.pop(cid, None)
        elif len(out) >= cap:
            out.pop(next(iter(out)), None)
        out[cid] = peer
    return out


def _context_sessions_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / _CONTEXT_SESSIONS_FILE


def _persist_context_sessions(sessions: Dict[str, dict]) -> None:
    """Best-effort write-through of the context→origin-session map (atomic)."""
    try:
        path = _context_sessions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix="a2a_sessions_", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(sessions, fh, ensure_ascii=False)
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug("A2A: could not persist context sessions", exc_info=True)


def _load_context_sessions() -> Dict[str, dict]:
    """Load the persisted context→origin-session map (empty on any failure)."""
    try:
        path = _context_sessions_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            out: Dict[str, dict] = {}
            for k, v in data.items():
                if isinstance(v, dict) and v.get("platform"):
                    out[str(k)] = v
            return out
    except Exception:
        logger.debug("A2A: could not load persisted context sessions", exc_info=True)
    return {}


def _merge_context_sessions(sessions: Dict[str, dict], extra: Dict[str, dict], cap: int = _MAX_CONTEXT_PEERS) -> Dict[str, dict]:
    """Merge ``extra`` into ``sessions``, bounded by *cap* (default _MAX_CONTEXT_PEERS)."""
    out = dict(sessions)
    for cid, origin in extra.items():
        if cid in out:
            out.pop(cid, None)
        elif len(out) >= cap:
            out.pop(next(iter(out)), None)
        out[cid] = origin
    return out


# ── URL / identity helpers ──────────────────────────────────────────────

_LOOPBACK_ADDRS = {"127.0.0.1", "localhost", "::1"}


def _is_ipv6_literal(host: str) -> bool:
    """Return True when *host* is an IPv6 address literal (e.g. ``::1``)."""
    return ":" in host


def _bracket_ipv6(host: str) -> str:
    """Bracket an IPv6 literal for inclusion in a URL, if needed."""
    return f"[{host}]" if _is_ipv6_literal(host) else host


def _own_a2a_url(host: str, port: int) -> str:
    """Build this gateway's own A2A endpoint URL (the one peers push to)."""
    bind_host = host or "127.0.0.1"
    if bind_host in ("0.0.0.0", "::", ""):
        bind_host = "127.0.0.1"
    return f"http://{_bracket_ipv6(bind_host)}:{int(port or _DEFAULT_PORT)}"


def _sender_url_acceptable(url: str, peers_cfg: dict) -> bool:
    """Whether a message ``sender.url`` may be trusted as a push target."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in _LOOPBACK_ADDRS or host.startswith("127."):
        return True
    for entry in peers_cfg.values():
        if not isinstance(entry, dict):
            continue
        try:
            eu = urllib.parse.urlparse(str(entry.get("url") or ""))
        except Exception:
            continue
        if eu.hostname and eu.hostname.lower() == host:
            return True
    return False


def _is_own_endpoint(url: str, host: str, port: int) -> bool:
    """Whether ``url`` points at this gateway's own A2A endpoint."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    if hostname not in _LOOPBACK_ADDRS and not hostname.startswith("127."):
        return False
    try:
        peer_port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    except (ValueError, TypeError):
        return False
    return peer_port == int(port or _DEFAULT_PORT)


def _loopback_fallback_url(identity: str, host: str, port: int) -> str:
    """Return this gateway's own A2A URL when ``identity`` is a loopback ``ip:`` identity."""
    if not identity.startswith("ip:"):
        return ""
    addr = identity[3:].strip().lower()
    if addr not in _LOOPBACK_ADDRS and not addr.startswith("127."):
        return ""
    bind_host = host or "127.0.0.1"
    if bind_host in ("0.0.0.0", "::", ""):
        bind_host = "127.0.0.1"
    return f"http://{_bracket_ipv6(bind_host)}:{int(port or _DEFAULT_PORT)}"


def _reply_timeout() -> float:
    """Seconds to wait for the agent to answer an inbound task."""
    try:
        return max(1.0, float(os.getenv("A2A_REPLY_TIMEOUT", "300")))
    except (ValueError, TypeError):
        return 300.0


def _profile_scoped() -> bool:
    """True when running inside a multiplexed secondary profile's scope."""
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active
        return bool(is_multiplex_active() and current_secret_scope() is not None)
    except Exception:
        return False


def _default_agent_name() -> str:
    name = "" if _profile_scoped() else os.getenv("A2A_AGENT_NAME", "").strip()
    if name:
        return name
    try:
        import socket
        return f"hermes-{socket.gethostname()}"
    except Exception:
        return "hermes-agent"


def _clean_slug(value: str) -> str:
    """Return a URL-safe-ish single-segment slug for a served agent."""
    slug = str(value or "").strip().strip("/")
    return "" if slug in ("", "default", "root") else slug.split("/")[0]


def _join_url(base: str, prefix: str) -> str:
    base = (base or "").strip() or "/"
    if not base.endswith("/"):
        base += "/"
    prefix = (prefix or "").strip("/")
    if not prefix:
        return base
    return urllib.parse.urljoin(base, prefix + "/")


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return os.getenv("HERMES_PROFILE", "default") or "default"


def _profile_home(profile: str) -> Optional[str]:
    try:
        from hermes_cli.profiles import get_profile_dir
        return str(get_profile_dir(profile))
    except Exception:
        if not profile or profile == "default":
            try:
                from hermes_cli.config import get_hermes_home
                return str(get_hermes_home())
            except Exception:
                return None
        return os.path.expanduser(f"~/.hermes/profiles/{profile}")


def _safe_context_slug(value: str, max_len: int = 96) -> str:
    """Sanitize attacker-provided context ids before using in session titles."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (slug or "ctx")[:max_len]
