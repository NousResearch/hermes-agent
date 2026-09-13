"""One-time upload of local memory files (MEMORY.md / USER.md / SOUL.md) into Honcho."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from plugins.memory.honcho.client import resolve_effective_base_url
from plugins.memory.honcho.session_auth import HonchoAuthError
from utils import atomic_json_write

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

logger = logging.getLogger("plugins.memory.honcho.session")

_MIGRATION_LOCK_TIMEOUT_SECONDS = 5.0
_MIGRATION_THREAD_LOCK = threading.Lock()
_HONCHO_ENVIRONMENT_ENDPOINTS = {
    "local": "http://localhost:8000",
    "production": "https://api.honcho.dev",
}

# (filename, upload name, description, peer kind) — peer kind picks user vs assistant peer.
_MEMORY_FILES = (
    ("MEMORY.md", "consolidated_memory.md", "Long-term agent notes and preferences", "user"),
    ("USER.md", "user_profile.md", "User profile and preferences", "user"),
    ("SOUL.md", "agent_soul.md", "Agent persona and identity configuration", "ai"),
)


def _acquire_migration_file_lock(lock_path: Path):
    """Acquire a portable non-blocking file lock before the deadline."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    if msvcrt is not None:
        lock_file.seek(0, 2)
        if lock_file.tell() == 0:
            lock_file.write(" ")
            lock_file.flush()

    deadline = time.monotonic() + _MIGRATION_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - supported platforms provide one primitive
                logger.warning("Honcho migration skipped: no file-lock primitive available")
                lock_file.close()
                return None
            return lock_file
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                logger.warning("Honcho migration lock failed: %s", exc)
                lock_file.close()
                return None
            if time.monotonic() >= deadline:
                logger.warning("Honcho migration skipped: timed out acquiring %s", lock_path)
                lock_file.close()
                return None
            time.sleep(0.05)


def _release_migration_file_lock(lock_file) -> None:
    """Release a lock acquired by :func:`_acquire_migration_file_lock`."""
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError as exc:
        logger.warning("Failed to release Honcho migration lock: %s", exc)
    finally:
        lock_file.close()


@contextmanager
def _migration_lock(lock_path: Path):
    """Bound concurrent migration attempts across threads and processes."""
    if not _MIGRATION_THREAD_LOCK.acquire(timeout=_MIGRATION_LOCK_TIMEOUT_SECONDS):
        logger.warning("Honcho migration skipped: timed out acquiring thread lock")
        yield False
        return

    lock_file = None
    try:
        try:
            lock_file = _acquire_migration_file_lock(lock_path)
        except OSError as exc:
            logger.warning("Honcho migration lock failed: %s", exc)
        if lock_file is None:
            yield False
            return
        yield True
    finally:
        if lock_file is not None:
            _release_migration_file_lock(lock_file)
        _MIGRATION_THREAD_LOCK.release()


def _load_migration_state(state_path: Path) -> dict[str, Any] | None:
    """Load valid migration state, failing closed on unreadable state."""
    if not state_path.exists():
        return {"version": 1, "targets": {}}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Honcho migration state is unreadable: %s", exc)
        return None
    if (
        not isinstance(state, dict)
        or state.get("version") != 1
        or not isinstance(state.get("targets"), dict)
    ):
        logger.warning("Honcho migration state has an unsupported format: %s", state_path)
        return None
    return state


class SessionMigrationMixin:
    def migrate_memory_files(self, session_key: str, memory_dir: str) -> bool:
        """Upload local memory files once per source and Honcho destination.

        A local ledger prevents repeat uploads across sessions. Before each upload,
        a deterministic remote migration marker is reconciled so a successful
        upload followed by a failed ledger write cannot produce a duplicate.
        """
        memory_path = Path(memory_dir).expanduser().resolve()
        if not memory_path.exists():
            return False

        session = self._cache.get(session_key)
        if not session:
            logger.warning("No local session cached for '%s', skipping memory migration", session_key)
            return False
        if session.honcho_session_id not in self._sessions_cache:
            logger.warning("No Honcho session cached for '%s', skipping memory migration", session_key)
            return False

        # Owner-scoped: these files describe the install owner; uploading them under another
        # human's peer would make Honcho attribute the owner's facts to that person. The owner is
        # the CONFIG peerName — never a re-resolution of the session's own peer (that would compare
        # the triggering user to themselves). No declared owner: single-operator only when there is
        # no runtime identity; with one, nobody can be proven to be the owner.
        owner_peer_id = self._declared_owner_peer_id()
        session_is_owner = (
            session.user_peer_id == owner_peer_id
            if owner_peer_id is not None
            else not self._runtime_user_ids()
        )
        if not session_is_owner:
            logger.info(
                "Skipping memory-file migration: session user peer '%s' is not the declared owner (peerName=%s)",
                session.user_peer_id,
                owner_peer_id or "unset",
            )
            return False

        config = self._config
        endpoint = resolve_effective_base_url(config) if config is not None else None
        environment = config.environment if config is not None else "production"
        endpoint_key = endpoint or _HONCHO_ENVIRONMENT_ENDPOINTS.get(
            environment,
            f"environment:{environment}",
        )
        workspace_id = config.workspace_id if config is not None else "hermes"
        target = {
            "endpoint": endpoint_key,
            "workspace_id": workspace_id,
            "user_peer_id": session.user_peer_id,
            "ai_peer_id": session.assistant_peer_id,
        }
        target_json = json.dumps(target, sort_keys=True, separators=(",", ":"))
        target_key = hashlib.sha256(target_json.encode("utf-8")).hexdigest()
        remote_marker_session_id = f"hermes-memory-migration-{target_key}"
        source_path = str(memory_path)
        source_key = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
        hermes_home = config.hermes_home if config is not None and config.hermes_home is not None else get_hermes_home()
        state_path = hermes_home / "state" / "honcho_migration.json"
        lock_path = state_path.with_suffix(".json.lock")

        with _migration_lock(lock_path) as acquired:
            if not acquired:
                return False
            state = _load_migration_state(state_path)
            if state is None:
                return False

            target_state = state["targets"].setdefault(
                target_key,
                {**target, "sources": {}},
            )
            if not isinstance(target_state, dict):
                logger.warning("Honcho migration target state is invalid: %s", target_key)
                return False
            sources = target_state.setdefault("sources", {})
            if not isinstance(sources, dict):
                logger.warning("Honcho migration target state is invalid: %s", target_key)
                return False
            source_state = sources.setdefault(
                source_key,
                {"path": source_path, "files": {}},
            )
            if not isinstance(source_state, dict):
                logger.warning("Honcho migration source state is invalid: %s", source_path)
                return False
            completed_files = source_state.get("files")
            if not isinstance(completed_files, dict):
                logger.warning("Honcho migration source state is invalid: %s", source_path)
                return False

            uploaded = False
            for filename, upload_name, description, target_kind in _MEMORY_FILES:
                if completed_files.get(filename) is True:
                    continue
                filepath = memory_path / filename
                content = filepath.read_text(encoding="utf-8").strip() if filepath.exists() else ""
                if not content:
                    continue

                migration_id = hashlib.sha256(
                    json.dumps(
                        {
                            "version": 1,
                            "target": target_key,
                            "source": source_key,
                            "filename": filename,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                migration_filter = {"metadata": {"migration_id": migration_id}}

                try:
                    from honcho.session import SessionPeerConfig

                    self._authed_call(
                        "memory migration marker session setup",
                        lambda: self._sdk_session(
                            remote_marker_session_id,
                            metadata={
                                "source": "hermes_memory_migration",
                                "target_key": target_key,
                            },
                            peers=[
                                (
                                    session.user_peer_id,
                                    SessionPeerConfig(observe_me=True, observe_others=False),
                                ),
                                (
                                    session.assistant_peer_id,
                                    SessionPeerConfig(observe_me=True, observe_others=False),
                                ),
                            ],
                        ),
                    )
                    remote_messages = self._authed_call(
                        "memory migration reconciliation",
                        lambda: self._sdk_session(remote_marker_session_id).messages(
                            filters=migration_filter,
                            page=1,
                            size=1,
                        ),
                    )
                except HonchoAuthError:
                    logger.error("Honcho memory migration stopped before %s: auth failed", filename)
                    break
                except Exception as exc:
                    logger.warning(
                        "Failed to reconcile Honcho migration for %s; skipping upload: %s",
                        filename,
                        exc,
                    )
                    continue

                if len(remote_messages) > 0:
                    completed_files[filename] = True
                    try:
                        atomic_json_write(state_path, state, mode=0o600, sort_keys=True)
                    except Exception as exc:
                        logger.warning(
                            "Failed to record reconciled Honcho migration for %s: %s",
                            filename,
                            exc,
                        )
                        break
                    logger.info("Reconciled prior Honcho migration for %s", filename)
                    continue

                wrapped = (
                    "<prior_memory_file>\n<context>\n"
                    "This file was consolidated from local conversations BEFORE Honcho was activated.\n"
                    f"{description}. Treat as foundational context for this user.\n"
                    f"</context>\n\n{content}\n</prior_memory_file>\n"
                )
                target_peer_id = (
                    session.user_peer_id if target_kind == "user" else session.assistant_peer_id
                )

                try:

                    def _upload() -> None:
                        self._sdk_session(remote_marker_session_id).upload_file(
                            file=(upload_name, wrapped.encode("utf-8"), "text/plain"),
                            peer=self._get_or_create_peer(target_peer_id),
                            metadata={
                                "source": "local_memory",
                                "original_file": filename,
                                "target_peer": target_kind,
                                "migration_id": migration_id,
                            },
                        )

                    self._authed_call("memory migration upload", _upload)
                except HonchoAuthError:
                    logger.error("Honcho memory migration stopped after %s: auth failed", filename)
                    break
                except Exception as exc:
                    logger.error("Failed to upload %s to Honcho: %s", filename, exc)
                    continue

                uploaded = True
                completed_files[filename] = True
                try:
                    atomic_json_write(state_path, state, mode=0o600, sort_keys=True)
                except Exception as exc:
                    logger.warning("Failed to record Honcho migration for %s: %s", filename, exc)
                    break
                logger.info(
                    "Uploaded %s to Honcho for %s (%s peer)",
                    filename,
                    session_key,
                    target_kind,
                )

            return uploaded
