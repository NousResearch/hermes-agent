"""Session mirroring for cross-platform message delivery.

When a message is sent to a platform (send_message or cron delivery), append a
"delivery-mirror" record to the target session's transcript so the receiving-side
agent knows what was sent.  Standalone: works from CLI, cron and gateway contexts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"


class OwnershipResolutionError(RuntimeError):
    """Raised when routing metadata cannot prove exactly one transcript owner."""


def _origin_user_id(entry: dict) -> str:
    return str((entry.get("origin") or {}).get("user_id") or "")


def _entry_session_key(key: str, entry: dict) -> str:
    """Canonical owner key, rejecting contradictory routing metadata."""
    key = str(key)
    embedded = str(entry.get("session_key") or "")
    if key.startswith("agent:"):
        if embedded and (not embedded.startswith("agent:") or embedded != key):
            raise OwnershipResolutionError(
                f"routing key {key!r} conflicts with embedded session key {embedded!r}")
        resolved = key
    elif embedded.startswith("agent:"):
        resolved = embedded
    else:
        profile = str((entry.get("origin") or {}).get("profile") or "").strip()
        if not profile:
            return ""
        namespace = "main" if profile == "default" else ("main~" if profile == "main" else profile)
        resolved = f"agent:{namespace}"

    parts = resolved.split(":")
    if len(parts) < 4 or parts[0] != "agent" or not parts[2].strip() or not parts[3].strip():
        raise OwnershipResolutionError(f"routing key {resolved!r} is not a canonical session key")

    from hermes_cli.profiles import get_profile_dir, normalize_profile_name

    namespace = parts[1]
    if namespace not in {"main", "main~"}:
        try:
            normalized_namespace = normalize_profile_name(namespace)
            get_profile_dir(normalized_namespace)
        except ValueError as exc:
            raise OwnershipResolutionError(
                f"routing key {resolved!r} has an invalid profile namespace") from exc
        if normalized_namespace != namespace or normalized_namespace in {"default", "main"}:
            raise OwnershipResolutionError(
                f"routing key {resolved!r} has a non-canonical profile namespace")

    origin_profile = str((entry.get("origin") or {}).get("profile") or "").strip()
    if origin_profile:
        routed_profile = "default" if namespace == "main" else ("main" if namespace == "main~" else namespace)
        if normalize_profile_name(origin_profile) != normalize_profile_name(routed_profile):
            raise OwnershipResolutionError(
                f"routing key {resolved!r} conflicts with origin profile {origin_profile!r}")
    return resolved


def _routing_db_path() -> Path:
    """The launch home's DB, which owns the cross-profile routing index."""
    return _SESSIONS_DIR.parent / "state.db"


def _routing_entries_from_db() -> List[Tuple[str, dict]]:
    """Read the primary routing index without assuming its sessions live in this DB."""
    from hermes_state_registry import acquire, release_or_close

    db = acquire(_routing_db_path())
    try:
        loader = getattr(db, "load_gateway_routing_entries", None)
        if not callable(loader):
            raise OwnershipResolutionError("routing DB does not expose its route index")
        rows = loader(scope=str(_SESSIONS_DIR.resolve()))
        if not isinstance(rows, dict):
            raise OwnershipResolutionError("routing DB index is not an object")
        entries = []
        for key, payload in rows.items():
            try:
                entry = json.loads(payload)
            except (TypeError, ValueError) as exc:
                raise OwnershipResolutionError(
                    f"routing DB entry {key!r} is not valid JSON") from exc
            if not isinstance(entry, dict):
                raise OwnershipResolutionError(
                    f"routing DB entry {key!r} is not an object")
            entries.append((str(key), entry))
        return entries
    finally:
        release_or_close(db)


def _routing_entries_from_json() -> List[Tuple[str, dict]]:
    """Read the pre-migration routing mirror, if enabled."""
    if not _SESSIONS_INDEX.exists():
        return []
    try:
        data = json.loads(_SESSIONS_INDEX.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OwnershipResolutionError("sessions.json routing index is unreadable") from exc
    if not isinstance(data, dict):
        raise OwnershipResolutionError("sessions.json routing index is not an object")
    entries = []
    for key, entry in data.items():
        if str(key).startswith("_"):
            continue
        if not isinstance(entry, dict):
            raise OwnershipResolutionError(
                f"sessions.json routing entry {key!r} is not an object")
        entries.append((str(key), entry))
    return entries


def _select_origin_target(
    entries: List[Tuple[str, dict]], platform: str, chat_id: str,
    thread_id: Optional[str], user_id: Optional[str],
) -> Optional[Tuple[str, str]]:
    """Select ``(session_id, session_key)`` with the existing contamination guards."""
    def _matches(item: Tuple[str, dict]) -> bool:
        _, entry = item
        origin = entry.get("origin") or {}
        return ((origin.get("platform") or entry.get("platform", "")).lower() == platform.lower()
                and str(origin.get("chat_id", "")) == str(chat_id)
                and (thread_id is None or str(origin.get("thread_id") or "") == str(thread_id)))

    candidates = [item for item in entries if _matches(item)]
    if not candidates:
        return None
    if user_id:
        exact = [item for item in candidates if _origin_user_id(item[1]) == str(user_id)]
        if exact:
            candidates = exact
        elif any(_origin_user_id(item[1]).strip() for item in candidates):
            raise OwnershipResolutionError(
                f"routed sessions match {platform}:{chat_id} but none own user {user_id!r}")
    elif len(candidates) > 1 and len({
        user.strip() for user in (_origin_user_id(item[1]) for item in candidates) if user.strip()
    }) > 1:
        raise OwnershipResolutionError(
            f"multiple routed users match {platform}:{chat_id}; user ownership is required")
    resolved_candidates = []
    owner_namespaces = set()
    for key, entry in candidates:
        session_id = entry.get("session_id")
        if not session_id:
            raise OwnershipResolutionError(
                f"matching route {key!r} has no session_id")
        session_key = _entry_session_key(key, entry)
        if not session_key:
            raise OwnershipResolutionError(
                f"session {session_id!r} has no profile ownership metadata")
        owner_namespaces.add(session_key.split(":", 2)[1])
        resolved_candidates.append((key, entry, str(session_id), session_key))
    if len(owner_namespaces) > 1:
        raise OwnershipResolutionError(
            f"matching routes for {platform}:{chat_id} have conflicting profile owners")

    _, _, session_id, session_key = max(
        resolved_candidates, key=lambda item: item[1].get("updated_at", ""))
    return session_id, session_key


def _db_path_for_session_id(session_id: str) -> Optional[Path]:
    """Resolve one explicit session ID to exactly one proven owner database."""
    owner_paths = set()
    matched = False
    for loader in (_routing_entries_from_db, _routing_entries_from_json):
        try:
            matches = [
                _entry_session_key(key, entry)
                for key, entry in loader()
                if str(entry.get("session_id") or "") == str(session_id)
            ]
        except OwnershipResolutionError:
            raise
        except Exception as exc:
            raise OwnershipResolutionError("routing index is unavailable") from exc
        if matches:
            matched = True
            for owner in matches:
                if owner:
                    db_path = _db_path_for_session_key(owner)
                    if db_path is None:
                        raise OwnershipResolutionError(
                            f"session {session_id!r} has no resolvable profile owner")
                    owner_paths.add(db_path.resolve())
        if matches and not all(matches):
            raise OwnershipResolutionError(
                f"session {session_id!r} has no profile ownership metadata")
    if len(owner_paths) > 1:
        raise OwnershipResolutionError(
            f"session {session_id!r} has conflicting profile owners")
    if len(owner_paths) == 1:
        return owner_paths.pop()
    if matched:
        raise OwnershipResolutionError(
            f"session {session_id!r} has no profile ownership metadata")
    return None


def _db_path_for_session_key(session_key: Optional[str]) -> Optional[Path]:
    """Resolve the transcript DB encoded by a gateway routing key."""
    if not session_key:
        return None
    parts = str(session_key).split(":")
    if len(parts) < 2 or parts[0] != "agent":
        return None
    from gateway.session import profile_from_session_key_namespace

    profile = profile_from_session_key_namespace(parts[1] or "main")
    if profile == "default":
        return _routing_db_path()
    from hermes_cli.profiles import get_profile_dir, profile_exists

    if not profile_exists(profile):
        raise RuntimeError(f"profile {profile!r} from routing key has no resolvable home")
    return Path(get_profile_dir(profile)) / "state.db"


def mirror_to_session(
    platform: str, chat_id: str, message_text: str, source_label: str = "cli", thread_id: Optional[str] = None,
    user_id: Optional[str] = None, role: str = "assistant", session_id: Optional[str] = None,
) -> bool:
    """Append a delivery-mirror message to the target session's SQLite transcript.

    Pass ``session_id`` when the caller already holds the exact session (e.g. the
    cron in_channel seed) to skip the origin scan, which refuses to guess on a
    populated chat (flat + N thread sessions per chat_id) and would drop the mirror.
    Text that is NOT the agent speaking (e.g. a cron brief) must pass
    ``role="user"``: ``mirror`` metadata is dropped at the SQLite boundary, so an
    assistant-role mirror replays as a real turn and yields assistant→assistant
    pairs that break strict-alternation providers, while a user-role mirror
    collapses safely via the consecutive-user merge.
    Returns True if mirrored, False if no matching session or error. Never raises.

    ``role`` defaults to ``"assistant"`` — correct for the interactive ``send_message`` mirror, where the
    mirrored text is the agent's own outgoing reply (a genuine assistant turn). See #2221.
    """
    try:
        db_path = None
        if not session_id:
            target = _find_session_target(
                platform, str(chat_id), thread_id=thread_id, user_id=user_id)
            if target:
                session_id, db_path = target
        elif session_id:
            db_path = _db_path_for_session_id(session_id)
        if not session_id:
            logger.warning(
                "Mirror: no session found for %s:%s thread=%s user=%s (explicit_id=none, origin-scan bailed)",
                platform, chat_id, thread_id, user_id,
            )
            return False
        if db_path is None:
            logger.warning(
                "Mirror: no proven transcript owner for session %s; refusing ambient fallback",
                session_id,
            )
            return False
        _append_to_sqlite(session_id, {
            "role": role, "content": message_text, "timestamp": datetime.now().isoformat(),
            "mirror": True, "mirror_source": source_label,
        }, db_path=db_path)
        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True
    except Exception as e:
        # WARNING, not debug: a silent mirror drop is the cron continuation-amnesia bug.
        logger.warning("Mirror failed for %s:%s thread=%s user=%s session=%s: %s", platform, chat_id, thread_id, user_id, session_id, e)
        return False


def _find_session_target(
    platform: str, chat_id: str, thread_id: Optional[str] = None, user_id: Optional[str] = None,
) -> Optional[Tuple[str, Path]]:
    """Active session ID and proven owner database for a platform/chat origin.

    Both routed ownership sources must be readable before legacy root lookup is
    allowed. A routed match always outranks the legacy finder, and disagreement
    between route sources fails closed.
    """
    try:
        db_entries = _routing_entries_from_db()
    except OwnershipResolutionError:
        raise
    except Exception as exc:
        raise OwnershipResolutionError("routing DB index is unavailable") from exc
    try:
        json_entries = _routing_entries_from_json()
    except OwnershipResolutionError:
        raise
    except Exception as exc:
        raise OwnershipResolutionError("sessions.json routing index is unavailable") from exc

    routed_targets = []
    for source, entries in (("state.db", db_entries), ("sessions.json", json_entries)):
        target = _select_origin_target(entries, platform, chat_id, thread_id, user_id)
        if not target:
            continue
        session_id, session_key = target
        db_path = _db_path_for_session_key(session_key)
        if db_path is None:
            raise OwnershipResolutionError(
                f"{source} session {session_id!r} has no resolvable profile owner")
        routed_targets.append((str(session_id), db_path.resolve()))

    if routed_targets:
        if len(set(routed_targets)) > 1:
            raise OwnershipResolutionError(
                f"routing sources disagree for {platform}:{chat_id}")
        return routed_targets[0]

    try:
        from hermes_state_registry import acquire, release_or_close
        routing_db_path = _routing_db_path()
        db = acquire(routing_db_path)
        try:
            finder = getattr(db, "find_session_by_origin", None)
            session_id = finder(
                platform=platform, chat_id=chat_id, thread_id=thread_id, user_id=user_id,
            ) if callable(finder) else None
            if session_id:
                return str(session_id), routing_db_path
        finally:
            release_or_close(db)
    except Exception as e:
        logger.debug("Mirror state.db session lookup failed: %s", e)
    return None


def _find_session_id(
    platform: str, chat_id: str, thread_id: Optional[str] = None, user_id: Optional[str] = None,
) -> Optional[str]:
    """Compatibility wrapper returning only the active session ID."""
    target = _find_session_target(platform, chat_id, thread_id=thread_id, user_id=user_id)
    return target[0] if target else None


def _append_to_sqlite(session_id: str, message: dict, db_path: Path) -> None:
    """Append a message to a transcript database with proven ownership.

    Raises on failure: ``mirror_to_session`` reports ``False`` (and warns) only when the
    exception reaches it — swallowing it here made every failed write look mirrored (#10130).
    """
    from hermes_state_registry import acquire, release_or_close

    db = acquire(db_path)
    try:
        db.append_message(
            session_id=session_id, role=message.get("role", "assistant"),
            content=str(message.get("content") or ""),
        )
    finally:
        release_or_close(db)
