"""HRM-T0a active-topic-pointer routing primitives.

Substrate for the same-chat active-topic-pointer model defined in the
``topic-real-session-routing-charter``. This module provides the read /
set helpers, the per-key in-process asyncio lock primitive, the typed
:class:`PlatformPrincipal` envelope, and the ``assert_registered``
adapter for the dev-os topic registry.

Scope:

- The pointer's persistent state lives in SessionDB (the existing
  SQLite). This module is the in-process wrapper that pairs each
  pointer read / write with a per-key asyncio lock so the slash-write
  and the next inbound-message read take place inside the same
  serialised window (charter invariant 1).

- :class:`PlatformPrincipal` is the typed envelope every routed turn
  carries. ``app_id`` is included so a single principal can hold
  concurrent topics in different repos without cross-talk.

- The lock dict is bounded by both a TTL (idle eviction) and a hard
  size cap (LRU eviction). Eviction never drops a held lock — the
  guard checks the lock's internal state before removing it. A
  re-creation of an evicted entry is correctness-safe because the
  underlying SessionDB ``BEGIN IMMEDIATE`` transaction remains the
  serialisation point of truth.

- :func:`assert_registered` is an adapter to the dev-os in-tree topic
  registry. In v1 it accepts a callable injected at process startup
  (``set_registered_check``); if no checker is wired, the adapter
  fails closed (refuses the switch) so the charter contract holds
  even on a partial deployment.

What this module does NOT do:

- Change ``/topic`` slash behaviour (step 5 — owner-gated).
- Add API server endpoints (step 6 — owner-gated).

Step 4 additions (this commit):

- :func:`build_topic_session_key` deterministically constructs the topic-
  routed session key from ``(principal, topic_id)``.
- :func:`resolve_topic_session_key` is the **inbound routing pre-pass**:
  reads the pointer row + (when wired) confirms the topic is registered,
  then returns the topic-routed session key. When no pointer exists or
  the registry check fails, returns ``None`` so the caller falls through
  to the legacy ``build_session_key`` path. Fails closed when no checker
  is wired: the routing flip is refused and the message keeps its legacy
  thread-derived session, so a partial deployment can't silently route
  to an unregistered topic.
- :func:`resolve_topic_session_key_async` is the locked variant. The lock
  window covers the pointer read and the route decision only — it is
  released **before** the agent's full response is generated. This is
  the explicit clarification of Critic finding #1: the lock is a route-
  serialisation primitive, not an agent-generation barrier.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


PointerKey = Tuple[str, str, str, str]  # (platform, user_id, chat_id, app_id)


# ── PlatformPrincipal envelope ────────────────────────────────────────


@dataclass(frozen=True)
class PlatformPrincipal:
    """Stable identity stamped onto every routed turn.

    Charter contract (`§ Platform principal envelope`): every routed
    turn carries this envelope; every downstream send/move/persist
    must byte-equal the stamped envelope, else the send is dropped
    and a leak alert is raised. Frozen so callers can't mutate a
    stamped envelope after the fact (immutability is the leak guard).

    ``platform`` is a string (``"telegram"``, ``"discord"``, …) — NOT
    a :class:`gateway.session.Platform` enum — so this module can be
    imported anywhere without dragging the Platform enum into
    foreign codebases (e.g. ``hermes_state``). Conversion happens at
    the boundary.
    """

    platform: str
    user_id: str
    chat_id: str
    app_id: str

    @property
    def key(self) -> PointerKey:
        return (self.platform, self.user_id, self.chat_id, self.app_id)

    @classmethod
    def from_source(
        cls,
        source: Any,
        *,
        app_id: str,
    ) -> "PlatformPrincipal":
        """Build a principal from a ``SessionSource``-shaped object.

        ``source`` is duck-typed (``.platform``, ``.user_id``, ``.chat_id``)
        so this module does not need to import the gateway's
        ``SessionSource`` class. ``app_id`` is supplied by the caller
        because its derivation lives at the dev-os boundary, not in
        this module (see charter § Cross-repo execution discipline
        and residual-risk note in the plan).
        """
        platform = getattr(source, "platform", None)
        if platform is None:
            raise ValueError("source.platform is required")
        # SessionSource.platform is a Platform enum with .value; accept
        # both shapes so this helper works in unit tests with raw strings.
        platform_str = getattr(platform, "value", None) or str(platform)

        user_id = getattr(source, "user_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not user_id:
            raise ValueError("source.user_id is required for principal envelope")
        if not chat_id:
            raise ValueError("source.chat_id is required for principal envelope")
        if not app_id:
            raise ValueError("app_id is required for principal envelope")
        return cls(
            platform=str(platform_str),
            user_id=str(user_id),
            chat_id=str(chat_id),
            app_id=str(app_id),
        )


class PlatformPrincipalLeak(Exception):
    """Raised when a queued turn's envelope differs from the routed surface.

    Charter contract: the send is DROPPED — never degraded to
    best-effort delivery. The exception carries both envelopes so an
    audit log line can diff them.
    """

    def __init__(self, *, expected: PlatformPrincipal, actual: PlatformPrincipal):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"PlatformPrincipalLeak: expected={expected!r} actual={actual!r}"
        )


def assert_principal_match(
    *,
    expected: PlatformPrincipal,
    actual: PlatformPrincipal,
) -> None:
    """Raise :class:`PlatformPrincipalLeak` unless ``expected == actual``.

    Wrapper kept so the call site reads as the contract intent.
    """
    if expected != actual:
        raise PlatformPrincipalLeak(expected=expected, actual=actual)


# ── Per-key asyncio lock primitive ────────────────────────────────────


# Tunable knobs (kept as module globals so tests can override). The
# defaults match § 1.B of the plan.
_LOCK_TTL_SECONDS = 300.0   # idle entries get evicted after this
_LOCK_MAX_ENTRIES = 10_000   # hard cap — LRU evicts past this

# OrderedDict so the LRU eviction is O(1). The value is
# ``(asyncio.Lock, last_touched_unix_ts)``.
_LOCKS: "OrderedDict[PointerKey, Tuple[asyncio.Lock, float]]" = OrderedDict()
_LOCKS_GUARD = asyncio.Lock()


def _evict_idle_locked(now: float) -> None:
    """Drop idle entries (caller must hold ``_LOCKS_GUARD``).

    Never drops a held lock — a locked entry is a guarantee that
    some other coroutine is mid-critical-section, and the entry
    will be re-touched when that coroutine releases.
    """
    cutoff = now - _LOCK_TTL_SECONDS
    stale: list = []
    for key, (lock, last_touched) in _LOCKS.items():
        if lock.locked():
            continue
        if last_touched < cutoff:
            stale.append(key)
    for key in stale:
        _LOCKS.pop(key, None)


def _enforce_cap_locked() -> None:
    """LRU-evict until ``len(_LOCKS) <= _LOCK_MAX_ENTRIES``.

    Held locks are skipped (correctness > cap). If every entry is
    held the dict can transiently exceed the cap; that's intentional.
    """
    if len(_LOCKS) <= _LOCK_MAX_ENTRIES:
        return
    # Iterate over a snapshot of keys so we can mutate the OrderedDict.
    for key in list(_LOCKS.keys()):
        if len(_LOCKS) <= _LOCK_MAX_ENTRIES:
            return
        lock, _ = _LOCKS[key]
        if lock.locked():
            continue
        _LOCKS.pop(key, None)


async def acquire_pointer_lock(key: PointerKey) -> asyncio.Lock:
    """Return the per-key :class:`asyncio.Lock`, creating it on first use.

    Caller is responsible for ``async with lock: ...``. The lock dict
    is bounded by TTL idle-eviction and a hard size cap (LRU); both
    run under the guard before the lookup so the bookkeeping cost is
    paid on lock acquisition, not on the hot read path.
    """
    now = time.time()
    async with _LOCKS_GUARD:
        _evict_idle_locked(now)
        entry = _LOCKS.get(key)
        if entry is None:
            lock = asyncio.Lock()
            _LOCKS[key] = (lock, now)
        else:
            lock = entry[0]
            # Refresh the LRU position + last-touched timestamp.
            _LOCKS.move_to_end(key)
            _LOCKS[key] = (lock, now)
        _enforce_cap_locked()
    return lock


def _reset_locks_for_tests() -> None:
    """Drop the lock dict — TEST-ONLY helper.

    Production code must never call this: it discards lock identity
    which is correctness-critical. Tests use it between cases so
    one test's lock dict doesn't bleed into another's.
    """
    _LOCKS.clear()


# ── assert_registered adapter ─────────────────────────────────────────


class TopicNotRegisteredError(Exception):
    """Raised when ``assert_registered`` rejects ``(app_id, topic_id)``."""


# Injection point: the dev-os boundary owns the actual registry. In v1
# the gateway calls ``set_registered_check(fn)`` at startup to supply a
# checker; until that happens, the adapter fails closed.
_REGISTERED_CHECK: Optional[Callable[[str, str], Awaitable[bool]]] = None


def set_registered_check(
    checker: Optional[Callable[[str, str], Awaitable[bool]]],
) -> None:
    """Wire in the dev-os registry's ``assert_registered`` checker.

    ``checker`` is an async callable taking ``(app_id, topic_id)`` and
    returning ``True`` when the pair is registered, ``False`` otherwise.
    Passing ``None`` un-wires (test cleanup).
    """
    global _REGISTERED_CHECK
    _REGISTERED_CHECK = checker


async def assert_registered(app_id: str, topic_id: str) -> None:
    """Refuse to proceed unless ``(app_id, topic_id)`` is registered.

    Fails closed: if no checker has been wired,
    :class:`TopicNotRegisteredError` is raised so a partial deployment
    can't silently let a switch through. The pointer write must take
    place inside the same transaction as this check (charter
    invariant 3); callers compose them at the SessionDB boundary.
    """
    if _REGISTERED_CHECK is None:
        raise TopicNotRegisteredError(
            f"no registry checker wired; refusing to register topic_id={topic_id!r} "
            f"under app_id={app_id!r}"
        )
    try:
        ok = await _REGISTERED_CHECK(app_id, topic_id)
    except TopicNotRegisteredError:
        raise
    except Exception as exc:  # noqa: BLE001 — re-wrap and fail closed
        raise TopicNotRegisteredError(
            f"registry check raised for app_id={app_id!r} topic_id={topic_id!r}: {exc}"
        ) from exc
    if not ok:
        raise TopicNotRegisteredError(
            f"topic_id={topic_id!r} is not registered under app_id={app_id!r}"
        )


# ── Read / set helpers (thin SessionDB wrappers) ───────────────────────
#
# These wrap ``SessionDB.read_active_topic`` / ``set_active_topic`` so
# the lock acquisition lives next to the persistent-state call. The
# wrappers are deliberately small — they exist so callers in the slash
# handler and the inbound router import one symbol instead of two.


async def read_active_topic(
    session_db: Any,
    principal: PlatformPrincipal,
) -> Optional[Dict[str, Any]]:
    """Return the pointer row for *principal*, or ``None`` if unset.

    Acquires the per-key lock for the read so the read is serialised
    against any in-flight switch on the same key. Charter invariant 1
    requires the read+route to take place under the same lock; this
    helper makes the lock window explicit for callers that don't
    compose it themselves.
    """
    lock = await acquire_pointer_lock(principal.key)
    async with lock:
        return await asyncio.to_thread(
            session_db.read_active_topic,
            platform=principal.platform,
            user_id=principal.user_id,
            chat_id=principal.chat_id,
            app_id=principal.app_id,
        )


async def set_active_topic(
    session_db: Any,
    principal: PlatformPrincipal,
    *,
    topic_id: str,
    bound_thread_id: Optional[str] = None,
    updated_by: str,
    require_registered: bool = True,
) -> Dict[str, Any]:
    """Set the pointer for *principal* to *topic_id*.

    When ``require_registered`` is True (default), ``assert_registered``
    must succeed before the pointer is upserted. The check and the
    write share one critical section so a concurrent switch can't
    interleave between them; the SessionDB call itself wraps the
    upsert in ``BEGIN IMMEDIATE`` so the write is atomic against
    other processes too.

    Returns the SessionDB response shape ``{"prior": ..., "current": ...}``
    so a caller composing a confirmation banner can compensate
    (re-set to ``prior``) if the banner emit fails — the rollback-
    over-silent-hold contract from the charter.

    NOTE: ``set_active_topic`` here is private to the slash handler
    family per charter invariant 2. Callers outside the slash family
    MUST NOT invoke this. The suggestion path imports
    ``read_active_topic`` only.
    """
    if require_registered:
        await assert_registered(principal.app_id, topic_id)
    lock = await acquire_pointer_lock(principal.key)
    async with lock:
        return await asyncio.to_thread(
            session_db.set_active_topic,
            platform=principal.platform,
            user_id=principal.user_id,
            chat_id=principal.chat_id,
            app_id=principal.app_id,
            topic_id=topic_id,
            bound_thread_id=bound_thread_id,
            updated_by=updated_by,
        )


async def clear_active_topic(
    session_db: Any,
    principal: PlatformPrincipal,
    *,
    updated_by: str,
) -> Optional[Dict[str, Any]]:
    """Clear the pointer for *principal*. Returns the prior row, or ``None``."""
    lock = await acquire_pointer_lock(principal.key)
    async with lock:
        return await asyncio.to_thread(
            session_db.clear_active_topic,
            platform=principal.platform,
            user_id=principal.user_id,
            chat_id=principal.chat_id,
            app_id=principal.app_id,
            updated_by=updated_by,
        )


# ── Inbound routing pre-pass (step 4) ─────────────────────────────────
#
# The routing pre-pass is the inbound-message read of the pointer. It
# returns a topic-routed session key when (and only when) the gateway
# can prove the pointer is safe to honour:
#
#   1. The pointer row exists for the inbound principal/app.
#   2. The registry checker confirms ``(app_id, topic_id)`` is still
#      registered at routing time (defends against a topic that was
#      de-registered after the slash switched to it).
#
# When either fails, the pre-pass returns ``None`` and the caller falls
# through to the legacy ``build_session_key`` path. The "fail closed"
# contract from Critic finding #5 is that an absent or misconfigured
# registry checker **never** silently routes to a pointer-driven topic —
# the routing flip simply doesn't happen, and the inbound message gets
# its legacy thread-derived session.
#
# Lock-window boundary (Critic finding #1): the asyncio per-key lock
# covers the pointer read + the route-key decision. It is released
# **before** the dispatcher hands the turn to the agent runner. The
# agent's full response generation runs **outside** the lock.


def _session_key_namespace_for_topic(profile: Optional[str]) -> str:
    """Mirror ``gateway.session._session_key_namespace`` for topic keys.

    Kept here (instead of importing from ``gateway.session``) so this
    module remains importable from ``hermes_state`` / test scaffolds
    without dragging the whole session module in. Behaviour is identical:
    no/default profile ⇒ ``agent:main``; any named profile ⇒ ``agent:<p>``.
    """
    if not profile or profile == "default":
        return "agent:main"
    return f"agent:{profile}"


def build_topic_session_key(
    principal: PlatformPrincipal,
    *,
    topic_id: str,
    profile: Optional[str] = None,
) -> str:
    """Construct the deterministic topic-routed session key.

    Shape: ``<ns>:topic:<platform>:<chat_id>:<user_id>:<app_id>:<topic_id>``
    where ``<ns>`` is ``agent:main`` for the default profile and
    ``agent:<profile>`` for a named one. The ``topic`` marker after the
    namespace is what distinguishes a topic-routed key from a
    legacy thread-derived key in the same chat — no shared prefix can
    collide with a legacy key.

    Including ``user_id`` in the key matches the principal contract
    (pointer is keyed by ``(platform, user_id, chat_id, app_id)``) — so
    two participants in the same group chat under the same app_id with
    the same topic_id still get separate sessions, which is the charter-
    required isolation when ``group_sessions_per_user`` is on.
    """
    if not topic_id:
        raise ValueError("topic_id is required for topic session key")
    ns = _session_key_namespace_for_topic(profile)
    return (
        f"{ns}:topic:{principal.platform}:{principal.chat_id}:"
        f"{principal.user_id}:{principal.app_id}:{topic_id}"
    )


def resolve_topic_session_key(
    source: Any,
    session_db: Any,
    *,
    app_id: Optional[str],
    profile: Optional[str] = None,
    pointer_mode_enabled: bool = True,
    require_registered_check: bool = True,
) -> Optional[str]:
    """Sync inbound-routing pre-pass. Returns topic key or ``None``.

    Fails closed when ``require_registered_check`` is True and either
    (a) the registry checker is not wired, or (b) the checker rejects
    ``(app_id, topic_id)``. Both cases return ``None`` — the caller
    keeps the legacy thread-derived routing.

    This is the **read** half of the lock window. Slash writes serialise
    against this read via SessionDB's ``BEGIN IMMEDIATE`` (the write
    transaction) and the asyncio per-key lock (when callers compose via
    :func:`resolve_topic_session_key_async`). On the routing hot path
    callers may invoke this sync variant directly, accepting that
    serialisation is via SessionDB transaction isolation alone — which
    is correct for a committed pointer; the narrow window between
    slash commit and slash banner-rollback is documented in the plan
    §9 (residual risks) and is the price of not holding the asyncio
    lock through synchronous routing code paths.
    """
    if not pointer_mode_enabled:
        return None
    if not app_id:
        return None
    try:
        principal = PlatformPrincipal.from_source(source, app_id=app_id)
    except ValueError:
        # Principal cannot be assembled (missing user_id/chat_id) — the
        # only safe route is legacy fall-through; the pointer table is
        # keyed by the full envelope and a partial principal can't
        # safely look it up.
        return None
    try:
        row = session_db.read_active_topic(
            platform=principal.platform,
            user_id=principal.user_id,
            chat_id=principal.chat_id,
            app_id=principal.app_id,
        )
    except Exception:  # noqa: BLE001 — defensive: any DB error falls through
        logger.debug("read_active_topic failed during routing pre-pass", exc_info=True)
        return None
    if not row:
        return None
    topic_id = row.get("topic_id")
    if not topic_id:
        return None
    if require_registered_check:
        # Fail-closed routing: an absent or rejecting checker means we do
        # NOT flip the route. The slash side (step 3 helper) already
        # gates write with the same checker, so a committed pointer
        # passed registry at write time. We re-check on every routing
        # read so a topic de-registered after switch can't keep routing.
        if _REGISTERED_CHECK is None:
            logger.debug(
                "topic registry checker unwired — refusing routing flip "
                "for app_id=%r topic_id=%r (fail-closed)",
                principal.app_id, topic_id,
            )
            return None
        try:
            ok = asyncio.run(_REGISTERED_CHECK(principal.app_id, topic_id))
        except RuntimeError as exc:
            # Already inside a running loop — caller should use the
            # async variant. Fail closed.
            logger.debug(
                "registry check could not run synchronously (use async variant): %s",
                exc,
            )
            return None
        except Exception:  # noqa: BLE001 — defensive
            logger.debug("registry check raised during routing", exc_info=True)
            return None
        if not ok:
            return None
    return build_topic_session_key(principal, topic_id=topic_id, profile=profile)


# ── Legacy mode (step 8) ───────────────────────────────────────────────
#
# HRM-T0a step 8: existing Telegram/forum-thread sessions and any session
# that predates the ``hrm_t0a_applied_at`` migration marker stay in place
# under their original (legacy thread-derived) session key. The routing
# pre-pass already falls through for them automatically because no pointer
# row is set; this section adds:
#
#   - :func:`is_legacy_principal_route` — read-only detection for a single
#     inbound principal/source. Used by the ``/topic`` slash mutator family
#     to refuse state-mutating subcommands with a user-visible banner so
#     a pre-HRM session is never silently migrated.
#
#   - :func:`legacy_inventory` — metadata-only inventory (counts +
#     principal/chat/thread/session ids). MUST NOT include message content
#     by construction so an inventory dump cannot leak transcript data.
#
#   - :data:`LEGACY_READONLY_MESSAGE` — the standard refusal banner.
#
# Policy (owner-approved):
#   * Existing sessions stay in place; no automatic migration.
#   * Legacy normal routing continues (the pre-pass returns ``None`` so
#     the legacy ``build_session_key`` is used as before).
#   * Legacy ``/topic`` mutators (switch/move-last/move-range/clear/
#     bind-thread/unbind-thread) are refused with a clear warning.
#   * Read-only subcommands (``list``/``show``/``help``) still work so a
#     user can inspect the state of their legacy chat.


LEGACY_READONLY_MESSAGE = (
    "gateway.topic.legacy_thread_readonly: this chat is in legacy thread-"
    "derived routing mode. Existing transcripts will not be migrated and "
    "/topic state-mutating subcommands are disabled here. Read-only "
    "subcommands (`/topic`, `/topic list`, `/topic help`) still work."
)


_LEGACY_BANNER_LOGGED = False


def _maybe_log_legacy_banner_once(reason: str) -> None:
    """Emit a single per-process log line when legacy mode is first hit.

    The owner-facing surface banner is the responsibility of the
    platform adapter — there is no shared one-time-per-thread emit
    surface inside this module. This logger pings give operators a
    breadcrumb that legacy mode is active; the user-visible banner
    is the slash refusal text from :data:`LEGACY_READONLY_MESSAGE`.

    Adapter hook (deferred): a per-adapter "first inbound under legacy
    mode" greeting can call into this module's banner constant; that
    hook is NOT implemented in v1 because the gateway has no shared
    emit channel that's safe to fire without an adapter context. See
    step-8 residual risks.
    """
    global _LEGACY_BANNER_LOGGED
    if _LEGACY_BANNER_LOGGED:
        return
    _LEGACY_BANNER_LOGGED = True
    logger.info(
        "HRM-T0a legacy mode engaged (reason=%s). Pointer mutators refused; "
        "legacy routing continues unchanged.",
        reason,
    )


def _reset_legacy_banner_for_tests() -> None:
    """TEST-ONLY: clear the one-time banner latch between cases."""
    global _LEGACY_BANNER_LOGGED
    _LEGACY_BANNER_LOGGED = False


def is_legacy_principal_route(
    session_db: Any,
    source: Any,
    *,
    app_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """Return ``(is_legacy, reason)`` for the given inbound source.

    The principal is in *legacy* mode iff one of these holds (checked in
    order; the first true short-circuits):

    1. The chat is in legacy Telegram-DM forum-thread topic mode
       (``telegram_dm_topic_mode.enabled = 1``). ``reason`` →
       ``"telegram_forum_thread"``.
    2. A pre-HRM-T0a session exists for this (platform, user_id, chat_id):
       ``sessions.started_at < hrm_t0a_applied_at`` AND no
       ``active_topic_pointer`` row is set for this principal/app yet.
       ``reason`` → ``"pre_migration_session"``.

    Returns ``(False, "")`` when neither holds — that is the default
    post-HRM state and the case where a pointer is already established.

    Read-only: never triggers the HRM-T0a migration; never reads message
    content. Safe to call on the slash hot path.
    """
    if session_db is None or source is None:
        return False, ""

    platform_attr = getattr(source, "platform", None)
    platform_str = getattr(platform_attr, "value", None) or (
        str(platform_attr) if platform_attr is not None else ""
    )
    user_id = getattr(source, "user_id", None)
    chat_id = getattr(source, "chat_id", None)
    chat_type = getattr(source, "chat_type", None)

    # 1) Telegram-DM forum-thread legacy path.
    if (
        platform_str == "telegram"
        and chat_type == "dm"
        and user_id
        and chat_id
    ):
        try:
            tm = session_db.is_telegram_topic_mode_enabled(
                chat_id=str(chat_id), user_id=str(user_id)
            )
        except Exception:  # noqa: BLE001 — defensive
            tm = False
        if tm:
            return True, "telegram_forum_thread"

    # 2) Pre-HRM session marker — uses the migration high-water-mark stamp.
    if not user_id or not chat_id or not platform_str:
        return False, ""
    try:
        marker_raw = session_db.get_meta("hrm_t0a_applied_at")
    except Exception:  # noqa: BLE001
        marker_raw = None
    if not marker_raw:
        # Migration never ran. We can't classify by marker; default to
        # not-legacy so a fresh post-step-7 install stays usable.
        return False, ""
    try:
        marker = float(marker_raw)
    except (TypeError, ValueError):
        return False, ""

    # If a pointer already exists for this principal/app, the user has
    # opted into the new mode — legacy guard no longer applies.
    if app_id:
        try:
            row = session_db.read_active_topic(
                platform=str(platform_str),
                user_id=str(user_id),
                chat_id=str(chat_id),
                app_id=str(app_id),
            )
        except Exception:  # noqa: BLE001
            row = None
        if row:
            return False, ""

    try:
        with session_db._lock:
            pre_row = session_db._conn.execute(
                """
                SELECT 1 FROM sessions
                WHERE source = ?
                  AND user_id = ?
                  AND started_at IS NOT NULL
                  AND started_at < ?
                LIMIT 1
                """,
                (str(platform_str), str(user_id), float(marker)),
            ).fetchone()
    except Exception:  # noqa: BLE001 — defensive: any DB error falls through
        logger.debug("legacy-detect: pre-migration session query failed", exc_info=True)
        return False, ""
    if pre_row is not None:
        return True, "pre_migration_session"
    return False, ""


def legacy_inventory(session_db: Any) -> Dict[str, Any]:
    """Metadata-only legacy session inventory.

    Returns a dict shaped::

        {
            "hrm_t0a_applied_at": <float|None>,
            "pre_migration_session_count": <int>,
            "telegram_forum_thread_chat_count": <int>,
            "pre_migration_principals": [
                {"platform": str, "user_id": str},
                ...
            ],
            "telegram_forum_thread_chats": [
                {"chat_id": str, "user_id": str},
                ...
            ],
            "telegram_forum_thread_bindings": [
                {"chat_id": str, "thread_id": str, "session_id": str},
                ...
            ],
        }

    Contract: NEVER includes message content, prompts, responses, titles,
    or any free-form user text. Only structural ids/counts. The principal
    list is deduplicated by ``(platform, user_id)``; chats by ``chat_id``.
    Bindings expose ``session_id`` because that's the routing-key handle,
    not transcript content.

    Caller responsibilities (operator surface): pipe this through any
    redaction step before publishing externally; the metadata IS still
    identifying for the owner's own principals.
    """
    if session_db is None:
        return {
            "hrm_t0a_applied_at": None,
            "pre_migration_session_count": 0,
            "telegram_forum_thread_chat_count": 0,
            "pre_migration_principals": [],
            "telegram_forum_thread_chats": [],
            "telegram_forum_thread_bindings": [],
        }
    try:
        marker_raw = session_db.get_meta("hrm_t0a_applied_at")
    except Exception:  # noqa: BLE001
        marker_raw = None
    try:
        marker = float(marker_raw) if marker_raw else None
    except (TypeError, ValueError):
        marker = None

    pre_count = 0
    principals: list = []
    if marker is not None:
        try:
            with session_db._lock:
                rows = session_db._conn.execute(
                    """
                    SELECT source, user_id, COUNT(*) AS c
                    FROM sessions
                    WHERE started_at IS NOT NULL AND started_at < ?
                    GROUP BY source, user_id
                    """,
                    (float(marker),),
                ).fetchall()
        except Exception:  # noqa: BLE001
            rows = []
        for r in rows:
            if isinstance(r, dict):
                src = r.get("source")
                uid = r.get("user_id")
                cnt = r.get("c") or 0
            else:
                # sqlite3.Row supports mapping access; fall back to index.
                try:
                    src = r["source"]
                    uid = r["user_id"]
                    cnt = r["c"] or 0
                except Exception:
                    src, uid, cnt = r[0], r[1], (r[2] or 0)
            pre_count += int(cnt)
            if src is not None and uid is not None:
                principals.append({"platform": str(src), "user_id": str(uid)})

    chats: list = []
    chat_count = 0
    bindings: list = []
    try:
        with session_db._lock:
            chat_rows = session_db._conn.execute(
                """
                SELECT chat_id, user_id
                FROM telegram_dm_topic_mode
                WHERE enabled = 1
                """
            ).fetchall()
    except Exception:  # noqa: BLE001
        chat_rows = []
    for r in chat_rows:
        try:
            cid = r["chat_id"] if not isinstance(r, tuple) else r[0]
            uid = r["user_id"] if not isinstance(r, tuple) else r[1]
        except Exception:
            cid, uid = r[0], r[1]
        chats.append({"chat_id": str(cid), "user_id": str(uid)})
    chat_count = len(chats)

    try:
        with session_db._lock:
            binding_rows = session_db._conn.execute(
                """
                SELECT chat_id, thread_id, session_id
                FROM telegram_dm_topic_bindings
                """
            ).fetchall()
    except Exception:  # noqa: BLE001
        binding_rows = []
    for r in binding_rows:
        try:
            cid = r["chat_id"] if not isinstance(r, tuple) else r[0]
            tid = r["thread_id"] if not isinstance(r, tuple) else r[1]
            sid = r["session_id"] if not isinstance(r, tuple) else r[2]
        except Exception:
            cid, tid, sid = r[0], r[1], r[2]
        bindings.append(
            {
                "chat_id": str(cid),
                "thread_id": str(tid),
                "session_id": str(sid),
            }
        )

    return {
        "hrm_t0a_applied_at": marker,
        "pre_migration_session_count": pre_count,
        "telegram_forum_thread_chat_count": chat_count,
        "pre_migration_principals": principals,
        "telegram_forum_thread_chats": chats,
        "telegram_forum_thread_bindings": bindings,
    }


async def resolve_topic_session_key_async(
    source: Any,
    session_db: Any,
    *,
    app_id: Optional[str],
    profile: Optional[str] = None,
    pointer_mode_enabled: bool = True,
    require_registered_check: bool = True,
) -> Optional[str]:
    """Async inbound-routing pre-pass with the per-key lock held.

    Lock window (Critic finding #1): held across the pointer read and
    the registry check + key derivation. Released **before** the caller
    hands the turn to the agent runner. The agent's response is NOT
    inside the lock — that would serialise unrelated topics behind a
    long generation.
    """
    if not pointer_mode_enabled or not app_id:
        return None
    try:
        principal = PlatformPrincipal.from_source(source, app_id=app_id)
    except ValueError:
        return None
    lock = await acquire_pointer_lock(principal.key)
    async with lock:
        try:
            row = await asyncio.to_thread(
                session_db.read_active_topic,
                platform=principal.platform,
                user_id=principal.user_id,
                chat_id=principal.chat_id,
                app_id=principal.app_id,
            )
        except Exception:  # noqa: BLE001
            logger.debug("read_active_topic failed during async routing pre-pass", exc_info=True)
            return None
        if not row:
            return None
        topic_id = row.get("topic_id")
        if not topic_id:
            return None
        if require_registered_check:
            if _REGISTERED_CHECK is None:
                logger.debug(
                    "topic registry checker unwired (async path) — refusing routing flip "
                    "for app_id=%r topic_id=%r",
                    principal.app_id, topic_id,
                )
                return None
            try:
                ok = await _REGISTERED_CHECK(principal.app_id, topic_id)
            except Exception:  # noqa: BLE001 — defensive: any checker fault falls through
                logger.debug("registry check raised during async routing", exc_info=True)
                return None
            if not ok:
                return None
        return build_topic_session_key(principal, topic_id=topic_id, profile=profile)
