"""Rotation-stable logical cache scope for prompt_cache_key derivation.

Context-compression rotation (legacy ``compression.in_place: false`` mode)
mints a new physical ``session_id`` mid-conversation to segment the
transcript. The prompt-cache scope introduced by #79161 was derived from that
physical id, so every rotation moved the conversation into a fresh cache
bucket even though it is logically the same conversation continuing
(issue #79017).

``resolve_prompt_cache_scope()`` maps the physical session id to the ROOT of
its *compression lineage* — the pre-rotation session id — using
``SessionDB.get_compression_lineage()``, whose fork-aware semantics
(hardened in #79193) give exactly the scope boundaries the cache key needs.
NOT ``SessionDB.get_conversation_root`` / ``run_agent._conversation_root_id``
(the Portal-attribution walk): that one follows ``parent_session_id`` blindly,
collapsing /branch children and whole delegate trees into one id, which would
violate the #79161 isolation this scope must preserve. The two resolvers are
intentionally different — do not "deduplicate" them.

- compression-rotation children walk back to the original segment
  (rotation-stable scope — the fix);
- ``/new`` starts a lineage-less session (fresh scope);
- ``/branch`` children (``_branched_from``), delegate subagents
  (``_delegate_from``), and tool-tagged children (``source="tool"``) are
  explicit fork children and keep their own isolated scope, preserving the
  sibling/subagent isolation #79161 established;
- cron fires keep their physical ``cron_<job>_<ts>`` id here — the per-fire
  timestamp is stripped later by ``_cache_scope_from_session_id`` exactly as
  before.

A host that mints one physical ``session_id`` per RESPONSE (Hermes Studio's
group chat, and ``POST /v1/responses`` with client-managed history, which
mints ``str(uuid4())`` per request) re-keys every conversation-affinity hint
Hermes sends — ``prompt_cache_key`` on both OpenAI-wire transports, plus the
OpenRouter/Nous sticky ``session_id`` and xAI's ``x-grok-conv-id`` through
``portal_tags`` (issue #96811). Those rows carry no lineage, so the walk
above correctly returns the physical id and the scope moves every reply.

Hermes must not infer the logical conversation from the id's SYNTAX (that
rule collides independent client-supplied ids and merges Studio members
truncated past its 96-character boundary — the #79017 failure class). The
host has to declare it, and one carrier already means exactly that:
``gateway_session_key`` — the "stable per-chat key" (``agent:main:telegram:
dm:123``) built by ``gateway.session.build_session_key`` from the
``X-Hermes-Session-Key`` header, which branching deliberately does NOT key
off. ``declared_conversation_scope()`` consumes it, and it wins over the
lineage walk because it is stable across rotation AND across per-response
ids. Two boundaries it must not cross:

- explicit fork children (``/branch``, delegate subagents, tool children)
  share their parent's chat key but are separate conversations — the row's
  fork markers keep them on their own scope (#79161);
- background-review forks run on a clone of the live runtime, so they are
  excluded by ``_persist_disabled`` for the same reason.

The declared key is hashed into ``gwk_<sha256[:24]>`` before it becomes a
scope: unlike a session id it embeds platform/chat/user identifiers, and
this value leaves the process verbatim as OpenRouter's sticky ``session_id``
and xAI's ``x-grok-conv-id``.

The resolution is memoized per (agent, session_id): the lineage walk runs
once per transcript segment — NOT per API call — and re-runs only when
rotation actually changes ``agent.session_id`` (per the no-DB-on-the-hot-path
constraint recorded on #79017). Default installs compact in place and never
rotate, so they hit the memo forever and behave byte-identically to before.
"""

import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MEMO_ATTR = "_prompt_cache_scope_memo"
# Namespace for a scope resolved from a host-declared conversation key.
_DECLARED_SCOPE_PREFIX = "gwk_"


def _lineage_root(session_id: str, session_db: Any) -> Optional[str]:
    """Return the compression-lineage root of *session_id*, or None.

    Defensive about the DB handle: test doubles and partially constructed
    agents can hand back non-list results — anything that is not a non-empty
    list/tuple whose first element is a non-empty string is ignored.
    """
    if session_db is None:
        return None
    try:
        lineage = session_db.get_compression_lineage(session_id)
    except Exception:
        logger.debug("prompt-cache scope lineage walk failed", exc_info=True)
        return None
    if isinstance(lineage, (list, tuple)) and lineage:
        root = lineage[0]
        if isinstance(root, str) and root:
            return root
    return None


def _agent_source(
    agent: Any, session_id: str, session_db: Any, row_source: Optional[str] = None
) -> str:
    """The ``sessions.source`` this agent's conversation is recorded under.

    Read from the agent's own row when it exists, because that is the value
    the peer queries below match on.

    ``row_source`` is that value when the caller already has it — the single
    identity read in :func:`declared_conversation_scope` — where ``""`` means
    "the row was read and carries no source". ``None`` means "not read yet"
    and keeps the original lookup, which is the path a ``SessionDB`` without
    :meth:`~hermes_state.SessionDB.declared_scope_identity` still takes.

    Before the row lands — this module resolves the first scope ahead of
    ``_ensure_db_session`` — it uses the SAME resolver persistence will use,
    ``run_agent._session_source_for_agent``, not ``agent.platform``. The two
    diverge whenever ``HERMES_SESSION_SOURCE`` overrides the platform, and the
    divergence is not a cosmetic one: the declared scope is non-``None``
    immediately, so ``resolve_prompt_cache_scope`` memoizes it for this session
    id and never re-resolves once the authoritative row appears. Both sides of
    a ``/new`` would then read the platform domain, miss the boundary recorded
    under the override, and hash the same scope.
    """
    if row_source is None and session_id and session_db is not None:
        try:
            row = session_db.get_session(session_id)
        except Exception:
            logger.debug("declared-scope source lookup failed", exc_info=True)
            row = None
        row_source = str(row.get("source") or "").strip() if row else ""
    if row_source:
        return row_source
    platform = getattr(agent, "platform", None)
    try:
        # Imported lazily: run_agent imports this module, and this is the
        # single owner of the source a session row is created with.
        from run_agent import _session_source_for_agent

        source = str(_session_source_for_agent(platform) or "").strip()
        if source:
            return source
    except Exception:
        logger.debug("declared-scope source authority unavailable", exc_info=True)
    return str(platform or "").strip()


def _conversation_generation(session_key: str, source: str, session_db: Any) -> str:
    """Return the generation marker for *session_key*'s current conversation.

    The declared key is a per-CHAT identifier and deliberately outlives any
    single conversation on it — ``reset_session()`` mints a fresh physical id
    on ``/new`` but keeps the key, and the idle/daily/suspended policy resets
    do the same. Hashing the key alone would therefore map the conversation
    before a reset and the one after it onto ONE affinity scope, violating the
    #79017/#86733 contract (warm across compression rotation, cold across a
    new conversation).

    The generation that must rotate is already durable: every one of those
    boundaries closes the outgoing row with an ``_RESET_END_REASONS``
    end_reason, and ``SessionDB.latest_conversation_boundary`` reports the most
    recent one. Qualifying the key with it gives a carrier that is

    - stable across a host's per-response physical ids (no boundary is written
      when nothing was reset, so every reply hashes the same value), and
    - rotating on every conversation replacement, ``/new`` and the policy
      auto-resets alike.

    The marker is a durable counter kept outside prunable session history
    (``conversation_generations``, advanced in the same transaction that writes
    each boundary). Deriving it from the session rows instead — an aggregate
    over ``_RESET_END_REASONS`` boundaries — cannot prove non-reuse: deleting or
    pruning an ended row makes the aggregate return a pair it already emitted,
    handing a new conversation a retired affinity identity. It is also
    wall-clock-free, so a backwards NTP correction cannot reorder it.

    No counter is introduced anywhere: the marker is read from state the
    reset paths already write, and it is read on the memoized resolution path,
    not per API call.

    Returns ``""`` when the key has never been reset, when the DB does not
    expose the lookup, or when it reports nothing.
    """
    reader = getattr(session_db, "latest_conversation_boundary", None)
    if not callable(reader):
        return ""
    generation = reader(session_key, source)
    if generation is None:
        return ""
    return str(int(generation))


def declared_conversation_scope(agent: Any) -> Optional[str]:
    """Return the host-declared logical conversation scope, or None.

    Resolved from ``agent._gateway_session_key`` (the ``X-Hermes-Session-Key``
    /``build_session_key`` per-chat key) qualified by the conversation
    generation currently live on it (:func:`_conversation_generation`), hashed
    together into ``gwk_<sha256[:24]>`` so no platform/chat/user identifier
    reaches a provider on the wire and the value stays inside every caller's
    length/charset budget.

    The key alone would outlive the conversation — it survives ``/new`` and the
    idle/daily policy resets by design — so the generation is what makes this
    carrier legal: stable across a host's per-response physical ids, and cold
    on every conversation replacement.

    None — meaning "fall back to the physical-id scope" — when no key was
    declared, when this agent is a background-review fork (``_persist_disabled``:
    it clones the live runtime, including the key), when the session row is an
    explicit fork child (``/branch``, delegate, tool), and on any DB error
    during either lookup.
    """
    key = str(getattr(agent, "_gateway_session_key", "") or "").strip()
    if not key:
        return None
    if getattr(agent, "_persist_disabled", False):
        return None
    sid = str(getattr(agent, "session_id", None) or "")
    db = getattr(agent, "_session_db", None)
    generation = ""
    row_source: Optional[str] = None
    if sid and db is not None:
        try:
            # One read for both halves of the row's identity: the fork verdict
            # and the source the peer queries match on live on the same
            # ``sessions`` row, and asking for them separately read it twice
            # per resolution (@teknium1 on #98811).  A SessionDB without the
            # combined view keeps the original call, so nothing that predates
            # it — including the doubles that certify the fail-closed contract
            # below — changes behaviour.
            identity = getattr(db, "declared_scope_identity", None)
            if callable(identity):
                is_fork, row_source = identity(sid)
            else:
                is_fork = db.is_explicit_fork_child(sid)
            if is_fork:
                return None
        except Exception:
            # Degrade to the physical-id scope rather than risk merging a
            # fork onto its parent's key on a transient DB failure.  The
            # source read is inside this same guard for the same reason: it
            # was always the second half of a read that had already failed
            # closed here.
            logger.debug("declared-scope fork check failed", exc_info=True)
            return None
    source = _agent_source(agent, sid, db, row_source)
    if db is not None:
        try:
            generation = _conversation_generation(key, source, db)
        except Exception:
            # Same fail-closed rule as the fork check: an unqualified key
            # spans /new, so degrade to the physical-id scope instead.
            logger.debug("declared-scope generation read failed", exc_info=True)
            return None
    # The carrier is the SAME identity tuple the peer queries use: two hosts
    # may legally declare the same key string under different sources, and the
    # scope leaves this process as a routing key, so it must not collapse them.
    carrier = f"{source}|{key}|{generation}"
    digest = hashlib.sha256(carrier.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"{_DECLARED_SCOPE_PREFIX}{digest}"


def resolve_prompt_cache_scope(agent: Any) -> str:
    """Resolve the rotation-stable cache-scope id for *agent*'s conversation.

    Returns the host-declared conversation scope when one applies
    (``declared_conversation_scope``), else the compression-lineage ROOT of
    ``agent.session_id`` (the physical id itself when the session has no
    compression ancestry, no DB is attached, or the walk fails). The result is memoized on the agent
    keyed by the current session id, so the DB walk happens once per
    transcript segment rather than once per API call.
    """
    sid = str(getattr(agent, "session_id", None) or "")
    if not sid:
        return ""
    db = getattr(agent, "_session_db", None)
    # Memo key includes DB presence: an agent that starts DB-less and gains a
    # handle later (run_agent._get_session_db_for_recall lazily attaches one)
    # must re-resolve instead of staying pinned to the physical id.
    key = (sid, db is not None)
    memo = getattr(agent, _MEMO_ATTR, None)
    if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == key:
        return memo[1]
    # A declared conversation key outranks the lineage walk: it is stable
    # across compression rotation AND across a host's per-response ids, which
    # the walk cannot see (#96811).
    root = declared_conversation_scope(agent) or (
        _lineage_root(sid, db) if db is not None else None
    )
    scope = root or sid
    # Memoize on a successful walk, or when there is no DB to consult at all,
    # or when the agent will never persist a row (background-review forks set
    # _persist_disabled but still hold a DB handle — without this, every API
    # call would re-run the lineage query forever).
    # A failed/empty walk on a persisting agent is NOT memoized: falling back
    # to the physical id is the correct degraded answer right now (row not
    # persisted yet, transient DB error), but pinning it for the whole segment
    # would keep the scope wrong after the session row lands.
    if (
        root is not None
        or db is None
        or getattr(agent, "_persist_disabled", False)
    ):
        try:
            setattr(agent, _MEMO_ATTR, (key, scope))
        except Exception:
            # Frozen/slotted test doubles — resolution still works, just
            # unmemoized.
            pass
    return scope


def declared_conversation_scope_safe(agent: Any) -> Optional[str]:
    """Never-raising variant of :func:`declared_conversation_scope`."""
    try:
        return declared_conversation_scope(agent)
    except Exception:
        logger.debug("declared conversation scope resolution failed", exc_info=True)
        return None


def resolve_prompt_cache_scope_safe(agent: Any) -> Optional[str]:
    """Never-raising variant of :func:`resolve_prompt_cache_scope`.

    Returns None on any failure (or when there is no scope). Consumers treat
    None/empty as "fall back to the physical session_id", so a resolution
    failure degrades to pre-#79017 behavior instead of blocking the caller —
    important at turn_context's call site, where an exception raised inside
    the ``set_runtime_main(...)`` argument list would otherwise skip the whole
    runtime binding, not just the cache scope.
    """
    try:
        return resolve_prompt_cache_scope(agent) or None
    except Exception:
        logger.debug("prompt-cache scope resolution failed", exc_info=True)
        return None
