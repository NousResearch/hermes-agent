"""Gateway routing index, routing peers, orphan repair, never-active pruning and Telegram DM topic state.

The routing index maps a session key to the gateway's ``SessionEntry`` JSON text, partitioned by ``routing_scope`` (the gateway passes its resolved sessions_dir so two stores sharing one database never share routing state). The store keeps the text as given and never interprets it beyond ``$.session_id``: the gateway serializes and parses entries. A store opened on the routing home holds the index every served profile shares; route fences written by ``fence_retired_routes`` and this store's own retirement markers make routing saves skip a retired route.

The peer, origin, orphan and never-active verbs read and stamp the routing columns of the bound store's ``sessions`` rows. Topic rows are keyed by the ROUTED profile (``profile_name``, normalized by :func:`normalize_topic_profile_name`), a row label inside one store, never a store selector; topic tables are created only by ``prepare_topic_storage`` (run by ``enable_topic_mode`` and ``bind_topic``), and reads and cleanups on a store without them answer their empty value, and a topic read that meets a pre-v3 table an upgrade left behind runs prepare_topic_storage and reads again.

Times are epoch-second floats.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from ..core.artifacts import SessionArtifacts
from ..core.errors import Conflict
from ..core.identity import JsonValue, SessionId

if TYPE_CHECKING:
    from ..conversation.sessions import SessionRecord

__all__ = [
    "BARE_BILLING_PROVIDERS",
    "END_REASON_SUPERSEDED_BY_REPAIR",
    "ORPHAN_ADOPTION_MAX_GAP_S",
    "PEER_RESET_END_REASONS",
    "TOPIC_SCHEMA_META_KEY",
    "GatewayRuntime",
    "NeverActivePrune",
    "NeverActiveSession",
    "OrphanReport",
    "RecoveredPeerSession",
    "RoutingRepository",
    "TopicAlreadyLinked",
    "TopicBinding",
    "gateway_runtime",
    "normalize_topic_profile_name",
    "route_namespace",
]

#: End reasons that fence peer recovery: a peer row ended with one of these after a candidate's last activity blocks the candidate.
PEER_RESET_END_REASONS: frozenset[str] = frozenset(
    {"session_reset", "session_switch", "idle", "daily", "suspended", "resume_pending_expired"}
)

#: The end reason ``adopt_orphan`` stamps on the retired donor; recovery never treats it as resumable.
END_REASON_SUPERSEDED_BY_REPAIR = "superseded_by_repair"

#: Default contiguity window of orphan adoption, in seconds.
ORPHAN_ADOPTION_MAX_GAP_S = 900.0

#: Billing buckets that are not routable provider identities.
BARE_BILLING_PROVIDERS: frozenset[str] = frozenset({"auto", "custom"})

#: The ``state_meta`` key ``prepare_topic_storage`` sets to ``"3"``.
TOPIC_SCHEMA_META_KEY = "telegram_dm_topic_schema_version"


def route_namespace(session_key: Optional[str]) -> str:
    """The ``<ns>`` slot of an ``agent:<ns>:...`` routing key; ``main`` for a key outside that shape."""
    parts = str(session_key or "").split(":")
    return parts[1] if len(parts) > 2 and parts[0] == "agent" and parts[1] else "main"


def normalize_topic_profile_name(profile_name: Optional[str] = None) -> str:
    """The stripped profile name of a topic row; empty or missing reads ``default``."""
    name = str(profile_name or "").strip()
    return name if name else "default"


class TopicAlreadyLinked(Conflict, ValueError):
    """``bind_topic`` refused: the session is already bound to another (profile, chat, thread)."""

    def __init__(self, session_id: str) -> None:
        super().__init__("topic_binding", session_id)
        self.args = ("session is already linked to another Telegram topic",)


@dataclass(frozen=True, slots=True)
class TopicBinding:
    """One ``telegram_dm_topic_bindings`` row: a (profile, chat, thread) linked to one session. ``managed_mode`` is ``auto`` or ``restored``; ``linked_at`` is the first bind, ``updated_at`` the latest."""

    profile_name: str
    chat_id: str
    thread_id: str
    user_id: str
    session_key: str
    session_id: SessionId
    managed_mode: str
    linked_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class RecoveredPeerSession:
    """The row ``latest_session_for_peer`` picked: the full session row and whether it has messages (a non-zero ``message_count`` or any stored message)."""

    session: SessionRecord
    has_messages: bool


@dataclass(frozen=True, slots=True)
class OrphanReport:
    """One message-bearing session without a ``session_key``. ``evidence`` is ``lineage``, ``contiguity`` or empty when no donor was named; ``reason`` says why a row is not adoptable."""

    orphan_id: SessionId
    source: Optional[str]
    message_count: int
    started_at: Optional[float]
    last_active: Optional[float]
    donor_id: Optional[SessionId]
    session_key: Optional[str]
    evidence: str
    adoptable: bool
    reason: str


@dataclass(frozen=True, slots=True)
class NeverActiveSession:
    """One keyed, still-open session with no evidence of a single turn."""

    session_id: SessionId
    session_key: str
    source: Optional[str]
    chat_id: Optional[str]
    chat_type: Optional[str]
    user_id: Optional[str]
    started_at: float


@dataclass(frozen=True, slots=True)
class NeverActivePrune:
    """Outcome of ``prune_never_active``: ``deleted_ids`` are the sorted candidate ids the prune did not keep; ``removed_ids`` are the sorted ids of every session the prune removed, the delegate children its delete cascades to included."""

    sessions_deleted: int
    routing_entries_deleted: int
    sessions_skipped: int
    deleted_ids: tuple[SessionId, ...]
    removed_ids: tuple[SessionId, ...]


@dataclass(frozen=True, slots=True)
class GatewayRuntime:
    """The persisted runtime route of a session as ``gateway_runtime`` projects it; every field ``None`` means nothing is stored and resume uses ambient config."""

    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return self.provider is None and self.base_url is None and self.api_mode is None


_ROUTE_KEYS = ("provider", "base_url", "api_mode")


def _route(members: Mapping[str, JsonValue], *, keep_falsy: bool) -> GatewayRuntime:
    picked = {
        key: value
        for key in _ROUTE_KEYS
        if isinstance(value := members.get(key), str) and (keep_falsy or value)
    }
    return GatewayRuntime(**picked)


def gateway_runtime(
    model_config: Optional[Mapping[str, JsonValue]],
    *,
    billing_provider: Optional[str] = None,
    billing_base_url: Optional[str] = None,
) -> GatewayRuntime:
    """Project the runtime route persisted on a session row.

    A nested ``gateway_runtime`` object with a truthy ``provider`` wins whole (its ``None`` members dropped). Otherwise the truthy top-level ``provider`` / ``base_url`` / ``api_mode``, with ``billing_provider`` (stripped) filling a missing provider unless it is a bare bucket of :data:`BARE_BILLING_PROVIDERS`; a bare ``custom`` bucket with no top-level provider surfaces as ``provider='custom'`` with ``billing_base_url`` when that endpoint is stored. When nothing above is set, a provider-less nested object's non-``None`` members; else the empty route. Members that are not strings read ``None``.
    """
    raw: Mapping[str, JsonValue] = model_config or {}
    nested = raw.get("gateway_runtime")
    runtime = nested if isinstance(nested, Mapping) else None
    if runtime is not None and runtime.get("provider"):
        return _route(runtime, keep_falsy=True)
    top_level: dict[str, JsonValue] = {key: raw.get(key) for key in _ROUTE_KEYS if raw.get(key)}
    provider = (billing_provider or "").strip()
    if provider and provider.lower() not in BARE_BILLING_PROVIDERS:
        top_level.setdefault("provider", provider)
    elif provider.lower() == "custom" and not top_level.get("provider"):
        base_url = (billing_base_url or "").strip()
        if base_url:
            top_level.setdefault("base_url", base_url)
            top_level["provider"] = "custom"
    if top_level:
        return _route(top_level, keep_falsy=False)
    if runtime is None:
        return GatewayRuntime()
    return _route(runtime, keep_falsy=True)


@runtime_checkable
class RoutingRepository(Protocol):
    """Gateway routing index, routing peers and Telegram DM topic state of the bound store."""

    # ── writes ──────────────────────────────────────────────────────────────

    def upsert_routing_entry(self, session_key: str, entry_json: str, *, routing_scope: str = "") -> None:
        """Insert or replace the entry under ``session_key`` in ``routing_scope``. An empty key or entry is a silent no-op; a retired route (``route_retired``) is silently skipped inside the same write."""
        ...

    def replace_routing_entries(self, entries: Mapping[str, str], *, routing_scope: str = "") -> set[str]:
        """Replace ``routing_scope``'s whole index with ``entries`` in one write (keys absent from ``entries`` go; other routing scopes are untouched). Pairs with an empty key or entry are dropped; retired routes are skipped, and their keys are returned so the sessions.json mirror leaves them out too."""
        ...

    def fence_retired_routes(self, session_ids: Iterable[str], *, namespace: str, expiring: Iterable[str] = ()) -> None:
        """Fence the routes of ``session_ids``, retired in the profile whose routing keys use the ``agent:<namespace>:`` slot, and drop them, in one write. Sweep fences older than the retry horizon are reclaimed first; each id gets a fence for ``namespace`` and, when this store holds no session of that id, for ``main`` too; a fence of an ``expiring`` id ages out at the horizon and never shortens a permanent one. Every routing row (any routing scope) whose key's namespace is fenced for the id its entry names is deleted. No id: no write."""
        ...

    def record_session_peer(
        self,
        session_id: str,
        *,
        source: str,
        user_id: str | None = None,
        session_key: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        thread_id: str | None = None,
        display_name: str | None = None,
        origin_json: str | None = None,
        include_compression_ancestors: bool = False,
        messaging_lane: bool = False,
        transport_profile: str | None = None,
    ) -> None:
        """Stamp the routing peer on the session row: ``source`` and the five routing columns are replaced (``None`` stores null); ``display_name``, ``origin_json`` and ``transport_profile`` replace only when given. An empty ``session_id`` or ``session_key`` is a silent no-op. A missing row is inserted with the full identity, ``created_source = source``, the store owner as ``profile_name`` and ``started_at`` now (no retirement check). With ``include_compression_ancestors`` the stamp lands on the row and every compression ancestor (parent ended ``compression``, child not a branch, delegate, reset or ``tool`` row) and nothing is inserted; ``messaging_lane`` is the caller's answer whether ``source`` names a messaging platform, and when it is True every local conversation whose receipt's current target is one of the stamped rows is handed off to that lane in the same write, as ``SessionBindingRepository.release_handed_off_receipts`` hands one off. ``messaging_lane`` is read only with ``include_compression_ancestors``."""
        ...

    def adopt_orphan(self, orphan_id: str, donor_id: str) -> bool:
        """Give ``orphan_id`` the routing identity of ``donor_id`` and retire the donor in one write. Both rows are re-read inside the write: False, nothing written, when an id is empty, the ids are equal, a row is missing, the donor is unkeyed, the orphan is keyed, the sources differ, or (store inside the profile tree) the effective profiles differ. Otherwise the orphan takes the donor's key and fills its null ``chat_id``, ``chat_type``, ``thread_id``, ``user_id``, ``origin``, ``display_name`` and ``parent_session_id`` (the donor's id) from the donor; the donor is ended now unless already ended, with ``end_reason`` :data:`END_REASON_SUPERSEDED_BY_REPAIR`."""
        ...

    def prune_never_active(self, *, older_than_days: float, artifacts: SessionArtifacts | None = None, media_root: str) -> NeverActivePrune:
        """Delete the ``never_active_keyed_sessions`` candidates (a sequence of transactions): local reset tips (a candidate whose local conversation holds other segments) are excluded; the routing entries naming the rest are counted; the rows are deleted as ``SessionRepository.delete_many(skip_guarded=True)`` deletes them (delegate cascade, retirement and route drop in one write; a row under a live turn lease or compression lock is kept and counted skipped). The prune ends with that delete: the caller runs the delete's post-commit work, then ``sweep_routing_entries_for_sessions`` over ``deleted_ids`` for the entries a concurrent writer re-added. ``routing_entries_deleted`` counts the entries that named a candidate that was not kept, read before the delete. With ``artifacts`` the transcript files of every removed id are unlinked after the delete commits. ``older_than_days < 0`` raises ``ValueError``. ``media_root`` goes to that delete as ``SessionRepository.delete`` takes it."""
        ...

    def sweep_routing_entries_for_sessions(self, session_ids: Iterable[str]) -> int:
        """Drop every routing entry, in any routing scope, whose entry names one of ``session_ids``, in one write; returns how many were dropped. No matching entry: no write."""
        ...

    def prepare_topic_storage(self) -> None:
        """Create the topic-mode and topic-binding tables with their indexes and set :data:`TOPIC_SCHEMA_META_KEY` to ``"3"``, in one write; idempotent. Run on explicit ``/topic`` opt-in, never at startup."""
        ...

    def enable_topic_mode(
        self,
        *,
        chat_id: str,
        user_id: str,
        profile_name: str = "default",
        has_topics_enabled: bool | None = None,
        allows_users_to_create_topics: bool | None = None,
    ) -> None:
        """Run ``prepare_topic_storage``, then upsert the (profile, chat) mode row in a second write: enabled, ``user_id`` the enabling user, the capability flags as given (``None`` unchecked), ``activated_at`` kept from the first enable."""
        ...

    def disable_topic_mode(self, *, chat_id: str, profile_name: str = "default", clear_bindings: bool = True) -> None:
        """Flip the (profile, chat) mode off and, with ``clear_bindings``, delete the chat's bindings of that profile, in one write. Never creates the tables; without them it is a no-op."""
        ...

    def delete_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> int:
        """Delete the (profile, chat, thread) binding and return the rows deleted (0 when absent or without the tables). Removing the chat's last binding of that profile flips its mode off in the same write."""
        ...

    def bind_topic(
        self,
        *,
        chat_id: str,
        thread_id: str,
        user_id: str,
        session_key: str,
        session_id: str,
        managed_mode: str = "auto",
        profile_name: str = "default",
        expected_session_id: str | None = None,
        move_session: bool = False,
    ) -> bool:
        """Run ``prepare_topic_storage``, then bind the (profile, chat, thread) to the session in a second write. A session already bound to another (profile, chat, thread) raises :class:`TopicAlreadyLinked`, unless ``move_session`` (a handoff moves the conversation) and that binding is in the same profile: it is dropped in the same write. Rebinding the same triple refreshes it (``linked_at`` kept); binding the triple to another session replaces the old session's binding. A session id with no row raises ``SessionRowMissing``. ``expected_session_id`` makes the bind a compare-and-swap inside that write: it goes through only while the triple's stored binding names that id, and an absent binding or one naming another session (a concurrent /new or /topic off that rebound or cleared the topic) returns ``False`` with nothing written. ``True`` otherwise."""
        ...

    # ── reads ───────────────────────────────────────────────────────────────

    def route_retired(self, session_key: str, entry_json: str) -> bool:
        """Whether a routing save must skip this entry, judged against the profile the route belongs to: this store's own retirement marker of the entry's ``$.session_id`` for a ``main`` route, or a route fence for the key's namespace that has not aged out (a ``main`` fence only while this store holds no session of that id)."""
        ...

    def load_routing_entries(self, *, routing_scope: str = "") -> dict[str, str]:
        """``routing_scope``'s index as ``{session_key: entry_json}``, the text as stored."""
        ...

    def routing_entry_for_session(self, session_id: str) -> Mapping[str, JsonValue] | None:
        """The parsed entry, in any routing scope, whose ``session_id`` is the given id, or ``None``; malformed or non-object entries are skipped and the first match wins."""
        ...

    def latest_session_for_peer(
        self,
        *,
        source: str,
        user_id: str | None = None,
        session_key: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        thread_id: str | None = None,
    ) -> RecoveredPeerSession | None:
        """The latest recoverable session of a routing peer, one read; an empty ``session_key`` reads ``None``.

        A candidate is open, ended with a reason in ``RECOVERABLE_END_REASONS``, or owned by a completed handoff; it is fenced when a peer row (itself included) ended with a :data:`PEER_RESET_END_REASONS` reason after the candidate's ``COALESCE(last_activity_at, started_at)``. Exact arm: rows with this ``session_key`` and ``source``, messages first, then the newest recency. Tuple arm, only when the exact arm found nothing and ``chat_id`` and ``chat_type`` are both given: same ``source`` and empty-string-equal ``user_id``, ``chat_id``, ``chat_type``, ``thread_id``, with messages, newest recency; inside the profile tree a candidate and a fencing row must carry the store owner's profile or none. Hidden and archived rows are candidates; equal recency has no promised order.
        """
        ...

    def session_for_origin(
        self, *, platform: str, chat_id: str, thread_id: str | None = None, user_id: str | None = None
    ) -> str | None:
        """The newest open keyed session whose ``source`` equals ``platform`` under ASCII case folding and whose ``chat_id`` equals ``chat_id`` (with ``thread_id``: its empty-string-equal thread too), or ``None``. With ``user_id`` the newest exact sender match wins and several rows without one read ``None``; without it, rows carrying more than one distinct non-empty (stripped) user read ``None``. An empty ``platform`` or ``chat_id`` reads ``None``."""
        ...

    def orphaned_sessions(self, *, max_gap_s: float | None = None) -> tuple[OrphanReport, ...]:
        """Every message-bearing session without a ``session_key`` that is no branch, delegate or ``tool`` row, oldest ``started_at`` first, one read.

        A row with a ``parent_session_id`` names its parent (``lineage``) when the parent is keyed with the same source, else reads not adoptable. A row without one names the one keyed same-source row (``contiguity``) whose user is compatible (either empty, or equal), whose last activity lies within ``max_gap_s`` (default :data:`ORPHAN_ADOPTION_MAX_GAP_S`) of the orphan's start and is older than the orphan's last activity; none or several read not adoptable. Inside the profile tree the donor's effective profile (``NULL`` reads as the store owner) must equal the orphan's. Rows naming one donor are all demoted. Nothing is written.
        """
        ...

    def never_active_keyed_sessions(self, *, older_than_days: float) -> tuple[NeverActiveSession, ...]:
        """Keyed, still-open rows started more than ``older_than_days`` ago with no title, no activity stamp, zero message, tool-call, API-call and token counters, no stored message, not pinned and not archived, ordered by ``started_at``. ``older_than_days < 0`` raises ``ValueError``."""
        ...

    def is_topic_mode_enabled(self, *, chat_id: str, user_id: str, profile_name: str = "default") -> bool:
        """Whether the (profile, chat) mode row of this user is enabled; ``False`` without the tables."""
        ...

    def get_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> TopicBinding | None:
        """The (profile, chat, thread) binding, or ``None``."""
        ...

    def list_topic_bindings_for_chat(self, *, chat_id: str, profile_name: str = "default") -> tuple[TopicBinding, ...]:
        """The chat's bindings of that profile, newest ``updated_at`` first; equal stamps have no promised order."""
        ...

    def get_topic_binding_for_session(self, *, session_id: str) -> TopicBinding | None:
        """The session's binding under any profile, or ``None``."""
        ...

    def is_session_linked_to_topic(self, *, session_id: str) -> bool:
        """Whether the session is bound to any topic under any profile."""
        ...
