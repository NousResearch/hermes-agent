"""Local creation receipts, API bindings and the local lineage they guard.

A local session's row, its gateway route and its frozen launch policy are reserved in ONE write; the stored receipt (``gateway.local_policy.v1:<id>``) is the session's identity, and its ``entry['session_id']`` is the current physical transcript, which compression and ``/new`` resets advance along ``lineage`` behind the boundary. An API session binds its route, its declared conversation key and its receipt in one write. The history-control check for unbound legacy history is one read.

Owner-memory checks arrive as data, never as callables: ``bind_api(gate=)`` is the admission gate the gateway decided, raised right after the epoch check; ``authorize_history(native_transport=)`` is the native transport registry's answer; ``owner=`` is the :class:`persistence.authority.vocabulary.OwnerRule` stored owners and client labels are read by (``session_rules.legacy_local``: without one only an equal stamp or the subject's own login label is its owner, and no label is a client app's). Refusals are :class:`persistence.AuthorityRefusal` with a reason code (``stale_epoch`` as :class:`persistence.FenceStale`, ``permission_denied`` as :class:`persistence.ProofRejected`).

The pure identity derivations the gateway, the backend's generation walks and its export share live here: :func:`local_session_id`, :func:`cron_request_id`, :func:`cron_run_from_request`, :func:`legacy_cron_run_job`, :func:`api_generation_id`, :func:`room_session_id`, :func:`generation_request_id`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from ..core.identity import JsonValue, SessionId
from .admissions import GateRefusal
from .vocabulary import CRON_REQUEST_PREFIX, OwnerRule

__all__ = [
    "ApiBinding",
    "BindingMarks",
    "LegacyAdoptionScan",
    "LocalLineage",
    "NewApiBinding",
    "NewSessionBinding",
    "ReservedBinding",
    "SessionBinding",
    "SessionBindingRepository",
    "api_generation_id",
    "cron_request_id",
    "cron_run_from_request",
    "generation_request_id",
    "legacy_cron_run_job",
    "local_session_id",
    "room_session_id",
]

_LEGACY_CRON_RUN_ID = re.compile(r"^cron_(.+)_\d{8}_\d{6}$")


def local_session_id(profile_id: str, principal_id: str, request_id: str) -> SessionId:
    """The id a bound local session must carry: ``"local-"`` plus the sha256 hex of the compact JSON array ``[profile_id, principal_id, request_id]`` (``gateway/session_local_recovery.py`` ``local_identity``). A receipt whose session id is not this derivation of its stored profile and owner/request identity fails :meth:`SessionBindingRepository.validate_lineage`."""
    identity = json.dumps([profile_id, principal_id, request_id], separators=(",", ":"))
    return SessionId("local-" + hashlib.sha256(identity.encode()).hexdigest())


def cron_request_id(job_id: str, execution_id: str) -> str:
    """The admission request id of one cron fire, ``cron:<job>:<execution>``."""
    return f"{CRON_REQUEST_PREFIX}{job_id}:{execution_id}"


def cron_run_from_request(request_id: object, job_id: Optional[str] = None) -> Optional[tuple[str, str]]:
    """``(job_id, execution_id)`` of a cron fire's request id, else None: without ``job_id`` the id must hold exactly one ``:`` after the prefix; with it, the id must start with that job's prefix."""
    if not isinstance(request_id, str) or not request_id.startswith(CRON_REQUEST_PREFIX):
        return None
    if job_id is None:
        rest = request_id[len(CRON_REQUEST_PREFIX):]
        if rest.count(":") != 1:
            return None
        job_id, _, execution_id = rest.partition(":")
    else:
        prefix = cron_request_id(job_id, "")
        execution_id = request_id[len(prefix):] if request_id.startswith(prefix) else ""
    return (job_id, execution_id) if job_id and execution_id else None


def legacy_cron_run_job(session_id: object) -> Optional[str]:
    """The job id a pre-unified ``cron_{job_id}_{YYYYmmdd_HHMMSS}`` run session names, else None."""
    match = _LEGACY_CRON_RUN_ID.match(str(session_id or ""))
    return match.group(1) if match else None


def api_generation_id(session_id: str, generation: int) -> SessionId:
    """Generation ``generation`` of an API session id: the id itself, then ``api-<digest(id, n)>``."""
    if not generation:
        return SessionId(session_id)
    return SessionId("api-" + hashlib.sha256(f"{session_id}\0{generation}".encode()).hexdigest()[:16])


def room_session_id(identity: tuple[str, str, str, str], generation: int = 0) -> SessionId:
    """``room_<digest>`` of a hosted room member's identity (home install, room, member, target profile); a later generation folds its number into the digest."""
    seed = "\0".join(identity) + (f"\0{generation}" if generation else "")
    return SessionId("room_" + hashlib.sha256(seed.encode()).hexdigest()[:32])


def generation_request_id(base_request_id: str, generation: int) -> str:
    """The creation request id of generation ``generation`` of a derived local session: the base id, then ``<base>:<n>``."""
    return base_request_id if generation == 0 else f"{base_request_id}:{generation}"


def _require_iso(value: object, field_name: str, record: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{record}.entry[{field_name!r}] must be an ISO-8601 string")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{record}.entry[{field_name!r}] is not ISO-8601: {value!r}") from exc


def _require_text(record: object, names: tuple[str, ...], *, optional: bool = False) -> None:
    for name in names:
        value = getattr(record, name)
        if optional and value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"{type(record).__name__}.{name} must be a non-empty string" + (" or None" if optional else ""))


@dataclass(frozen=True, slots=True)
class NewSessionBinding:
    """The receipt ``reserve`` stores.

    ``policy`` is the frozen launch policy (the store reads ``source``, ``model``, ``cwd``, ``request_json``, ``client_env_json``, ``client_env_rules_json``); ``entry`` the routing entry (the store reads ``session_id``, ``created_at`` and ``origin['chat_id']``); ``created_source`` the creation surface the caller computed from ``policy['source']`` (``gateway.session_policy.created_surface``), ``None`` or empty for a source that is no surface (``local``, ``unknown``), stored as NULL. ``legacy_session_id`` and ``lineage`` are set only on the legacy adoption arm. Everything else is caller wire state stored verbatim.
    """

    profile_id: str
    principal_id: str
    request_id: str
    session_id: SessionId
    route: str
    entry: Mapping[str, JsonValue]
    policy: Mapping[str, JsonValue]
    created_source: Optional[str]
    legacy_session_id: Optional[str] = None
    lineage: Optional[tuple[SessionId, ...]] = None

    def __post_init__(self) -> None:
        _require_text(self, ("profile_id", "principal_id", "request_id", "session_id", "route"))
        _require_text(self, ("legacy_session_id",), optional=True)
        if self.created_source is not None and not isinstance(self.created_source, str):
            raise ValueError("NewSessionBinding.created_source must be a string or None")
        source = self.policy.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("NewSessionBinding.policy['source'] must be a non-empty string")
        if not isinstance(self.policy.get("request_json"), str):
            raise ValueError("NewSessionBinding.policy['request_json'] must be a string (the replay dedup digest)")
        for name in ("model", "cwd"):
            if name not in self.policy:
                raise ValueError(f"NewSessionBinding.policy must carry {name!r} (a string or None)")
            value = self.policy.get(name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"NewSessionBinding.policy[{name!r}] must be a string or None")
        if self.entry.get("session_id") != self.session_id:
            raise ValueError("NewSessionBinding.entry['session_id'] must equal session_id: a reservation targets its own transcript")
        _require_iso(self.entry.get("created_at"), "created_at", "NewSessionBinding")
        _require_iso(self.entry.get("updated_at"), "updated_at", "NewSessionBinding")
        origin = self.entry.get("origin")
        if not isinstance(origin, Mapping) or not origin.get("chat_id"):
            raise ValueError("NewSessionBinding.entry['origin'] must be a mapping with a non-empty 'chat_id'")
        if self.lineage is not None and not self.lineage:
            raise ValueError("NewSessionBinding.lineage must be non-empty when set")


@dataclass(frozen=True, slots=True)
class SessionBinding:
    """One stored creation receipt, detached from the store. ``lineage`` is the stored chain, ``(session_id,)`` when the receipt carries none; ``current_target`` is the stored ``entry['session_id']``. Compression and resets advance both, so a re-read may show a later target."""

    profile_id: str
    principal_id: str
    request_id: str
    session_id: SessionId
    route: str
    entry: Mapping[str, JsonValue]
    policy: Mapping[str, JsonValue]
    lineage: tuple[SessionId, ...]
    legacy_session_id: Optional[str] = None

    @property
    def current_target(self) -> SessionId:
        """The receipt's current physical transcript (``entry['session_id']``)."""
        return SessionId(str(self.entry.get("session_id")))


@dataclass(frozen=True, slots=True)
class ReservedBinding:
    """``reserve``'s outcome: the stored binding and whether this call inserted it (False on the stored-receipt replay)."""

    binding: SessionBinding
    created: bool


@dataclass(frozen=True, slots=True)
class LocalLineage:
    """One receipt's owner (its creation id) and its physical segments, as ``list_lineages`` reads them."""

    owner: SessionId
    lineage: tuple[SessionId, ...]


@dataclass(frozen=True, slots=True)
class BindingMarks:
    """Whether a creation receipt (``policy``) and a legacy binding (``legacy``) exist for an id, and, when asked and either exists, the lineage owner of ``lineage_of``."""

    policy: bool
    legacy: bool
    lineage_owner: Optional[SessionId] = None


@dataclass(frozen=True, slots=True)
class LegacyAdoptionScan:
    """The legacy local rows ``hermes sessions adopt`` walks: ``session_ids`` every row whose source is a legacy local source, by start time then id; ``marked`` every id holding a creation receipt or a legacy binding."""

    session_ids: tuple[SessionId, ...]
    marked: frozenset[str]


@dataclass(frozen=True, slots=True)
class NewApiBinding:
    """An API route binding.

    ``storage_source`` is ``api_server`` or ``bot_room`` (a ``bot_room`` row is hidden and must carry ``title``); ``declared_key`` the declared conversation key the request selected the session by; ``caller_key`` the request's validated session key, used only to adopt a row an earlier release stamped with it; ``origin`` the routing origin (the store reads ``user_id``); ``entry`` the routing entry (the store reads ``session_id``); ``receipt`` the binding receipt stored as JSON (``profile_id``, ``session_id``, ``route``, ``entry``, optional ``declared_key``, ``storage_source``, ``room_identity``, ``room_generation``); ``started_at`` the epoch seconds a freshly inserted row starts at.
    """

    session_id: SessionId
    route: str
    storage_source: str
    title: Optional[str]
    declared_key: Optional[str]
    caller_key: Optional[str]
    origin: Mapping[str, JsonValue]
    entry: Mapping[str, JsonValue]
    receipt: Mapping[str, JsonValue]
    started_at: float

    def __post_init__(self) -> None:
        _require_text(self, ("session_id", "route", "storage_source"))
        _require_text(self, ("title", "declared_key", "caller_key"), optional=True)
        if not isinstance(self.origin.get("user_id"), str):
            raise ValueError("NewApiBinding.origin['user_id'] must be a string")
        if self.entry.get("session_id") != self.session_id:
            raise ValueError("NewApiBinding.entry['session_id'] must equal session_id")
        _require_iso(self.entry.get("created_at"), "created_at", "NewApiBinding")
        _require_iso(self.entry.get("updated_at"), "updated_at", "NewApiBinding")
        if self.receipt.get("session_id") != self.session_id:
            raise ValueError("NewApiBinding.receipt['session_id'] must equal session_id")


@dataclass(frozen=True, slots=True)
class ApiBinding:
    """A stored API binding receipt, parsed: ``storage_source`` ``api_server`` when absent, ``room_generation`` 0 when absent."""

    profile_id: str
    session_id: SessionId
    route: str
    entry: Mapping[str, JsonValue]
    declared_key: Optional[str] = None
    storage_source: str = "api_server"
    room_identity: Optional[tuple[str, str, str, str]] = None
    room_generation: int = 0


@runtime_checkable
class SessionBindingRepository(Protocol):
    """Local creation receipts, API bindings and the lineage reads (module docstring); the runtime check is an assembly diagnostic only, never contract coverage."""

    def reserve(self, *, epoch: int, binding: NewSessionBinding, client_request: Mapping[str, JsonValue] | None = None, owner: OwnerRule | None) -> ReservedBinding:
        """Reserve the row, the route and the receipt in one write.

        Before the write: ``client_request`` None while ``policy['client_env_json']`` is set is ``invalid_params``; the receipt is encoded (an unencodable one is ``invalid_params``). In the write, in order: epoch; a stored receipt replays when profile, principal, request, session and route are equal and the request is the same create retry (``request_json`` as the client sent it and ``client_request``'s ``client_env`` under the frozen rules), else ``invalid_params`` (an undecodable stored receipt is ``storage_unavailable``); a retired id is ``not_found``; an existing row is ``storage_unavailable`` (not on the legacy arm); the route held in the routing index is ``admission_conflict``; the legacy arm binds the old row (``legacy_session_id`` must equal the id, the row must be unrouted legacy history of this profile (a local surface's, or a client app's under ``owner``) whose stamp the principal owns under ``owner``, not imported by another owner, without a compression child, idle, without a live turn lease, not yet bound; a client app's row is relabelled ``desktop``); then the row (inserted if absent, ``created_source`` stamped), the routing entry and the receipt. ``created`` is True only when this call inserted the receipt.
        """
        ...

    def adopt_untracked_compression(self, *, epoch: int, session_id: SessionId, expected: SessionBinding) -> None:
        """Append compression steps an older writer published past the receipt's tip, once: one write; nothing happens when the stored receipt is no longer ``expected`` or nothing is adoptable (every step must carry the receipt's route, owner and principal, and the route must still target the tip); otherwise the receipt, its routing entry and the logical revision move to the last step, validated first."""
        ...

    def reset_target(self, *, epoch: int, parent_session_id: SessionId, entry: Mapping[str, JsonValue]) -> None:
        """A local ``/new``: one write. A missing parent or one ended by compression is ``admission_conflict``; no receipt for its owner is ``storage_unavailable``; a receipt whose tip is not the parent is ``admission_conflict``; running or unknown work on the owner or parent refuses (``session_busy``, ``unknown_execution``). Then the child row ``entry['session_id']`` (``_reset_from`` the parent, the policy's source, model and cwd), the parent ended ``session_reset`` with the conversation generation advanced, the receipt advanced to the child and the owner's runtime generation bumped."""
        ...

    def end_idle(self, *, epoch: int, session_id: SessionId, target_id: SessionId, reason: str) -> int:
        """Stamp ``ended_at`` and ``reason`` on the open ``target_id`` only while ``session_id`` has no queued, started or unknown admission and neither id a live worker; returns the rows stamped."""
        ...

    def bind_api(self, *, epoch: int, binding: NewApiBinding, gate: GateRefusal | None = None, walked_from: SessionId | None = None) -> None:
        """Bind an API route in one write, in order: epoch; ``gate`` raised as ``AuthorityRefusal``; a retired id is ``not_found``; with ``walked_from`` every generation of ``api_generation_id(walked_from, n)`` before the id must still be retired (its sweep expiry is dropped), else ``chain_moved``; a stored binding of another storage source is ``permission_denied``, one with another declared key ``admission_conflict``, else nothing is written; the declared key held by another session is ``admission_conflict``, else reserved; a row of another source is ``permission_denied``; a ``bot_room`` title that differs or another session holds is ``admission_conflict``; a row on another session key adopts only the caller's declared or legacy key (registered, stamped into the receipt), else ``admission_conflict``; the route held by another session that has its own binding is ``admission_conflict``. Then the row (inserted if absent, hidden for ``bot_room``), its route columns, the routing entry and the receipt."""
        ...

    def release_handed_off_receipts(self, *, epoch: int, lanes: frozenset[str]) -> dict[str, tuple[str, str]]:
        """One write: epoch, then every receipt whose current target row now carries another route and a source in ``lanes`` is handed off: its earlier segments take the lane's identity, the conversation's title follows its head, and the receipt and its legacy binding are deleted. Returns owner to (row, source) for the receipts handed off."""
        ...

    def target_has_history(self, session_id: str) -> bool:
        """Whether ``session_id`` has a message, an admission, a worker execution or a child session."""
        ...

    def receipt(self, session_id: SessionId) -> SessionBinding:
        """The stored creation receipt; no receipt, an undecodable one or one that is not an object is ``storage_unavailable``. The identity is not checked against the addressed id."""
        ...

    def local_session_ids(self) -> list[SessionId]:
        """Every id holding a creation receipt, in store order."""
        ...

    def list_lineages(self) -> list[LocalLineage]:
        """Every receipt's owner and lineage; undecodable receipts, non-string owners and non-list lineages are skipped and non-string members dropped."""
        ...

    def binding_marks(self, session_id: SessionId, *, lineage_of: SessionId | None = None) -> BindingMarks:
        """Whether a creation receipt and a legacy binding exist for ``session_id``; with ``lineage_of`` and either present, also ``lineage_owner(lineage_of)``, in one read."""
        ...

    def lineage_owner(self, session_id: SessionId) -> SessionId:
        """The creation id whose receipt lineage holds the segment ``session_id`` (named by the row's ``chat_id``), else ``session_id``."""
        ...

    def conversation_ids(self, session_id: SessionId) -> list[SessionId]:
        """Every row of the local conversation ``session_id`` belongs to: its creation id first, then each lineage segment, deduplicated; ``[session_id]`` without a receipt lineage."""
        ...

    def validate_lineage(self, receipt: SessionBinding) -> SessionId:
        """Validate the passed receipt against the rows and return its tip: the id must be the local identity (or the legacy binding must match), the lineage non-empty, duplicate-free, from the root to ``entry['session_id']``, each hop a reset continuation or the compression chain step, and the tip without a chain step; anything else is ``storage_unavailable``."""
        ...

    def compression_adoptable(self, receipt: SessionBinding) -> bool:
        """Whether ``adopt_untracked_compression`` would adopt steps for this receipt."""
        ...

    def authorize_history(self, *, session_id: SessionId, profile_id: str, subject: str, may_create: bool, native_transport: bool | None = None, owner: OwnerRule | None) -> bool:
        """One read. ``native_transport`` None skips the claim check (the listing); a boolean runs the claim check (row missing or routed, record present, own-profile owner match under ``owner``, then the registry answer) with the gateway's answer, then the authorization: a missing or routed row is False; a history-control record names ``(profile_id, subject)`` under ``owner`` (True) or refuses ``permission_denied``; a row that is neither legacy history (a local surface's, or a client app's under ``owner``) nor an import is False; a stored owner the subject does not own under ``owner`` refuses; no ``may_create`` refuses; else True."""
        ...

    def api_binding(self, session_id: SessionId) -> ApiBinding | None:
        """The stored API binding receipt, parsed, or None."""
        ...

    def declared_session(self, declared_key: str) -> SessionId | None:
        """The session a declared conversation key is registered to, or None (the registry read only)."""
        ...

    def legacy_adoption_scan(self, owner: OwnerRule) -> LegacyAdoptionScan:
        """The adoption candidates and the ids already bound, in one read: the rows of a legacy local source, and every routeless row whose label ``not_client_labels(owner)`` does not refuse, by start time then id; the caller walks each candidate's tip and judges it itself."""
        ...

    def history_binding(self, session_id: SessionId) -> Optional[str]:
        """The stored import-binding text of ``session_id`` exactly as written (compact JSON ``[profile_id, subject]``), or None."""
        ...

    def handoff_head(self, session_id: SessionId) -> tuple[Optional[SessionId], SessionId]:
        """``(local owner or None, head row)`` a ``/handoff`` of ``session_id`` moves: the lineage owner and its receipt in one read; when the receipt names a current target, the owner and that target's compression tip; otherwise None and the tip of ``session_id``. A malformed receipt names no target; a missing row answers ``(None, session_id)``."""
        ...

    def boot_session_ids(self) -> list[SessionId]:
        """The receipt ids boot restores, in one read (gateway/session_local_recovery.py:366-371, :412-417): the receipt's target row is open, missing or ended by compression, or the receipt's id has a queued, started or unknown admission, or a stored viewer close. A receipt that does not decode names no target row, so it is kept. Order unspecified."""
        ...
