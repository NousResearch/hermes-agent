"""Session runtime authority: the store-wide epoch, receipted session mutations, deferred and chat picks, the frozen-policy refresh, retirement reads and the ledger-gated ends.

``SessionRuntimeRepository.mutate`` is one operation over the closed :data:`MutationCommand` union, one typed command per mutation operation. Each command's ``payload()`` is the operation's wire payload, so the receipt key and digest (:func:`mutation_receipt_key`) and the stored receipt are computed over that payload. :func:`parse_mutation` validates an operation and its payload and returns the typed command.

Owner checks arrive as data: ``live_guard`` is the gateway's runtime-coordination verdict, raised after the receipt replay, existence, revision and generation checks and the delete target walk; ``history`` is the history-control write, run inside the same transaction right after that point and before the mutation is applied.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from ..core.canonical import canonical_json
from ..core.errors import AuthorityRefusal
from ..core.identity import SessionId
from .admissions import GateRefusal
from .vocabulary import OwnerRule, admission_fingerprint

__all__ = [
    "COMPACTED_TARGET_DETAIL",
    "METADATA_FIELDS",
    "ApiGenerations",
    "Archive",
    "Branch",
    "ChatPickParams",
    "ClaimHistory",
    "Compress",
    "DeferredPick",
    "Delete",
    "DeleteFootprint",
    "EditPolicy",
    "EditorMcp",
    "GenerationChain",
    "GenerationPick",
    "GenerationRow",
    "HistoryActor",
    "HistoryWrite",
    "Import",
    "ImportControl",
    "LocalGenerations",
    "LocalSnapshot",
    "ModelPick",
    "MutationCommand",
    "MutationOutcome",
    "PinnedIdentity",
    "PreparedMutation",
    "ReasoningPick",
    "Rename",
    "Reset",
    "Rewind",
    "RoomGenerations",
    "SessionGone",
    "SessionRuntimeRepository",
    "SessionRuntimeState",
    "Sidebar",
    "StoredDeferredPick",
    "Workspace",
    "chat_pick_key",
    "mutation_receipt_key",
    "parse_mutation",
]

#: The sidebar metadata fields and their exact value types.
METADATA_FIELDS: Mapping[str, type] = {"title": str, "archived": bool, "hidden": bool, "pinned": bool, "unread": bool}

#: The rewind refusal's ``detail`` for a target compaction summarized; clients match the ``compacted:`` prefix.
COMPACTED_TARGET_DETAIL = "compacted: target user message was summarized when the session was compressed"

# The values of ``hermes_constants.VALID_REASONING_EFFORTS``.
_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_COMPRESS_FIELDS: Mapping[str, type] = {"focus": str, "preview": bool, "partial": bool, "keep_last": int}
# Operations whose mutation requires ``expected_generation``.
_GENERATION_REQUIRED = frozenset({"delete", "rewind", "reset", "branch", "model", "compress"})


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise AuthorityRefusal("invalid_params")
    return value


def _reasoning_level(value: str) -> bool:
    """A string level ``hermes_constants.parse_reasoning_effort`` accepts."""
    effort = value.strip().lower()
    return effort in {"none", "false", "disabled"} or effort in _REASONING_EFFORTS


def mutation_receipt_key(principal_id: str, session_id: str, request_id: str, *, operation: str, payload: Mapping[str, Any], expected_revision: int | None, expected_generation: int | None = None) -> tuple[str, str]:
    """``(state_meta key, digest)`` of a mutation request's exact-retry receipt."""
    key = "gateway.mutation.v1." + admission_fingerprint(canonical_target=session_id, payload={"principal": principal_id, "request": request_id})
    fingerprint: dict[str, Any] = {"operation": operation, "payload": payload, "expected_revision": expected_revision}
    if expected_generation is not None:
        fingerprint["expected_generation"] = expected_generation
    return key, admission_fingerprint(canonical_target=session_id, payload=fingerprint)


@dataclass(frozen=True, slots=True)
class ChatPickParams:
    """A viewer's pick on a messaging chat: the mutation request the receipt key names."""

    request_id: str
    operation: str
    payload: Mapping[str, Any]
    expected_revision: int
    expected_generation: int


def chat_pick_key(principal_id: str, session_id: str, params: ChatPickParams) -> tuple[str, str]:
    """``(key, digest)`` of the pick ``params`` on ``session_id``."""
    for value in (principal_id, session_id, params.request_id):
        _text(value)
    for fence in (params.expected_revision, params.expected_generation):
        if type(fence) is not int or fence < 0:
            raise AuthorityRefusal("invalid_params")
    payload = json.loads(canonical_json(params.payload))
    return mutation_receipt_key(principal_id, session_id, params.request_id, operation=params.operation, payload=payload, expected_revision=params.expected_revision, expected_generation=params.expected_generation)


class _Command:
    """One typed mutation command; ``payload()`` is the wire payload with absent keys left out."""

    __slots__ = ()
    operation: ClassVar[str]
    _keys: ClassVar[tuple[str, ...]] = ()

    def payload(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._keys if getattr(self, key) is not None}


@dataclass(frozen=True, slots=True)
class ModelPick(_Command):
    """``model``: a model (and provider, guard token, reasoning level picked with it)."""

    model: str
    provider: str | None = None
    confirm: str | None = None
    reasoning: str | None = None
    operation: ClassVar[str] = "model"
    _keys: ClassVar[tuple[str, ...]] = ("model", "provider", "confirm", "reasoning")


@dataclass(frozen=True, slots=True)
class Branch(_Command):
    """``branch``: an independent local route copying the active transcript through ``through_message_id``."""

    title: str | None = None
    through_message_id: int | None = None
    mcp_servers_fingerprint: str | None = None
    operation: ClassVar[str] = "branch"

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.title is not None:
            out["title"] = self.title
        if self.through_message_id is not None:
            out["through_message_id"] = self.through_message_id
        if self.mcp_servers_fingerprint is not None:
            out["mcp_servers"] = {"fingerprint": self.mcp_servers_fingerprint}
        return out


@dataclass(frozen=True, slots=True)
class ReasoningPick(_Command):
    """``reasoning``: the session's level; ``None`` returns it to the profile's."""

    effort: str | None
    operation: ClassVar[str] = "reasoning"

    def payload(self) -> dict[str, Any]:
        return {"effort": self.effort}


@dataclass(frozen=True, slots=True)
class Workspace(_Command):
    """``workspace``: move the conversation to ``cwd``."""

    cwd: str
    operation: ClassVar[str] = "workspace"
    _keys: ClassVar[tuple[str, ...]] = ("cwd",)


@dataclass(frozen=True, slots=True)
class EditorMcp(_Command):
    """``editor_mcp``: the editor's MCP servers, by fingerprint only."""

    fingerprint: str
    operation: ClassVar[str] = "editor_mcp"

    def payload(self) -> dict[str, Any]:
        return {"mcp_servers": {"fingerprint": self.fingerprint}}


@dataclass(frozen=True, slots=True)
class EditPolicy(_Command):
    """``edit_policy``: an ACP session's edit approval policy."""

    policy: str
    operation: ClassVar[str] = "edit_policy"
    _keys: ClassVar[tuple[str, ...]] = ("policy",)


@dataclass(frozen=True, slots=True)
class Compress(_Command):
    """``compress``: commit a prepared summary."""

    focus: str | None = None
    preview: bool | None = None
    partial: bool | None = None
    keep_last: int | None = None
    operation: ClassVar[str] = "compress"
    _keys: ClassVar[tuple[str, ...]] = ("focus", "preview", "partial", "keep_last")


@dataclass(frozen=True, slots=True)
class Delete(_Command):
    """``delete``: retire the session and every row of its conversation."""

    operation: ClassVar[str] = "delete"


@dataclass(frozen=True, slots=True)
class Reset(_Command):
    """``reset``: publish a fresh physical target for a local conversation."""

    operation: ClassVar[str] = "reset"


@dataclass(frozen=True, slots=True)
class Import(_Command):
    """``import``: import session and message projections; the first names the mutated session. A parent outside the payload is linked only when its history-control record names the child's owner under ``mutate``'s ``owner`` (whatever home path wrote it); any other parent is detached."""

    sessions: tuple[Mapping[str, Any], ...]
    operation: ClassVar[str] = "import"

    def payload(self) -> dict[str, Any]:
        return {"sessions": [dict(item) for item in self.sessions]}


@dataclass(frozen=True, slots=True)
class Rewind(_Command):
    """``rewind``: take back the transcript from the user row ``target_message_id``. The receipt's ``target_message`` content is the row's live ask with a withheld prompt's placeholder swapped back for its ``display_metadata`` ``withheld_prompt``; the stored rows keep the placeholder."""

    target_message_id: int
    preserve_compaction_handoff: bool | None = None
    operation: ClassVar[str] = "rewind"
    _keys: ClassVar[tuple[str, ...]] = ("target_message_id", "preserve_compaction_handoff")


@dataclass(frozen=True, slots=True)
class Rename(_Command):
    """``rename``: the user's title."""

    title: str
    operation: ClassVar[str] = "rename"
    _keys: ClassVar[tuple[str, ...]] = ("title",)


@dataclass(frozen=True, slots=True)
class Archive(_Command):
    """``archive``: archive or restore the conversation; archiving retires the canonical Bot Chat as ``SessionRepository.set_archived`` does."""

    archived: bool
    operation: ClassVar[str] = "archive"
    _keys: ClassVar[tuple[str, ...]] = ("archived",)


@dataclass(frozen=True, slots=True)
class Sidebar(_Command):
    """``sidebar``: any of the :data:`METADATA_FIELDS`; ``archived=True`` retires the canonical Bot Chat as ``SessionRepository.set_archived`` does."""

    title: str | None = None
    archived: bool | None = None
    hidden: bool | None = None
    pinned: bool | None = None
    unread: bool | None = None
    operation: ClassVar[str] = "sidebar"
    _keys: ClassVar[tuple[str, ...]] = ("title", "archived", "hidden", "pinned", "unread")


#: The closed command union of ``mutate`` and ``prepare_mutation``.
MutationCommand = ModelPick | Branch | ReasoningPick | Workspace | EditorMcp | EditPolicy | Compress | Delete | Reset | Import | Rewind | Rename | Archive | Sidebar


def _editor_fingerprint(value: Any) -> str | None:
    if isinstance(value, dict) and set(value) == {"fingerprint"} and isinstance(value["fingerprint"], str):
        return value["fingerprint"]
    return None


def _parse_model(operation: str, payload: dict[str, Any]) -> MutationCommand | None:
    if "model" in payload and not set(payload) - {"model", "provider", "confirm", "reasoning"} and all(isinstance(v, str) and v.strip() for v in payload.values()):
        if "reasoning" not in payload or _reasoning_level(payload["reasoning"]):
            return ModelPick(**payload)
    return None


def _parse_branch(operation: str, payload: dict[str, Any]) -> MutationCommand | None:
    if set(payload) - {"title", "through_message_id", "mcp_servers"}:
        return None
    fingerprint = _editor_fingerprint(payload["mcp_servers"]) if "mcp_servers" in payload else None
    if "mcp_servers" in payload and fingerprint is None:
        return None
    if "title" in payload and not isinstance(payload["title"], str):
        return None
    through = payload.get("through_message_id")
    if "through_message_id" in payload and not (type(through) is int and through > 0):
        return None
    return Branch(title=payload.get("title"), through_message_id=through, mcp_servers_fingerprint=fingerprint)


def _parse_policy(operation: str, payload: dict[str, Any]) -> MutationCommand | None:
    if operation == "reasoning" and set(payload) == {"effort"} and (payload["effort"] is None or (isinstance(payload["effort"], str) and _reasoning_level(payload["effort"]))):
        return ReasoningPick(payload["effort"])
    if operation == "workspace" and set(payload) == {"cwd"} and isinstance(payload["cwd"], str) and payload["cwd"].strip():
        return Workspace(payload["cwd"])
    if operation == "editor_mcp" and set(payload) == {"mcp_servers"} and (fingerprint := _editor_fingerprint(payload["mcp_servers"])) is not None:
        return EditorMcp(fingerprint)
    if operation == "edit_policy" and set(payload) == {"policy"} and isinstance(payload["policy"], str) and payload["policy"] in {"ask", "workspace_session", "session"}:
        return EditPolicy(payload["policy"])
    return None


def _parse_compress(operation: str, payload: dict[str, Any]) -> MutationCommand | None:
    if set(payload) - _COMPRESS_FIELDS.keys() or any(type(value) is not _COMPRESS_FIELDS[key] for key, value in payload.items()):
        return None
    if "keep_last" in payload and not (payload.get("partial") is True and payload["keep_last"] >= 1):
        return None
    return Compress(**payload)


def _parse_transcript(operation: str, payload: dict[str, Any]) -> MutationCommand | None:
    if operation in {"delete", "reset"} and not payload:
        return Delete() if operation == "delete" else Reset()
    if operation == "import" and set(payload) == {"sessions"} and isinstance(payload["sessions"], list):
        return Import(tuple(payload["sessions"]))
    target = payload.get("target_message_id")
    if operation == "rewind" and "target_message_id" in payload and not set(payload) - {"target_message_id", "preserve_compaction_handoff"} and type(target) is int and target > 0 and type(payload.get("preserve_compaction_handoff", False)) is bool:
        return Rewind(target, payload.get("preserve_compaction_handoff"))
    return None


def _parse_metadata(operation: str, payload: dict[str, Any]) -> MutationCommand:
    required = {"rename": {"title"}, "archive": {"archived"}}
    if operation == "sidebar":
        valid = bool(payload) and not set(payload) - METADATA_FIELDS.keys()
    else:
        valid = operation in required and set(payload) == required[operation]
    if not valid or any(type(value) is not METADATA_FIELDS[key] for key, value in payload.items()):
        raise AuthorityRefusal("invalid_params")
    if operation == "rename":
        return Rename(payload["title"])
    if operation == "archive":
        return Archive(payload["archived"])
    return Sidebar(**payload)


# The operations with their own payload rule; every other one is a metadata edit (``rename``, ``archive``, ``sidebar``) or refused.
_PARSERS = {
    "model": _parse_model, "branch": _parse_branch, "compress": _parse_compress,
    "reasoning": _parse_policy, "workspace": _parse_policy, "editor_mcp": _parse_policy, "edit_policy": _parse_policy,
    "delete": _parse_transcript, "reset": _parse_transcript, "import": _parse_transcript, "rewind": _parse_transcript,
}


def parse_mutation(operation: Any, payload: Any) -> MutationCommand:
    """Validate an operation and its payload and return the typed command: an unknown operation or an invalid payload is ``AuthorityRefusal('invalid_params')``."""
    if not isinstance(payload, dict) or not isinstance(operation, str):
        raise AuthorityRefusal("invalid_params")
    parser = _PARSERS.get(operation)
    parsed = parser(operation, payload) if parser is not None else None
    return parsed if parsed is not None else _parse_metadata(operation, payload)


@dataclass(frozen=True, slots=True)
class SessionRuntimeState:
    """A session row's runtime counters: ``runtime_revision`` and ``runtime_generation``."""

    revision: int
    generation: int


@dataclass(frozen=True, slots=True)
class LocalSnapshot:
    """The local snapshot: the local policy receipt, its physical target and that target's revision."""

    receipt: Mapping[str, Any]
    target: str
    target_revision: int


@dataclass(frozen=True, slots=True)
class PreparedMutation:
    """The owner's prepared commit for a policy, branch or compress mutation: the snapshot it was derived from and the prepared values under their keys (``policy``, ``git``, ``tools_changed``, ``mcp_servers``, ``in_place``, ``messages``, ``summary``, ``child``, ``request_id``, ``parent_editor_mcp_json``, ``editor_mcp_json``, ``toolsets``, ``config_secret_ref``)."""

    snapshot: LocalSnapshot
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionGone:
    """The ``on_session_delete`` event a committed mutation owes its plugins: the caller fires it after the commit."""

    session_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    """``mutate``'s answer: the receipt ``result`` (the stored dict), ``replayed`` for an exact retry, ``gone`` for a delete or a first archive."""

    result: dict[str, Any]
    replayed: bool
    gone: SessionGone | None


@dataclass(frozen=True, slots=True)
class HistoryActor:
    """The authenticated actor of a history-control write: ``login_label`` is the ``<provider>:<user>`` label the gateway derived from ``subject`` (None when it maps to none), ``may_create`` whether it holds ``session:create``."""

    profile_id: str
    subject: str
    login_label: str | None
    may_create: bool


@dataclass(frozen=True, slots=True)
class ClaimHistory:
    """Claim unbound legacy history for ``actor``."""

    actor: HistoryActor


@dataclass(frozen=True, slots=True)
class ImportControl:
    """Bind the import's not-yet-existing ``session_ids`` to ``actor``."""

    actor: HistoryActor
    session_ids: tuple[str, ...]


#: The history-control write ``mutate`` runs inside its transaction after the fence, revision and generation checks, before the mutation is applied.
HistoryWrite = ClaimHistory | ImportControl


@dataclass(frozen=True, slots=True)
class DeferredPick:
    """A model or reasoning pick refused only because a turn is running, kept for the next turn boundary. ``shown`` is the model a model pick's receipt names instead of the payload's (the configured default a queued reset adopts)."""

    operation: Literal["model", "reasoning"]
    principal_id: str
    request_id: str
    payload: Mapping[str, Any]
    shown: str | None = None


@dataclass(frozen=True, slots=True)
class StoredDeferredPick:
    """A session's deferred slot: ``pick`` is the stored value decoded (a dict: the model pick's fields, a ``reasoning`` entry, ``awaiting_confirm``), ``stored`` the raw value, the compare token."""

    pick: Mapping[str, Any]
    stored: str


@dataclass(frozen=True, slots=True)
class PinnedIdentity:
    """An exact retry's identity: a retired generation holding it is still its own."""

    principal_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class ApiGenerations:
    """The generations of an API session id (``api_generation_id``)."""

    session_id: str


@dataclass(frozen=True, slots=True)
class RoomGenerations:
    """The generations of a hosted room member's session (``room_session_id``)."""

    identity: tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class LocalGenerations:
    """The generations of a derived local session (``local_session_id`` over ``generation_request_id``)."""

    profile_id: str
    principal_id: str
    base_request_id: str


#: A derived session id chain ``first_generation`` walks.
GenerationChain = ApiGenerations | RoomGenerations | LocalGenerations


@dataclass(frozen=True, slots=True)
class GenerationRow:
    """The picked generation's session row, as far as the spent rule reads it."""

    title: str | None
    source: str | None
    archived: bool


@dataclass(frozen=True, slots=True)
class GenerationPick:
    """The first usable generation, its session id and its row (``None`` when no row exists)."""

    generation: int
    session_id: str
    row: GenerationRow | None


@dataclass(frozen=True, slots=True)
class DeleteFootprint:
    """What deleting a session reaches: ``targets`` are the rows ``mutate`` with ``Delete`` deletes (the id first, the rest sorted), ``children`` every row whose parent is one of them, in no defined order."""

    targets: tuple[SessionId, ...]
    children: tuple[SessionId, ...]


@runtime_checkable
class SessionRuntimeRepository(Protocol):
    """The runtime authority of the handle's profile; every verb is one transaction."""

    # Writes

    def begin_epoch(self, *, instance_id: str) -> int:
        """Advance the store's authority epoch (1 on first acquisition), record ``instance_id`` as holder, return it. The caller holds the profile ownership lock (documented, not enforced)."""
        ...

    def prepare_mutation(self, epoch: int, *, principal_id: str, session_id: str, request_id: str, expected_revision: int, command: MutationCommand, expected_generation: int | None = None, live_guard: GateRefusal | None = None) -> LocalSnapshot | MutationOutcome:
        """``mutate``'s checks up to ``live_guard``, then the local snapshot (``runtime_coordination_required`` without a policy receipt; ``require_target_advanceable`` on the session and its target); a ``Branch`` also requires both idle. An exact retry returns its stored ``MutationOutcome``."""
        ...

    def mutate(self, epoch: int, *, principal_id: str, session_id: str, request_id: str, expected_revision: int, command: MutationCommand, expected_generation: int | None = None, live_guard: GateRefusal | None = None, history: HistoryWrite | None = None, prepared: PreparedMutation | None = None, media_root: str | None = None, owner: OwnerRule | None) -> MutationOutcome:
        """Commit one closed mutation and its retry receipt.

        Before the write: identity texts, ``expected_revision``, the command (``parse_mutation`` of its payload), ``expected_generation`` (required for delete, rewind, reset, branch, model and compress), all ``invalid_params``. In the write: epoch; the receipt (another digest ``admission_conflict``; an exact retry returns it, ``replayed``); the session (``not_found``, except an import); revision (``revision_conflict``); generation (``stale_generation``); ``live_guard``; ``history``; the operation; every affected row's revision bumps; the receipt is stored. ``gone`` names the deleted ids, or the archived ones on a first archive. ``media_root`` is read only by ``Delete``: the folder whose hosted documents the retirement of the deleted sessions' admissions records, as ``SessionRepository.delete`` takes it; it is in no command's payload, so it never enters the receipt digest. ``owner`` is the owner rule ``history`` and the ``Delete`` walk read stored owners by (as ``SessionBindingRepository.authorize_history`` takes it); it is in no payload either.
        """
        ...

    def defer_model_pick(self, epoch: int, *, key: str, digest: str, session_id: str, pick: DeferredPick) -> dict[str, Any]:
        """Keep ``pick`` in the session's deferred slot with its receipt: epoch; an exact retry returns its receipt; the session (``not_found``); the slot merge (a reasoning pick drops the slot model pick's own level; a model pick keeps an older level only when it carries none); the receipt ``{session_id, operation, status: deferred, <picked>, revision, execution_generation}``, where a model pick's ``model`` is ``pick.shown`` when given, else the payload's."""
        ...

    def forget_deferred_pick(self, session_id: str, *, stored: str) -> None:
        """Delete the deferred slot while it still holds exactly ``stored``; no epoch fence."""
        ...

    def mark_deferred_pick_asked(self, session_id: str, *, stored: str, confirm: str) -> None:
        """Store the slot with ``awaiting_confirm = confirm`` while it still holds exactly ``stored``; no epoch fence."""
        ...

    def replace_deferred_pick(self, session_id: str, *, stored: str, pick: Mapping[str, Any]) -> str:
        """Store ``pick`` in canonical JSON as the slot while it still holds exactly ``stored``, and return that text whether or not it was stored: the caller's next compare token (gateway/session_deferred_pick.py:119-125). No epoch fence; an unencodable pick is ``invalid_params``."""
        ...

    def commit_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str, params: ChatPickParams, result: Mapping[str, Any]) -> dict[str, Any]:
        """Bump the mount's revision and store ``result`` plus that revision as the receipt, re-checking the fences; an exact retry returns its receipt."""
        ...

    def commit_refreshed_policy(self, epoch: int, *, session_id: str, old: Mapping[str, Any], new: Mapping[str, Any], generation: int) -> int:
        """The owner's config refresh of an idle session: epoch, the local snapshot, ``require_idle``; the stored policy must be ``old`` and ``runtime_generation`` ``generation`` (``revision_conflict``); ``new`` is written and the generation bumps; returns it."""
        ...

    def store_viewer_close(self, epoch: int, *, session_id: str, record: str) -> None:
        """Store the pending viewer close under the epoch fence."""
        ...

    def forget_viewer_close(self, epoch: int, *, session_id: str, record: str) -> bool:
        """Delete the stored close while it is exactly ``record``, under the epoch fence."""
        ...

    def end_unless_ledger_busy(self, session_id: str, *, session_key: str | None, end_reason: str) -> bool:
        """End the row unless the ledger holds work for the session, its key, its ``local-`` chat id or its lineage owner; True when ended now or already ended. No epoch fence."""
        ...

    def end_after_settlement(self, epoch: int, *, session_id: str, admission_id: str, generation: int, admission_epoch: int, reason: str) -> bool:
        """One conditional end: the epoch is current, ``runtime_generation`` is ``generation``, the admission is terminal on this session at ``(generation, admission_epoch)``, nothing on the session is pending, the row is open. Every miss is False."""
        ...

    # Reads

    def current_epoch(self) -> int | None:
        """The stored epoch, ``None`` before the first ``begin_epoch``."""
        ...

    def has_mutation_receipt(self, principal_id: str, session_id: str, request_id: str) -> bool:
        """Whether a mutation receipt exists for the identity, whatever its digest."""
        ...

    def deferred_pick(self, session_id: str) -> StoredDeferredPick | None:
        """The session's deferred slot."""
        ...

    def prepare_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str, params: ChatPickParams) -> dict[str, Any] | SessionRuntimeState:
        """The stored receipt of an exact retry, else the fenced mount's counters: epoch, receipt (``admission_conflict``), session (``not_found``), revision (``revision_conflict``), generation (``stale_generation``)."""
        ...

    def policy_picks(self, session_id: str) -> set[str]:
        """The policy operations (``model``, ``reasoning``, ``workspace``) the session's non-deferred receipts committed, its receipt's ``picks``, or its branch parent's picks below the branch's revision."""
        ...

    def retired_session(self, session_id: str) -> bool:
        """Whether the id carries a retirement marker."""
        ...

    def first_generation(self, chain: GenerationChain, *, pinned: PinnedIdentity | None = None, start: int = 0) -> GenerationPick:
        """From ``start``, the first generation whose id is not retired, or is retired and holds the ``pinned`` identity's tombstone, with its row's title, source and archived flag."""
        ...

    def expiring_retirements(self, session_ids: Sequence[str]) -> set[str]:
        """The ids whose own marker is a sweep's (it ages out)."""
        ...

    def viewer_close(self, session_id: str) -> str | None:
        """The stored close record."""
        ...

    def viewer_closes(self) -> list[tuple[str, str]]:
        """``(session_id, record)`` of every stored close."""
        ...

    def pending_ledger_status(self, session_key: str) -> str | None:
        """The status of a non-terminal admission targeted at ``session_key``, else of a non-terminal worker execution on it, else None, in one read (tui_gateway/session_lifecycle.py:142-143, :193-200). Which row answers when several of one kind match is unspecified."""
        ...

    def delete_footprint(self, session_id: SessionId, owner: OwnerRule | None) -> DeleteFootprint:
        """In one read, the target walk ``mutate`` with ``Delete`` runs under ``owner`` (the id, its local conversation's segments, its compression chain in both directions within one history owner, delegate subtrees) and every row whose parent is one of the targets (gateway/session_history_mount.py:59-66). A missing row's targets are the id alone. Refuses ``AuthorityRefusal('storage_unavailable')`` when the walk's local receipt lineage does not validate."""
        ...
