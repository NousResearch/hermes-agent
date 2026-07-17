"""Owner-private, expiring, noncanonical Sol food proposal state.

A *proposal* is the transient presentation state between "Sol saw a
possible meal" and "the owner explicitly confirmed a commit". It is NOT
canonical food storage: Health owns the canonical commit (v3 contract);
this state is deleted on terminal state or expiry, and only value-free
consumed-action / update-to-receipt linkage survives (48 h) for replay
dedup.

Frozen semantics implemented here:

- One active proposal per single origin update/message. While a proposal
  is non-terminal, a second proposal for the same origin — or any new
  photo proposal for the same owner+thread — is refused.
- At most 4 candidate choices; at most 24 normalized items per candidate;
  at most 120 Unicode chars per UI label; canonical proposal JSON at most
  64 KiB; rendered display at most 3,500 chars.
- Lifetime: 30 minutes from *issuance*. Editing creates a new immutable
  candidate version INSIDE the same lifetime and immediately invalidates
  every earlier-version token.
- Explicit actions only: ``confirm``, ``edit``, ``cancel``, and bounded
  candidate choices (``choice:0``.. ``choice:3``). No Health write before
  ``confirm``.
- Callback validation checks owner, chat, thread, presentation message,
  candidate version/hash, action, and expiry BEFORE the action runs;
  consumption + update dedup are atomic and durable across restart.
- A duplicate delivery of the exact same callback update is transport
  replay: it returns the original receipt reference with no second
  effect. A NEW callback update against a consumed action is denied as
  already resolved and mints nothing.
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from plugins.sol_food.limits import (
    FOOD_CANDIDATE_MAX_CHOICES,
    FOOD_CANDIDATE_MAX_ITEMS,
    FOOD_DISPLAY_MAX_CHARS,
    FOOD_LABEL_MAX_CHARS,
    FOOD_PROPOSAL_JSON_MAX_BYTES,
    FOOD_PROPOSAL_TTL_SECONDS,
)
from plugins.sol_food.tokens import mint_token

__all__ = [
    "ProposalError",
    "ProposalState",
    "Candidate",
    "FoodProposal",
    "ACTION_CONFIRM",
    "ACTION_EDIT",
    "ACTION_CANCEL",
    "choice_action",
    "canonical_json_bytes",
    "candidates_hash",
    "validate_candidates",
    "render_display",
]

ACTION_CONFIRM = "confirm"
ACTION_EDIT = "edit"
ACTION_CANCEL = "cancel"

# Stable value-free reason codes.
REASON_TOO_MANY_CHOICES = "food_proposal_too_many_choices"
REASON_NO_CHOICES = "food_proposal_no_choices"
REASON_TOO_MANY_ITEMS = "food_proposal_too_many_items"
REASON_NO_ITEMS = "food_proposal_no_items"
REASON_LABEL_TOO_LONG = "food_proposal_label_too_long"
REASON_JSON_TOO_LARGE = "food_proposal_json_too_large"
REASON_BAD_CANDIDATE_SHAPE = "food_proposal_bad_candidate_shape"


class ProposalError(Exception):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class ProposalState(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        return self is not ProposalState.PENDING


@dataclass(frozen=True)
class Candidate:
    """One bounded candidate choice: a UI label + normalized items.

    Items are the transport shape of ``health.food_meal.v1`` entries:
    each item is exactly ``{"plant_key": str, "is_plant": bool}``.
    """

    label: str
    items: Tuple[Mapping[str, Any], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "items": [dict(i) for i in self.items]}


def _nfc_len(text: str) -> int:
    return len(unicodedata.normalize("NFC", text))


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical UTF-8 JSON: sorted keys, minimal separators."""
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def candidates_hash(candidates: Sequence[Candidate]) -> str:
    payload = [c.as_dict() for c in candidates]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_candidates(candidates: Sequence[Candidate]) -> None:
    """Enforce every frozen candidate ceiling; raise ProposalError."""
    if len(candidates) < 1:
        raise ProposalError(REASON_NO_CHOICES)
    if len(candidates) > FOOD_CANDIDATE_MAX_CHOICES:
        raise ProposalError(REASON_TOO_MANY_CHOICES)
    for candidate in candidates:
        if not isinstance(candidate.label, str) or not candidate.label:
            raise ProposalError(REASON_BAD_CANDIDATE_SHAPE)
        if _nfc_len(candidate.label) > FOOD_LABEL_MAX_CHARS:
            raise ProposalError(REASON_LABEL_TOO_LONG)
        if len(candidate.items) < 1:
            raise ProposalError(REASON_NO_ITEMS)
        if len(candidate.items) > FOOD_CANDIDATE_MAX_ITEMS:
            raise ProposalError(REASON_TOO_MANY_ITEMS)
        for item in candidate.items:
            if set(item.keys()) != {"plant_key", "is_plant"}:
                raise ProposalError(REASON_BAD_CANDIDATE_SHAPE)
            if not isinstance(item["plant_key"], str) or not item["plant_key"]:
                raise ProposalError(REASON_BAD_CANDIDATE_SHAPE)
            if not isinstance(item["is_plant"], bool):
                raise ProposalError(REASON_BAD_CANDIDATE_SHAPE)
    blob = canonical_json_bytes([c.as_dict() for c in candidates])
    if len(blob) > FOOD_PROPOSAL_JSON_MAX_BYTES:
        raise ProposalError(REASON_JSON_TOO_LARGE)


def render_display(candidates: Sequence[Candidate]) -> str:
    """Render the candidate list for Telegram, hard-capped at 3,500 chars.

    Truncation is per-whole-candidate: rather than cutting text mid-item,
    later candidates are dropped with a bounded continuation note.
    """
    lines: List[str] = []
    for index, candidate in enumerate(candidates):
        item_bits = ", ".join(str(i["plant_key"]) for i in candidate.items)
        lines.append(f"{index + 1}. {candidate.label} — {item_bits}")
    text = "\n".join(lines)
    if len(text) <= FOOD_DISPLAY_MAX_CHARS:
        return text
    note = "\n…"
    kept: List[str] = []
    used = len(note)
    for line in lines:
        needed = len(line) + (1 if kept else 0)
        if used + needed > FOOD_DISPLAY_MAX_CHARS:
            break
        kept.append(line)
        used += needed
    return ("\n".join(kept) + note)[:FOOD_DISPLAY_MAX_CHARS]


def choice_action(index: int) -> str:
    if not 0 <= index < FOOD_CANDIDATE_MAX_CHOICES:
        raise ProposalError(REASON_TOO_MANY_CHOICES)
    return f"choice:{index}"


@dataclass
class FoodProposal:
    """In-memory/serialized shape of one proposal. Managed by the store."""

    proposal_id: str
    # Frozen origin binding (of the owner's food message).
    bot_id: str
    owner_chat_id: str
    thread_id: int
    origin_update_id: int
    origin_message_id: int
    # Immutable-per-version candidate content.
    version: int
    candidates: List[Candidate]
    version_hash: str
    # Lifecycle.
    state: ProposalState
    issued_at: float
    expires_at: float
    # Presentation binding: the bot message that carries the buttons.
    presentation_message_id: Optional[int] = None
    # token -> {"action": str, "version": int, "consumed": bool}
    tokens: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Set after a confirmed Health commit: opaque receipt reference
    # (value-free — a hash/id, never food content).
    receipt_ref: Optional[str] = None
    # True from the moment the Confirm token is durably consumed until the
    # verified Health receipt is recorded. While set, the proposal must
    # NOT expire or be treated as re-confirmable: the frozen envelope +
    # reconcile() own completion.
    awaiting_commit: bool = False

    # ── serialization ───────────────────────────────────────────────────
    def to_json(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "bot_id": self.bot_id,
            "owner_chat_id": self.owner_chat_id,
            "thread_id": self.thread_id,
            "origin_update_id": self.origin_update_id,
            "origin_message_id": self.origin_message_id,
            "version": self.version,
            "candidates": [c.as_dict() for c in self.candidates],
            "version_hash": self.version_hash,
            "state": self.state.value,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "presentation_message_id": self.presentation_message_id,
            "tokens": self.tokens,
            "receipt_ref": self.receipt_ref,
            "awaiting_commit": self.awaiting_commit,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "FoodProposal":
        return cls(
            proposal_id=str(data["proposal_id"]),
            bot_id=str(data["bot_id"]),
            owner_chat_id=str(data["owner_chat_id"]),
            thread_id=int(data["thread_id"]),
            origin_update_id=int(data["origin_update_id"]),
            origin_message_id=int(data["origin_message_id"]),
            version=int(data["version"]),
            candidates=[
                Candidate(
                    label=str(c["label"]),
                    items=tuple(dict(i) for i in c["items"]),
                )
                for c in data["candidates"]
            ],
            version_hash=str(data["version_hash"]),
            state=ProposalState(data["state"]),
            issued_at=float(data["issued_at"]),
            expires_at=float(data["expires_at"]),
            presentation_message_id=(
                int(data["presentation_message_id"])
                if data.get("presentation_message_id") is not None
                else None
            ),
            tokens={str(k): dict(v) for k, v in dict(data.get("tokens", {})).items()},
            receipt_ref=(
                str(data["receipt_ref"]) if data.get("receipt_ref") is not None else None
            ),
            awaiting_commit=bool(data.get("awaiting_commit", False)),
        )

    # ── lifecycle helpers (invoked by the store under its lock) ─────────
    def expired(self, now: float) -> bool:
        return now >= self.expires_at

    def mint_action_tokens(self) -> Dict[str, str]:
        """Mint one token per explicit action for the CURRENT version.

        Returns {action: token}. Called on creation and on each edit
        (new version); the store invalidates prior-version tokens first.
        """
        actions = [choice_action(i) for i in range(len(self.candidates))]
        actions += [ACTION_CONFIRM, ACTION_EDIT, ACTION_CANCEL]
        minted: Dict[str, str] = {}
        for action in actions:
            token = mint_token()
            self.tokens[token] = {
                "action": action,
                "version": self.version,
                "consumed": False,
            }
            minted[action] = token
        return minted

    def invalidate_all_tokens(self) -> None:
        for record in self.tokens.values():
            record["consumed"] = True

    def scrub_content(self) -> None:
        """Delete candidate payloads/labels (terminal-state hygiene)."""
        self.candidates = []
        self.tokens = {token: rec for token, rec in self.tokens.items()}
        for rec in self.tokens.values():
            rec["consumed"] = True


def new_proposal_id() -> str:
    import secrets

    return secrets.token_hex(16)


def make_proposal(
    *,
    bot_id: str,
    owner_chat_id: str,
    thread_id: int,
    origin_update_id: int,
    origin_message_id: int,
    candidates: Sequence[Candidate],
    now: Optional[float] = None,
) -> FoodProposal:
    """Validate ceilings and build a fresh v1 proposal (not yet stored)."""
    validate_candidates(candidates)
    timestamp = time.time() if now is None else now
    return FoodProposal(
        proposal_id=new_proposal_id(),
        bot_id=bot_id,
        owner_chat_id=owner_chat_id,
        thread_id=thread_id,
        origin_update_id=origin_update_id,
        origin_message_id=origin_message_id,
        version=1,
        candidates=list(candidates),
        version_hash=candidates_hash(candidates),
        state=ProposalState.PENDING,
        issued_at=timestamp,
        expires_at=timestamp + FOOD_PROPOSAL_TTL_SECONDS,
    )
