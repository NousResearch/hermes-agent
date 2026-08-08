"""Production contract source adapter for the EvidencePackEngine.

The contract source is a metadata-backed view of the active
``GoalContract`` stored in ``SessionDB.state_meta`` under
``goal:<session_id>``. The canonical production data lives in
``GoalState.contract``; the adapter reads it through an injected
read-only state loader and projects the five contract fields
(outcome, verification, constraints, boundaries, stop_when) onto a
single ``KnowledgeHitV2``.

The adapter is deliberately self-contained:

* It does not import from ``hermes_cli`` (no ``GoalManager``,
  ``GoalState``, or ``GoalContract`` import).
* It accepts state through a narrow Protocol
  (``ContractStateLoader``) so a test can inject any object that
  exposes the documented structural contract (or ``None``) without
  importing the CLI module.
* It mutates nothing — neither the loader result nor the underlying
  contract dict. ``deepcopy`` is used internally before any read-side
  inspection so accidental attribute writes cannot leak back.
* It holds no external resources: no network, filesystem, HOME
  discovery, global cache, singleton, background thread, or process.
  ``close()`` is intentionally not defined — there is nothing to
  release.

The adapter matches the engine source contract::

    provider(query, *, max_hits, observed_at) -> list[KnowledgeHitV2]

It honors ``max_hits`` exactly as passed and clamps negative values
to zero (slice semantics). Loader exceptions are NOT swallowed: they
propagate so the engine's existing ``source_failed`` accounting can
observe them. The empty contract (all five fields blank) is a normal
``source-unavailable`` state and returns ``[]`` — never a fabricated
hit.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping, Optional, Protocol

from agent.executive.knowledge_discovery import (
    KnowledgeHitV2,
    KnowledgeQuery,
    SNIPPET_MAX_LEN,
    SOURCE_TTL_DAYS,
    TITLE_MAX_LEN,
    _clamp,
    _hit_fingerprint,
    _make_freshness,
    _make_provenance,
    _tokenize,
)


# Canonical SessionDB state_meta key where GoalState (and its
# GoalContract) is persisted per session_id. Tied to the goal key so
# the provenance URI stays anchored to production storage without ever
# revealing an absolute filesystem path.
STATE_META_GOAL_KEY = "goal:{session_id}"

# The five contract fields. Order matters: it is the canonical
# iteration order for snippet composition and hit_id derivation so
# repeated calls against the same contract yield stable ids.
CONTRACT_FIELDS: tuple[str, ...] = (
    "outcome",
    "verification",
    "constraints",
    "boundaries",
    "stop_when",
)

# Human-friendly labels for the snippet block. Rendered in the
# canonical field order so the snippet is byte-stable across
# invocations.
CONTRACT_FIELD_LABELS: Mapping[str, str] = {
    "outcome": "outcome",
    "verification": "verification",
    "constraints": "constraints",
    "boundaries": "boundaries",
    "stop_when": "stop_when",
}

# Schema-style URI scheme for the contract source. Anchored to the
# SessionDB state_meta key, never to a filesystem path.
SOURCE_URI_SCHEME = "state_meta"
SOURCE_URI_PREFIX = "state_meta[goal:"

# Provenance producer name. Distinct from the canary fake
# (``fake_contract_provider_v1``) so audit trails can tell production
# from fixture hits.
PRODUCER = "contract_provider_v1"


class ContractStateLoader(Protocol):
    """Read-only loader for the per-session goal state.

    The engine composition layer is responsible for wiring an
    implementation that reads ``SessionDB.state_meta[goal:<sid>]``
    and returns the deserialised ``GoalState`` (or a duck-typed
    object exposing the same attributes). Returning ``None`` is a
    normal "no goal for this session" outcome.

    The adapter never calls any other method on the loader and never
    expects the result to be the ``GoalState`` dataclass from
    ``hermes_cli.goals``; the loader is permitted to return any
    object that exposes ``contract`` as a ``Mapping[str, str]`` or
    object with the same five attributes.
    """

    def __call__(self, session_id: str) -> Optional[Any]:
        ...


def _normalize_contract(contract: Any) -> dict[str, str]:
    """Return a defensive copy of the contract as ``{field: text}``.

    Accepts either a Mapping (e.g. the on-disk JSON dict) or an
    object exposing the five contract attributes. Blank / non-text
    values are coerced to ``""``. ``None`` returns an empty dict.
    """
    if contract is None:
        return {field: "" for field in CONTRACT_FIELDS}
    out: dict[str, str] = {}
    if isinstance(contract, Mapping):
        for field in CONTRACT_FIELDS:
            value = contract.get(field)  # type: ignore[union-attr]
            out[field] = "" if value is None else str(value).strip()
    else:
        for field in CONTRACT_FIELDS:
            value = getattr(contract, field, None)
            out[field] = "" if value is None else str(value).strip()
    return out


def _is_empty_contract(contract: dict[str, str]) -> bool:
    return not any(contract.get(field, "") for field in CONTRACT_FIELDS)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def _build_source_uri(session_id: str) -> str:
    """Deterministic source URI tied to the SessionDB goal key.

    Returns a URI of the form
    ``state_meta[goal:<session_id>].contract``. Never exposes an
    absolute filesystem path.
    """
    return f"{SOURCE_URI_PREFIX}{session_id}].contract"


def _build_hit_id(session_id: str, contract: dict[str, str]) -> str:
    """Deterministic hit_id anchored to the session and contract body.

    Stable across repeated invocations against the same contract,
    but unique per session so distinct sessions produce distinct
    hits (and thus distinct fingerprints and conflict identities).
    """
    parts = [f"{field}={contract.get(field, '')}" for field in CONTRACT_FIELDS]
    body = "\n".join(parts)
    digest = _stable_hash(f"contract:{session_id}:{body}")
    return f"contract:{session_id}:{digest[:16]}"


def _build_snippet(contract: dict[str, str]) -> str:
    """Render the contract fields as a labelled snippet.

    Empty fields are omitted so the snippet stays focused on the
    parts the user actually filled in. Field order is canonical.
    """
    lines = []
    for field in CONTRACT_FIELDS:
        text = contract.get(field, "")
        if text:
            label = CONTRACT_FIELD_LABELS[field]
            lines.append(f"{label}: {text}")
    return "; ".join(lines)


def _score_against_query(query: KnowledgeQuery, contract: dict[str, str]) -> float:
    """Deterministic token-overlap score in [0.0, 1.0].

    Score is the Jaccard overlap between the query tokens and the
    contract tokens, clamped to [0.0, 1.0]. Returns a small floor
    when there is any contract content so populated contracts are
    always returned even when the query is token-empty (canonical
    session-discovery surface) — but a token-overlap of 0 with
    truly empty query tokens stays at 0.
    """
    contract_text = " ".join(
        text for text in contract.values() if text
    ).strip()
    if not contract_text:
        return 0.0
    q_tokens = _tokenize(query.objective_text or "")
    c_tokens = _tokenize(contract_text)
    if not q_tokens or not c_tokens:
        # Token-empty query → fall back to a fixed mid score so the
        # populated contract still surfaces as a candidate without
        # pretending it overlapped.
        return 0.5
    overlap = q_tokens & c_tokens
    union = q_tokens | c_tokens
    if not union:
        return 0.0
    return _clamp(len(overlap) / len(union))


def make_contract_provider(
    session_id: str,
    *,
    state_loader: ContractStateLoader,
) -> Any:
    """Build a contract source provider callable.

    Parameters
    ----------
    session_id
        The session whose goal/contract the provider reads. Stored
        on the closure so the loader is invoked with exactly this
        identifier — no other source of ``session_id`` is consulted.

    state_loader
        Injected read-only loader that returns the GoalState-like
        object for ``session_id`` (or ``None`` when no goal exists).
        Must never be ``None`` at call time: callers (the engine
        composition layer) are responsible for injecting a real
        loader. The provider does not import hermes_cli or fall
        back to a global / singleton loader.

    Returns
    -------
    A callable matching the engine source contract:
    ``(query, *, max_hits, observed_at) -> list[KnowledgeHitV2]``.

    The callable is stateless and side-effect free; it does not
    hold any resource that requires ``close()``.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    if state_loader is None:
        raise ValueError("state_loader must be an injected callable")

    def _provider(
        query: KnowledgeQuery,
        *,
        max_hits: int,
        observed_at: str,
    ) -> list[KnowledgeHitV2]:
        # Loader exceptions are NOT caught here — the engine relies
        # on them propagating to its ``source_failed`` accounting.
        loaded = state_loader(session_id)

        # Normalize / guard against an unexpected loader result.
        # Defensive deepcopy so any attribute access below cannot
        # mutate loader-owned state even by accident.
        state = copy.deepcopy(loaded) if loaded is not None else None
        if state is None:
            return []
        # Read contract through either attribute access (dataclass
        # / GoalState-like) or item access (raw mapping). The adapter
        # never inspects the state object's class — it only reads
        # ``state.contract`` and never the goal/other fields.
        if isinstance(state, Mapping):
            contract_obj = state.get("contract")
        else:
            contract_obj = getattr(state, "contract", None)
        contract = _normalize_contract(contract_obj)
        if _is_empty_contract(contract):
            return []

        snippet = _build_snippet(contract)
        # ``_build_snippet`` only emits non-empty fields and we have
        # already returned [] when every field is blank, so the
        # snippet is guaranteed to be a non-empty string here.
        hit_id = _build_hit_id(session_id, contract)
        source_uri = _build_source_uri(session_id)
        score = _score_against_query(query, contract)
        title = (
            f"Goal contract ({CONTRACT_FIELDS[0]}): "
            f"{(contract.get(CONTRACT_FIELDS[0]) or '')[:120]}"
        )
        if title.endswith(": "):
            title = "Goal contract (production)"

        # Production retrieval mode reflects metadata-backed state:
        # the contract is a structured completion contract that the
        # adapter reads from state_meta (a metadata-only backing).
        # We construct the hit manually instead of via _make_hit_v2
        # because _make_hit_v2 hardcodes a ``fake_*_provider_v1``
        # producer fallback that would mislabel our production hits.
        # All other dataclass helpers (provenance, freshness,
        # fingerprint, clamping) are reused directly.
        fingerprint = _hit_fingerprint("contract", hit_id, snippet)
        provenance = _make_provenance(
            "contract",
            source_uri,
            retrieval_mode="metadata_only",
            observed_at=observed_at,
            producer=PRODUCER,
        )
        freshness = _make_freshness(
            observed_at=observed_at,
            source_updated_at=observed_at,
            ttl_days=SOURCE_TTL_DAYS["contract"],
        )
        hit = KnowledgeHitV2(
            source="contract",
            hit_id=hit_id,
            title=title[:TITLE_MAX_LEN],
            relevance_score=_clamp(score),
            snippet=snippet[:SNIPPET_MAX_LEN],
            location=source_uri,
            fingerprint=fingerprint,
            created_at=observed_at,
            provenance=provenance,
            freshness=freshness,
            effective_score=0.0,
        )
        # Cap to max_hits (slice semantics). A session has at most
        # one GoalContract, so the adapter can never produce more
        # than one hit per call. ``max_hits <= 0`` returns [] (slice
        # [:0]/[:1] semantics; matches engine ``max_hits_per_source``
        # clamp behavior).
        if max_hits <= 0:
            return []
        return [hit]

    return _provider


__all__ = [
    "ContractStateLoader",
    "CONTRACT_FIELDS",
    "CONTRACT_FIELD_LABELS",
    "PRODUCER",
    "SOURCE_URI_PREFIX",
    "SOURCE_URI_SCHEME",
    "STATE_META_GOAL_KEY",
    "make_contract_provider",
]
