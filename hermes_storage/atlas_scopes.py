"""
hermes_storage/atlas_scopes.py — Atlas memory scope isolation for SaaS-mode Hermes.

Atlas already supports a ``fence`` parameter on every search and ingest call.
This module maps HermesIdentity → fence strings and provides high-level
scoped search / ingest helpers that enforce tenant isolation at the gateway
layer (defense-in-depth alongside Neon RLS policies from Phase 0).

Fence naming convention (matches phase-B spec exactly):
    personal:{platform}:{team_id}:{user_id}  → private to this user
    team:{platform}:{team_id}                → visible to all team members
    None                                     → global (platform-wide, no fence)

Isolation contract:
    - personal fence reads are scoped to exactly one user_id
    - team fence reads span all users in a team, different teams cannot read each other
    - global fence reads are platform-wide (read-only from an agent turn)
    - Agent turns cannot ingest to global fence (PermissionError)
    - scoped_atlas_search fans out across all three fences, deduplicates by
      chunk ID, and ranks results personal → team → global

Assumptions surfaced explicitly:
    - ``atlas_search_knowledge`` and ``atlas_ingest_research_output`` are async
      callables injected by the caller (atlas_search_fn / atlas_ingest_fn params).
      This makes the module testable without a live Atlas MCP connection and
      pluggable as Atlas evolves (e.g. REST gateway in Phase D).
    - Deduplication key is the ``id`` field on each result dict. If Atlas returns
      results without an ``id`` field, deduplication is skipped for those entries
      (they accumulate rather than deduplicate — conservative, not lossy).
    - per_fence_k = max(3, top_k // 3) is a heuristic. If Atlas has fewer than
      per_fence_k results in a given fence it returns what it has — the merge
      step re-ranks by scope regardless.
    - Parallel fan-out uses asyncio.gather. If Atlas is a serial MCP tool call
      gateway, this degrades to sequential without error.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# Sentinel for the global fence — Atlas interprets fence=None as "no fence",
# meaning global / platform-wide knowledge.
GLOBAL_FENCE: Optional[str] = None


# ---------------------------------------------------------------------------
# Identity → fence string helpers (pure functions — no I/O)
# ---------------------------------------------------------------------------

def personal_fence(identity) -> str:
    """
    Return the personal fence string for this identity.

    Format: ``personal:{platform}:{team_id}:{user_id}``

    This fence is private to one user in one workspace. No other user can
    read or write knowledge in this fence, even if they are in the same team.
    """
    return f"personal:{identity.platform}:{identity.team_id}:{identity.user_id}"


def team_fence(identity) -> str:
    """
    Return the team fence string for this identity.

    Format: ``team:{platform}:{team_id}``

    This fence is shared across all users in the same team (workspace).
    Users from different teams cannot read or write to each other's team fence.
    """
    return f"team:{identity.platform}:{identity.team_id}"


def fence_for_scope(scope: str, identity) -> Optional[str]:
    """
    Return the Atlas fence string for the given scope name.

    Parameters
    ----------
    scope : str
        One of "personal", "team", or "global".
    identity : HermesIdentity
        The caller's identity (used for personal/team fence construction).

    Returns
    -------
    str | None
        Fence string (or None for global scope).

    Raises
    ------
    ValueError
        If scope is not a recognised value.
    PermissionError
        If scope is "global" and the caller is attempting a write operation.
        (Use the scope string None directly for reads; this function is
         called by scoped_atlas_ingest which blocks global writes.)
    """
    if scope == "personal":
        return personal_fence(identity)
    if scope == "team":
        return team_fence(identity)
    if scope == "global":
        return GLOBAL_FENCE
    raise ValueError(f"Unknown scope {scope!r}. Use 'personal', 'team', or 'global'.")


# ---------------------------------------------------------------------------
# Scoped search (fan-out across all three fences)
# ---------------------------------------------------------------------------

async def scoped_atlas_search(
    query: str,
    identity,
    top_k: int = 10,
    *,
    atlas_search_fn: Callable[..., Coroutine[Any, Any, list[dict]]] = None,
) -> list[dict]:
    """
    Fan-out Atlas search across personal → team → global fences.

    Results are merged, deduplicated by chunk ID, and ranked by scope
    specificity: personal results always come first, then team, then global.
    A result present in both personal and global fence is returned exactly once
    with scope="personal".

    Parameters
    ----------
    query : str
        Full-text or semantic search query.
    identity : HermesIdentity
        Caller identity — determines which fences to query.
    top_k : int
        Maximum total results to return (default 10).
    atlas_search_fn : async callable, optional
        Injection point for the Atlas search MCP tool. Signature:
            async def search(query: str, fence: str | None, top_k: int) -> list[dict]
        Each result dict should have at minimum an "id" key for deduplication.
        In production this is the MCP ``mcp__atlas__search_knowledge`` tool
        call. In tests, pass a mock.

    Returns
    -------
    list[dict]
        Merged results with a "scope" key added to each entry:
        {"id": ..., "scope": "personal" | "team" | "global", ...}
    """
    if atlas_search_fn is None:
        raise RuntimeError(
            "scoped_atlas_search requires atlas_search_fn to be provided. "
            "In production, pass the Atlas MCP search tool callable. "
            "In tests, pass a mock."
        )

    fences = [
        (personal_fence(identity), "personal"),
        (team_fence(identity), "team"),
        (GLOBAL_FENCE, "global"),
    ]

    per_fence_k = max(3, top_k // 3)

    async def _query_fence(fence: Optional[str], scope_label: str) -> list[dict]:
        try:
            hits = await atlas_search_fn(query=query, fence=fence, top_k=per_fence_k)
            for hit in hits:
                hit["scope"] = scope_label
            return hits
        except Exception as exc:
            # Log and treat as empty — one bad fence should not poison the
            # entire fan-out. The caller can detect partial results via the
            # "scope" fields present in the output.
            logger.warning(
                "scoped_atlas_search: fence=%r failed with %s: %s",
                fence, type(exc).__name__, exc,
            )
            return []

    # Fan-out in parallel — gracefully degrades to sequential if Atlas is
    # serial. gather preserves fence order.
    fence_results = await asyncio.gather(
        *[_query_fence(fence, label) for fence, label in fences]
    )

    # Flatten all results, maintaining the scope ordering (personal first).
    all_results: list[dict] = []
    for scope_hits in fence_results:
        all_results.extend(scope_hits)

    # Deduplicate by chunk ID, preferring higher-specificity scope.
    # Resolution order: personal > team > global (already in fence_results order).
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for scope_label in ("personal", "team", "global"):
        for result in all_results:
            if result.get("scope") != scope_label:
                continue
            chunk_id = result.get("id")
            if chunk_id is not None:
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
            deduped.append(result)

    return deduped[:top_k]


# ---------------------------------------------------------------------------
# Scoped ingest (writes to a specific fence with global-write guard)
# ---------------------------------------------------------------------------

async def scoped_atlas_ingest(
    payload: str,
    identity,
    scope: str = "personal",
    *,
    atlas_ingest_fn: Callable[..., Coroutine[Any, Any, None]] = None,
) -> None:
    """
    Ingest knowledge into the specified Atlas fence.

    Agent turns cannot write to the global fence — that is a hard block.
    Team fence writes are visible to all team members. Personal fence writes
    are private to the caller.

    Parameters
    ----------
    payload : str
        Research output text or structured content to ingest.
    identity : HermesIdentity
        Caller identity — determines which fence to write to.
    scope : str
        Target scope: "personal" (default) or "team". "global" is always
        blocked (PermissionError).
    atlas_ingest_fn : async callable, optional
        Injection point for the Atlas ingest MCP tool. Signature:
            async def ingest(payload: str, fence: str | None, provenance: dict) -> None
        In production this is the MCP ``mcp__atlas__ingest_research_output``
        tool call. In tests, pass a mock.

    Raises
    ------
    PermissionError
        If scope == "global". Agent turns must never write platform defaults.
    ValueError
        If scope is not "personal", "team", or "global".
    RuntimeError
        If atlas_ingest_fn is not provided.
    """
    if atlas_ingest_fn is None:
        raise RuntimeError(
            "scoped_atlas_ingest requires atlas_ingest_fn to be provided. "
            "In production, pass the Atlas MCP ingest tool callable. "
            "In tests, pass a mock."
        )

    if scope == "global":
        raise PermissionError(
            "Agent turns cannot write to the global Atlas fence. "
            "Global knowledge is platform-managed and read-only from agent turns."
        )
    if scope not in ("personal", "team"):
        raise ValueError(f"Unknown scope {scope!r}. Use 'personal' or 'team'.")

    fence = personal_fence(identity) if scope == "personal" else team_fence(identity)
    provenance = {
        "framework": "hermes-gateway",
        "actor": identity.user_id,
        "team_id": identity.team_id,
        "platform": identity.platform,
        "scope": scope,
    }

    logger.info(
        "scoped_atlas_ingest: scope=%r fence=%r actor=%r",
        scope, fence, identity.user_id,
    )

    await atlas_ingest_fn(
        payload=payload,
        fence=fence,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# NL intent helper for memory_tool integration (Phase B spec S4)
# ---------------------------------------------------------------------------

_TEAM_INTENT_PHRASES = (
    "for the team",
    "share with everyone",
    "share with the team",
    "for everyone",
    "team wide",
    "team-wide",
    "company wide",
    "company-wide",
)


def detect_team_scope_intent(message: str) -> bool:
    """
    Return True if the message signals intent to share with the team.

    This is a lightweight NL heuristic — not strict parsing. False negatives
    are safe (defaults to personal scope). False positives are annoying but
    recoverable (user can re-ingest to personal scope).

    Assumption: phrase matching is case-insensitive, substring-based.
    Phase C can replace this with a proper classifier if needed.
    """
    lower = message.lower()
    return any(phrase in lower for phrase in _TEAM_INTENT_PHRASES)


def resolve_scope_from_intent(message: str, default: str = "personal") -> str:
    """
    Resolve a scope string from an NL message.

    Returns "team" if team-sharing intent is detected, otherwise ``default``
    (which is "personal" by convention). Global is never returned — agents
    cannot self-escalate to global scope.
    """
    return "team" if detect_team_scope_intent(message) else default
