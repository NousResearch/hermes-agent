# Phase B: Scoped Atlas Memory

**Status**: Complete
**Depends on**: Phase 0
**Blocks**: None

## Goal

Atlas knowledge is partitioned by scope using the existing **fence** mechanism. Personal knowledge stays private. Team knowledge is readable by all team members. Global knowledge is platform-wide (read-only to agent turns). Searches fan out across all three fences and merge results ranked by specificity.

## Context

Atlas already supports a `fence` parameter on every `ingest_research_output` and `search_knowledge` call. We map `HermesIdentity` scopes directly to fence strings — no Atlas code changes required. What changes is *which fence is used* and *how searches fan out* based on caller identity.

**Fence naming convention:**
```
personal:{platform}:{team_id}:{user_id}   → private to this user
team:{platform}:{team_id}                 → visible to all team members
None (fence=None)                         → global (platform-wide)
```

## Specifications

### S1: Identity → fence mapping

Pure functions in `hermes_storage/atlas_scopes.py` that convert a `HermesIdentity` to its three fence strings.

### S2: Scoped search (fan-out)

`scoped_atlas_search(query, identity, top_k)` — queries all three fences in parallel, merges results, deduplicates by chunk ID, ranks personal results first.

### S3: Scoped ingest

`scoped_atlas_ingest(payload, identity, scope)` — ingests to the specified scope's fence. Writing to `global` fence raises `PermissionError`.

### S4: Memory tool scope gate

The `memory` tool (personal notes) defaults to personal fence. If the user's message contains intent like "for the team" / "share with everyone", ingest to team fence instead. Intent detected at NL layer — no strict parsing.

### S5: Future — pgvector migration (Atlas)

Out of scope for this phase. Tracked in Atlas repo (army-of-one) as a separate plan. Atlas currently uses sqlite-vec; pgvector migration is a breaking change requiring its own plan.

## Steps

| # | Action | File | Expected Result |
|---|--------|------|-----------------|
| 1 | Create `hermes_storage/atlas_scopes.py` with fence helpers | `hermes_storage/atlas_scopes.py` | Functions return correct fence strings per identity |
| 2 | Implement `scoped_atlas_search` with parallel fence queries | `hermes_storage/atlas_scopes.py` | Fan-out across 3 fences, personal results ranked first |
| 3 | Implement `scoped_atlas_ingest` with scope gate | `hermes_storage/atlas_scopes.py` | global scope raises PermissionError |
| 4 | Write tests (mock Atlas calls, verify isolation) | `tests/test_atlas_scopes.py` | Personal knowledge not returned for different user_id |
| 5 | Wire `scoped_atlas_search` into agent's Atlas tool calls | `run_agent.py` or tool handler | Agent searches scoped when identity is present |
| 6 | Wire `scoped_atlas_ingest` into memory tool | `tools/memory_tool.py` | NL "for team" triggers team fence |
| 7 | Integration test: user A ingests personal fact; user B in same team cannot retrieve it | `tests/test_atlas_integration.py` | Isolation confirmed |
| 8 | Integration test: team-scoped fact visible to both users in same team | same | Sharing confirmed |
| 9 | Commit + push | git | `feat: phase-B scoped atlas memory` |

## Acceptance Criteria

- [x] `personal_fence(identity)` returns `"personal:{platform}:{team_id}:{user_id}"`
- [x] `team_fence(identity)` returns `"team:{platform}:{team_id}"`
- [x] `scoped_atlas_search` returns results from all three fences, personal ranked first
- [x] Personal result deduplicates: same chunk in personal + global returns only once (personal wins)
- [x] `scoped_atlas_ingest(..., scope="global")` raises `PermissionError`
- [x] User A's personal-fence knowledge is not retrievable by User B (different `user_id`, same `team_id`)
- [x] Team-fence knowledge is retrievable by both User A and User B in the same team
- [x] Existing Atlas calls without identity are unaffected (`fence=None` behavior unchanged)
- [x] `pytest tests/test_atlas_scopes.py -v` — all pass (32/32)

## Key Code

```python
# hermes_storage/atlas_scopes.py
from hermes_identity import HermesIdentity

def personal_fence(identity: HermesIdentity) -> str:
    return f"personal:{identity.platform}:{identity.team_id}:{identity.user_id}"

def team_fence(identity: HermesIdentity) -> str:
    return f"team:{identity.platform}:{identity.team_id}"

GLOBAL_FENCE = None

async def scoped_atlas_search(query: str, identity: HermesIdentity, top_k: int = 10) -> list[dict]:
    """Fan-out across personal → team → global fences, merge, rank personal first."""
    fences = [
        (personal_fence(identity), "personal"),
        (team_fence(identity), "team"),
        (GLOBAL_FENCE, "global"),
    ]
    results = []
    per_fence_k = max(3, top_k // 3)
    for fence, scope_label in fences:
        hits = await atlas_search_knowledge(query=query, fence=fence, top_k=per_fence_k)
        for hit in hits:
            hit["scope"] = scope_label
            results.append(hit)
    # Deduplicate — personal > team > global
    seen, deduped = set(), []
    for scope in ["personal", "team", "global"]:
        for r in results:
            if r["scope"] == scope and r["id"] not in seen:
                seen.add(r["id"])
                deduped.append(r)
    return deduped[:top_k]

async def scoped_atlas_ingest(payload: str, identity: HermesIdentity, scope: str = "personal") -> None:
    if scope == "global":
        raise PermissionError("Agent turns cannot write to global Atlas scope.")
    fence = personal_fence(identity) if scope == "personal" else team_fence(identity)
    await atlas_ingest_research_output(
        payload=payload,
        fence=fence,
        provenance={"framework": "hermes-gateway", "actor": identity.user_id}
    )
```
