# Held Specification Ledger — Local Cursor/Hermes Snapshot

Generated: 2026-05-23
Scope: local workspace bootstrap for Hermes MCP `fleet_context_snapshot`
Source: `.learnings/memory.md` HOT-tier memory in this checkout

## Active Held / Blocked Items

- HELD: Architecture freeze active through the h20d checkpoint (~2026-05-26).
- HELD: Do not change enforcement logic, scoring, ranking, or production weights before freeze lift without explicit operator approval.
- HELD: Spec 100 (ranker IC tooling) is blocked by architecture freeze; implement post-May-26.
- HELD: Spec 095 indicates composite-score IC scope is invalid; old `composite_score` IC claims are invalidated.
- HELD: Use `final_score` as the corrected IC diagnostic target where applicable.
- HELD: Spec 088 Phase B pending Spec 087 full closure.
- HELD: KG Phase 2 Step 5 blocked until freeze lifts (June 1 target in HOT memory).
- HELD: Town-Hermes Feedback Protocol frozen until after h20d (2026-05-26).

## Active Governance Constraints

- MUST: Treat backtests as evidence only; production weights require governance approval.
- MUST: Tier 4 architecture/signal promotion changes require memo and human approval.
- MUST: Lane A agents must not depend on LLM gateway tokens.
- MUST: Only `crt_resolution_watcher` holds `mutate_data` authority.
- MUST NOT: Claim "true PIT" unless archived raw inputs, archived code, and archived artifacts all exist.

## Notes

This file is a local MCP held-spec bootstrap artifact. It is intentionally
conservative and mirrors HOT memory so `fleet_context_snapshot` can surface
governance flags locally. It does not replace canonical held-spec review
records or operator approval.
