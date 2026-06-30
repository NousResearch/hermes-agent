# Live-tree disposition (Task 1, 2026-06-29)

Re-measured at execution start. The 4 changes flagged at spec-time (v0.1) had **already landed
as real fork commits** between spec and execution — fork advanced 220 → 222 ahead:

| Spec-time parked item | Disposition at execution |
|---|---|
| `toolsets.py` (mem0 resident) | LANDED — `d71fa4a8f fix(mem0): make mem0_remember RESIDENT in tools[] (#118)` |
| `tests/plugins/memory/test_mem0_remember.py` | LANDED — part of #116/#118 mem0 background-review wiring |
| `_bgr_mem0_proof.py` (scratch) | GONE — throwaway proof file, cleaned, never committed (correct) |
| `package-lock.json` | not those edits anymore — see below |

**Remaining live-tree dirt:** `package-lock.json` only — a 26-line deletion of `"peer": true`
markers. This is **incidental `npm install` churn** (an npm run in the live tree rewrote peer-dep
annotations), NOT deliberate work. Disposition: **DISCARD** (`git checkout -- package-lock.json`).
Rationale: (a) it's not deliberate, (b) the merge regenerates lockfiles in Task 11 anyway, (c) the
build worktree branches from `fork/main` which doesn't carry it.

**Frozen target confirmed:** `929dd9c0d` still valid (2026-06-29 12:09 CT). Behind-frozen = **1734**
(stable — that's the point of freezing). Fork ahead = **222** (was 220; +2 from the landed mem0
commits). Merge-base unchanged: `c6b0eb4`.

**Net:** no un-landed fork work to preserve beyond what's already on `fork/main`. INV-8 satisfied.
