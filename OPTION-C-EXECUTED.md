# OPTION C executed — first clean extraction shipped (2026-06-22)

Per the Council's explicit directive ("execute OPTION C limited to a small, clearly clean
slice, e.g. the prefetch cap"), I extracted the cleanest self-contained contributable
feature from the residual into a real draft PR.

## Shipped: #50758 — feat(memory): cap prefetch query length before recall embedding
- The `_PREFETCH_QUERY_MAX_CHARS` memory-recall query cap, extracted from the entangled
  overlay (it originated in the `6658ed6fa` refusal+vision commit, intertwined with
  autopilot/refusal content) into a clean, self-contained PR off origin/main.
- Refactored into a pure helper `agent.turn_context._bound_prefetch_query(query)` so it's
  unit-testable; 6 tests (default cap, short-unchanged, env override, 0-disables,
  malformed-env-fallback, empty) — all green; ruff clean; **0 private-token leaks**.
- +79/-0, 2 files. Draft on fork arminanton.
- v0.17.0 resolution patch added (`v017-conflict-resolutions/agent_turn_context.py.v017.patch`,
  re-anchors to v0.17.0's `prefetch_all` call site at :392; git apply --check CLEAN).

## Full set re-verified WITH the new PR
- 40 code PRs (was 39) + #50111 manifest = 41 open.
- Pull-down PASS: **40/40 onto v0.17.0** (30 clean + 10 resolved) AND **40/40 onto current
  main** (40 clean), tree compiles. (`pull_down_onto.sh`, reproducible.)

## Honest scope of what OPTION C does and does NOT close
This proves the extraction PATH works and reduces the contributable residual by one
clean feature. It does NOT claim the *entire* ~1584-line residual is now extracted:
- The prefetch cap was the **cleanest, most self-contained** slice (a single helper, no
  surrounding private content).
- The REMAINING contributable residual (refusal-handling messages, async-fallback logging,
  and scattered clean lines in copilot/anthropic/limits files) is **intertwined with
  private content in the same hunks** (account caps, fable/opus effort tables, codex/agy
  infra). Extracting each requires the same per-hunk surgery as #50758 — feasible but
  multiplied across ~5-10 shared files, each a judgment call about whether the clean
  fraction justifies splitting it from the private content it's woven into.

## What remains a genuine user decision
OPTION C is now demonstrated, not just described. The question of HOW FAR to push it —
extract every clean line (many more surgical PRs) vs. accept the intertwined remainder as
overlay drift — is a real scope/effort trade-off. The cleanest slice is shipped; the
deeper extractions are queued behind that decision.
