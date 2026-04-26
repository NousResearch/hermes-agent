# NOTES

## Constraint surface
- Invariants:
  - Existing `mixture_of_agents` behavior must keep working.
  - Existing filesystem checkpoints and `/rollback` flows must remain backward-compatible.
  - New tools must be opt-in, not part of the default Hermes core toolsets.
  - No secrets in failure logs, eval artifacts, or campaign persistence.
- Perf bounds:
  - New snapshot/review helpers should stay bounded and avoid unbounded retry loops.
  - Spar review loops are capped at one fix round and one judge pass.
- API contracts:
  - Tool handlers return JSON strings.
  - `SessionDB` remains backward-compatible for existing session/message APIs.
- Data sensitivity class: internal
- Must NOT break:
  - `run_agent.py` tool execution loop
  - `tools/checkpoint_manager.py` checkpoint and restore semantics
  - gateway/ACP/TUI session persistence
  - focused MoA tests already shipped on this branch
- Escalation required: no

## Current objective
Implement the Hermes extraction set: ratchet snapshots, Spar review tool, judge disagreement, scars failure registry, campaign persistence, and proving matrix scaffolding.

## Task mode
invent

## Success condition
All six features exist as working code with focused tests, notes, decisions, eval scaffolding, and a final audit pass on local plus VPS sync readiness.

## Risk tier
high-blast-radius

## Evidence available
- Existing Hermes checkpoint/session infrastructure
- DomCode Spar + forensic source files
- Orchestrator/NIGHTSHIFT scars and campaign source files
- Existing Hermes tool registry and auxiliary routing code

## Likely files
- `tools/checkpoint_manager.py`
- `tools/spar_tool.py`
- `tools/registry.py`
- `toolsets.py`
- `agent/failure_registry.py`
- `agent/campaign_manager.py`
- `hermes_state.py`
- `run_agent.py`
- `tests/tools/test_ratchet.py`
- `tests/tools/test_spar_tool.py`
- `tests/test_campaigns.py`
- `scripts/proving_matrix.py`
- `EVALS/golden_set.md`
- `EVALS/rubric.yaml`

## Change-impact map
- Surfaces touched:
  - checkpoint infrastructure
  - tool registration and availability
  - session persistence schema
  - run loop failure breadcrumbs
  - eval tooling/docs
- Blast radius:
  - medium-to-high for checkpoint/session storage
  - low for new opt-in toolsets
- Verify:
  - checkpoint regression tests
  - new ratchet tests
  - new Spar/Judge tests
  - new campaign persistence tests
  - proving matrix dry-run

## Deletion opportunity?
- Reuse existing checkpoint/session infrastructure instead of creating parallel stores.
- Keep UI work out of scope.

## Branches considered
- A: Build entirely new snapshot/review subsystems outside Hermes primitives.
- B: Extend Hermes checkpoint/session infrastructure and add thin new modules.
- C: Overload MoA into Spar/Judge behavior.

## Rejected paths
- A rejected: too much architectural drag.
- C rejected: MoA and Spar solve different problems and should stay separate.

## Winning path
Extend existing Hermes primitives with small opt-in modules and thin integrations.

## Open risks
- Session-aware ratchet metadata may outpace current UI exposure.
- Run-loop failure capture can become noisy if the trigger is too broad.
- Campaign persistence needs to stay off by default to avoid surprise state growth.
- TODO on the next deletion pass: simplify ratchet helper sprawl in `tools/checkpoint_manager.py`; the current +310 line slice is correct but heavier than the intended budget.
- TODO on the next deletion pass: `acp_adapter/server.py` is now 1113 lines and the prompt handler is 187 lines; extract routing-mode dispatch helpers using the same helper-split pattern used for Spar.
- TODO for ACP auto-routing: the current heuristic is English-only and keyword-based; upgrade to a semantic classifier only after at least 20 real misroutes are observed.
- DONE: `~/.hermes/logs/route_forensics.jsonl` now rotates at 10MB and full raw MoA outputs are opt-in via `HERMES_MOA_FULL_FORENSICS=1`.
- TODO asymmetric refactor: extract shared route helpers first, then evaluate `agent/llm_contract.py` as the common route -> call -> parse -> repair -> forensic primitive.

## Perf measurements
- Ratchet tests: `59 passed`
- Spar + scars + campaigns focused tests: `12 passed`
- Combined focused suite: `71 passed`
- Proving matrix: `ratchet`, `spar`, and `campaigns` all approved locally

## Next validation step
Commit, push, and fast-forward the VPS checkout after one final `git diff --check` audit.

## Post-mortem 2026-04-26 — auto-titler race overwrote user rename
- **Trigger:** User renamed a freshly-created chat from Scarf's pencil button. Title flipped to the user's value briefly, then was overwritten with an LLM-generated title within seconds.
- **Symptom:** End-to-end smoke test of the new pencil button typed "Rename smoke test ✓"; a few seconds later the chat title in the list was "test 123" (LLM output, not user input).
- **Root cause:** `agent/title_generator.py:auto_title_session` checked `get_session_title` ONLY at the top of the worker thread. The `generate_title` LLM call below it takes 1–30s. Any title set during that window — Scarf pencil, `/title` slash, `hermes sessions rename` — was unconditionally overwritten when the LLM call returned.
- **Fix:** Re-check `get_session_title` between the LLM return and the `set_session_title` write. If a title now exists, skip. ~10 LoC.
- **Prevention:** Regression test `test_skips_if_user_set_title_during_generation` mocks `get_session_title.side_effect = [None, "User Set This"]` so the worker only writes when both the pre-check AND the post-check see no user title.
- **Considered alternative:** `UPDATE sessions SET title = ? WHERE id = ? AND (title IS NULL OR title = '')` is atomic but would also block intentional overwrites from CLI rename and `/title`. Worker-side guard is the smaller blast radius.
