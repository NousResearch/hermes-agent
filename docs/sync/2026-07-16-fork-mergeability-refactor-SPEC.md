# SPEC: Fork merge-friendliness refactor (extract fork-only logic behind stable seams)

**Author:** Apollo · **Date:** 2026-07-16 · **Status:** v0.3 (post Momus pass-2 AWC; RC-A/B/C folded)
**Motivation:** Two parity syncs (2026-07-15 #369 = 81 conflict files; 2026-07-16 #370 = 8)
showed conflicts concentrate where **fork-only logic lives inline in files upstream churns**.
The fix is structural: move fork logic OUT of high-churn upstream files into fork-owned
modules with 1-line call sites. A 1-line call site conflicts trivially (or not at all); a
200-line inline block conflicts semantically and must be re-read both-sides every sync.

**Non-goal:** changing ANY runtime behavior. This is a pure structure move. The entire value
is "smaller, more mechanical conflicts next sync" — if behavior changes, the refactor failed.

---

## The behavior-preservation problem (Ace's gate: "can we E2E verify no behavior change?")

**Do we have tests for this today?** Partially. The touched subsystems have unit suites
(`tests/gateway/`, `tests/agent/`, `tests/run_agent/`) but they do NOT constitute a
*characterization lock* — they test intended behavior, not "byte-identical before vs after a
refactor." A pure-move refactor needs a **differential/characterization harness** that proves
the extracted code produces identical outputs for identical inputs. We should BUILD it as the
first deliverable, because it's also reusable for every future extraction.

### Deliverable 0 (build FIRST): the refactor-equivalence harness

`scripts/refactor_equiv.py` — a golden-transcript differential runner:

1. **Capture (pre-refactor, on current `fork/main`):** for each target subsystem, run a fixed
   corpus of realistic inputs through the REAL entrypoint (not mocks) against a temp
   `HERMES_HOME`, and serialize every observable output — return values, emitted messages
   (`adapter.send.await_args_list`), DB row mutations (`state.db` transcript diff), and any
   status/telemetry side effects — to a golden JSON keyed by input hash.
2. **Verify (post-refactor, on the branch):** replay the identical corpus, serialize outputs
   the same way, and assert **byte-identical** to the golden. Any diff = a behavior change =
   the refactor is wrong, fail loud with the exact input + field that diverged.
3. **Determinism harness (CB1 — byte-identical must be achievable without ad-hoc scrubbing):**
   the runner OWNS nondeterminism at the source, never by post-hoc scrubbing:
   - frozen clock — the seam list covers ALL clock sources the targets use (RC-A):
     `time.time`/`datetime.now` AND `time.monotonic` (the cron-timeout idiom) via monkeypatch,
     AND the SQL layer: `state.db` columns using `DEFAULT CURRENT_TIMESTAMP`/`datetime('now')`
     are frozen at capture by an SQLite connection hook (`sqlite3.connect` wrapper registering
     a deterministic `datetime`/`strftime` override) — never punted to the allowlist. rowid/
     AUTOINCREMENT sequences are made deterministic by the fixed seed DB (same starting rowids
     both runs). Pre-flight per extraction: grep the moved code + touched schema for
     `monotonic|CURRENT_TIMESTAMP|datetime\('now'\)|AUTOINCREMENT`; every hit must map to a
     named seam, or the extraction doesn't start.
   - seeded IDs (monkeypatch `uuid4`/token mint to a counter),
   - fixed temp `HERMES_HOME` with a fixed seed DB,
   - a REVIEWED field-normalization allowlist (`equiv_normalize.py`, checked in) for the few
     fields that cannot be frozen (e.g. absolute tmp paths) — each allowlisted field carries a
     one-line justification; adding a field to the allowlist is a reviewed diff, not a runtime
     fallback. Any output field that differs and is NOT allowlisted = RED. No exceptions.
     The allowlist may NEVER contain a clock/sequence field — those are frozen at the seam
     (above); an allowlist entry matching `/time|date|_at$|_ts$|id$/` fails the harness's own
     self-lint.
4. **Coverage gate (RC3):** the corpus must achieve 100% BRANCH coverage
   (`coverage run --branch` + `--fail-under=100` scoped to the extracted module), or the golden
   is blind to a path. Branch, not line — an uncovered else-arm is exactly where a move bug hides.
5. **Mutation self-test — the detector must detect (CB2):** coverage proves lines RAN, not that
   their EFFECT is captured. Per extraction, the ritual includes a falsifiability step: mutate
   the extracted module (flip one branch condition / neuter one emit / drop one DB write — 3
   mutations minimum, one per output class: return-value, message-emit, DB-write) and assert the
   golden replay goes RED on each. A mutation the replay misses = the capture is incomplete →
   extend the serializer before proceeding. This proves capture completeness the same way we
   RED-prove regression tests.
6. **Enforcement is mechanical, not narrated (RC-B):** the corpus, goldens, normalization
   allowlist, and a scripted mutation harness (`refactor_equiv.py mutate --module <m>` —
   mutate → replay → assert-RED → revert, exit nonzero on any missed mutation) are COMMITTED
   with the extraction PR, and `golden-replay` + `mutation-selftest` run as blocking CI jobs on
   any PR touching a `fork_ext` module or its god-file call site. The PR body cites the CI run,
   not prose attestation.

This is the "prove it, don't assert it" gate: the refactor PR is GREEN only when the golden
replay is byte-identical AND all mutations go RED AND the pre-existing unit suites stay green
AND the full CI matrix is green. Four independent locks.

### Per-extraction proof ritual (every module move follows this)
1. Golden-capture the subsystem on `fork/main` BEFORE touching anything.
2. Extract: `git mv` the logic into `<subsystem>/fork_ext/*.py`; replace the inline block with
   a single call. NO logic edits — if you're tempted to "clean it up while here," STOP (that's
   a separate PR; mixing them destroys the equivalence proof).
3. Golden-replay on the branch → byte-identical.
4. Mutation self-test (Deliverable 0 step 5) → all mutations RED, then revert mutations.
5. Run the subsystem's unit suite + neighbors.
6. **Migrate `fork-features.json` paths in the SAME commit (RC2):** every manifest entry whose
   `paths` include the god-file being shrunk gets its path list updated to the new `fork_ext`
   module (the restart + compaction-announce entries key on `gateway/run.py` today). The item-2
   manifest linter (parity-tooling spec) then enforces this forever. Miss it and the manifest
   guards an emptied region — fake-green.
7. `git diff --stat` sanity (RC4, arithmetic fixed per RC-C): the unit of "extraction" is ONE
   `fork_ext` MODULE (a rank-row like gateway/run.py yields several modules, each its own
   check). NET addition budget = `10 (module header/docstring/imports) + 2 × call_sites`;
   exceeding it means logic crept in — reject and re-split. Call-site count is measured, not
   estimated (`grep -c 'fork_ext.<module>' <god-file>`).
8. **Import-order audit (Impl lens):** before landing, confirm the extracted code has no
   module-level state whose import TIMING matters (grep the moved block for module-level
   assignments/side effects; the god-files carry import-order-sensitive globals). A lazy
   call-site import changes when side effects fire; if any exist, hoist the import to the
   god-file's top, matching the original timing.

---

## Extraction targets (ROI order, measured by hunks-per-sync)

| Rank | Fork logic | Lives in (high-churn) | Move to | 07-15 hunks |
|---|---|---|---|---|
| 1 | model-switch announce, safe-restart hooks, denorm session list, undo/redo dispatch, config→env bridge | `gateway/run.py` | `gateway/fork_ext/{announce,restart,session_denorm}.py` | 27 |
| 2 | undo/redo stacks, denorm session columns, skew history, effective-last-active backfill | `hermes_state.py` | already partly in `hermes_undo.py`; extract denorm→`hermes_state_fork_denorm.py` | 24 |
| 3 | cron ContextVar/cron_mode, per-job reasoning, 7200s timeout | `cron/scheduler.py` | `cron/fork_ext/scheduler_ext.py` | 12 |
| 4 | LCM/compaction customization | `agent/context_compressor.py` | `agent/fork_ext/compaction_ext.py` | 9 |
| 5 | relay-pool lane headers, surrogate repair | `agent/chat_completion_helpers.py` | `agent/fork_ext/relay_headers.py` | 6 |
| 6 | block-seam chain, code_execution exemption | `agent/tool_executor.py` | `agent/fork_ext/tool_gate.py` | 6 |

**Precedent (verified via `git show <merge> --stat`):** `hermes_undo.py` — a standalone
fork-owned module — came through BOTH parity merges (07-15 #369 = 81 conflict files, 07-16
#370) with **zero conflicts**, while its consumer god-files conflicted heavily. That's the
target state for everything above. (Honest metric per the DevOps lens: extraction shrinks
conflict SIZE — hunks in the god-file — not necessarily conflict COUNT; the 1-line call sites
still live in churned files. Success criterion for the next sync: rank-1's hunk count drops
from 27 to single digits, measured by the same conflict-buckets report.)

### The seam discipline
- Each `fork_ext` module exposes a small, stable function surface. The god-file call site is
  `from gateway.fork_ext.announce import emit_model_switch_announce; emit_model_switch_announce(...)`.
- Where a fork block MUST stay inline, wrap it in a **FORK-FENCE sentinel comment**
  (`# ── FORK-EXT: <name> ──`). **The fence is a FLAG, not an auto-resolution (CB3):** it tells
  the resolver "fork intent lives here — read BOTH sides before deciding," never "keep-fork
  blindly." A blanket keep-fork over a region upstream actively churns would silently drop
  upstream bug/security fixes every sync. The resolution-spec wording the fence earns is:
  "conflicts inside a FORK-FENCE: preserve the fork behavior AND fold in upstream's changes to
  the surrounding logic; if the two are incompatible, flag for the orchestrator — never blind
  side-pick EITHER way." We already fence `toolsets.py` this way; formalize it.
- `AUTHOR_MAP` in `scripts/release.py`: move all fork entries to a fenced `# ── FORK-LOCAL ──`
  block at the END of the dict. Both syncs conflicted there because both sides append at the
  same tail; a fenced tail block conflicts as a clean union.

## Upstream-PR opportunity (free conflict reduction)
Upstream's own rubric *rewards* god-file extraction ("Refactor god-files into clean modules …
merge regularly"). Some extractions (the ones that aren't fork-specific behavior, e.g. a
generic `gateway/run.py` → mixin split) could be **PR'd upstream**, shrinking our diff from
BOTH sides permanently. Sequence: extract locally first (prove equivalence), then offer the
non-fork-specific structural moves upstream as neutral refactors.

## Rollout
- One PR per subsystem (independently reviewable/revertable), each carrying its golden-replay
  + mutation-RED proof in the PR body.
- **Harness-shakedown order ≠ conflict-ROI order (RC1):** build + prove Deliverable 0 on the
  CHEAPEST target first — rank-5 `relay_headers` (pure function surface, easy corpus) — so the
  harness's own bugs surface on a subsystem where a miss is recoverable. Then rank-6, then
  descend the ROI table toward rank-1 `gateway/run.py` (hardest corpus: async adapters,
  lifecycle I/O) with a battle-tested harness.
- Re-measure conflict surface on the NEXT sync to validate the payoff (metric above).
- Register each extracted module's behavior in `docs/sync/fork-features.json` so the manifest
  gate guards it.

## Risk & mitigation
- **Risk:** a "pure move" silently changes behavior (a closure captured a god-file local; an
  import cycle). **Mitigation:** the golden-replay harness is exactly the detector; coverage
  gate ensures no branch is unproven.
- **Risk:** extraction churns the file enough that THIS sync's diff is large. **Mitigation:**
  it's a one-time cost that pays back every subsequent sync; and the diff is mechanical (moves),
  which upstream merges cleanly.
- **Risk:** import cycles (`fork_ext` needs the god-file's symbols). **Mitigation:** pass
  dependencies as function args, don't import back — the seam is a function boundary, not a
  shared-module boundary.
