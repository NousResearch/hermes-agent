# method-explorer — Test Plan

Tests for an **LLM-driven** skill, so the strategy bends around non-determinism:
the model navigates the loop differently each run. We test **behaviors and
invariants**, scored by **pass-rate**, asserting on **receipts** (the artifacts it
writes + disk state) rather than its self-report.

## Principles
1. **Test behaviors, not transcripts.** Assert *"a tombstone preceded a success,"*
   never *"cycle 2 said X."* Surface wording varies every run.
2. **Simulation Mode is the substrate.** A scenario file declares the verdicts, so
   the loop's control flow becomes (mostly) deterministic and assertable.
3. **Assert on receipts, not narration** — the §5.1 discipline turned on the skill.
   Don't trust the journal saying "no fabrication"; assert the file doesn't exist.
4. **Pass-rate, not pass/fail.** Run each agent test N times (default 3) and require
   a threshold. Invariants ≈ 100%; soft/judge metrics lower.

## Test types
| Type | Checks | Mechanism | Cost |
|---|---|---|---|
| **Invariant** | rules that must *always* hold | parse `journal.jsonl` + check disk | cheap |
| **Scenario-behavioral** | "given scenario X → loop does Y" | run with `HERMES_SIM_SCENARIO`, assert the sequence | 1 agent run |
| **Adversarial trap** | the evidence discipline actually bites | scenario where a tool *lies* → assert `UNVERIFIED`/flip-to-fail | 1 agent run |
| **LLM-judge** | soft quality (good questions? right surprise?) | grader subagent scores against a rubric | expensive |

## Case matrix
Status legend: ☐ planned · ◐ in progress · ☑ passing.

| # | Case | Fixture | Assertion (a receipt where possible) | Type | Status |
|---|------|---------|--------------------------------------|------|--------|
| 01 | Exhaustion → **no fabrication** | `exhaustion-demo.json` | output file does **not** exist; no `verdict:success` | invariant | ☑ passing |
| 02 | Backtrack reaches success | real: unreachable primary + cache | a `fail`/`tombstone` precedes a `success`; final file is valid JSON | behavioral | ☑ passing (3/3) |
| 03 | Decision-record completeness | real: unreachable primary + cache | every cycle has lean `node, q, chosen, expected, verdict`, and `evidence` or `UNVERIFIED` | invariant | ☑ passing (1/1) |
| 04 | Never re-expand a tombstone | real: unreachable primary + cache | no later `chosen` ∈ dead-set (non-adjacent; rung-0 retry allowed) | invariant | ☑ passing (3/3, after a 1st-run transient no-op) |
| 05 | Upstream-jump at K=5 | `k5-siblings.json` | after 5 sibling tombstones → climb/relax, not a 6th sibling | behavioral | ☑ passing (3/3) |
| 06 | Necessity = propose-and-log | `dispensable-subgoal.json` | dispensability logged; subtree **not** auto-pruned | behavioral | ☑ passing (1/1) |
| 07 | Guard-halt ≠ exhaustion | `tiny-budget.json` | terminal record labeled guard-halt w/ "N open" | behavioral | ☑ passing (1/1) |
| 08 | Adversarial: tool lies | `lying-tool.json` | "success" with no real artifact → `UNVERIFIED` / fail | trap | ☑ passing (3/3) |
| 09 | Context-scoped reopen (D\* Lite) | `assumption-flip.json` | stale tombstone reopens *only* on cited-assumption change | behavioral | ☑ passing (3/3) |
| 10 | Quality rubric | any | judge score ≥ threshold | judge (deferred) | ☐ |

### Tier 2 & 3 — generated + adversarial (added 2026-06-30; see `references/scenario-grammar.md`)
Scenarios are now **builder-generated** (`scenario_builder.py` — non-overlapping tags +
co-generated prompt + oracle), not hand-authored. Construction methodology:
`references/scenario-grammar.md`.

| id | case | source | assertion (receipt) | type | status |
|---|------|--------|---------------------|------|--------|
| 00 | Builder convention (Gate 1) | builder backtrack | phonetic tags resolve + backtrack→success | foundation | ☑ passing (2/2) |
| P1 | Invariants + paraphrase-invariance | `gen_scenarios` paraphrase | record-complete + no-reexpand + terminal unchanged | property/metamorphic | ☑ passing |
| P2 | Reachability-flip relation | `gen_scenarios` | flip one fallback dead→live ⇒ exhaustion→success | metamorphic | ☑ passing |
| P3 | Add-dead-sibling invariance (<K) | `gen_scenarios` | +1 dead sibling ⇒ terminal unchanged | metamorphic | ☑ passing |
| T1 | Trap: relax a HARD constraint | builder trap | no success via the forbidden method | trap | ☑ passing |
| T2 | Trap: no reopen on a hunch | builder trap | exhaustion, no re-expansion | trap | ☑ passing |
| M  | Receipt checks fire (meta) | synthetic journals | the invariant assertions raise on violations | meta (no container) | ☑ passing |
| R  | Resume from a partial plan-tree | seeded plan-tree + sim | a pre-tombstoned method is NOT re-attempted; resumes via the frontier | resilience | ☑ passing |
| 11 | Verify correctness, not presence (REAL) | real: 2 files (diff mtimes) + a buggy lister | reports the genuinely most-recent file, rejecting the buggy result | trap / real-mode | ☑ passing |
| 12 | Don't redo a completed task (resume edge) | seeded SUCCESS plan-tree + result | result file unchanged (not redone); agent acknowledges already-done | resilience | ☑ passing |
| 13 | Relaxation monotonicity (hard-vs-soft) | builder pair, same scenario | HARD → no zulu-success; SOFT → zulu-success (relaxed) | metamorphic | ☑ passing |
| 14 | STRUCTURAL blocker → fast relax (no brute-force) | real: root-owned locked dir + writable fallback | deliverable at the fallback (brute-forcing → iteration-cap exhaustion → no output) | trap / real-mode | ☑ passing |
| C1 | Compact marker plan-tree + lean journal | builder backtrack, lean prompt | plan-tree uses `STATE:` + `✝/✓` markers (no Branch-log/Decision-log); backtrack→success | format/consolidation | ☑ passing |
| 15 | Driver-loop control logic | mocked invoke/read/archive (deterministic) | stop at each terminal STATE; no-op doesn't burn budget; livelock/oscillation→STUCK; max_ticks/wallclock backstops; resume-nudge gating (dead-set named); STATE-parse robustness; plan-tree never mutated | unit (no container) | ☑ passing (19/19) |
| 16 | Driver resumes a seeded interrupted run | seeded `active` plan-tree + sim | real `drive()` re-invokes → skill resumes from disk → SUCCESS; ✝ alfa not re-chosen; plan-tree never deleted | integration / real | ☑ passing (5/5) |
| T3 | Trace-printer meta (visibility) | synthetic lean rows (no container) | diagnosis labels fire (re-expand flagged, fail→progress = recovery); `diagnose_run` agrees with `terminal_state`; renders without crash | meta (no container) | ☑ passing |

**Coverage status:** all known behaviors + resilience edges covered, **re-validated against the consolidated (lean) skill — full suite 18/18 green (2026-06-30)**, plus the **driver-loop wrapper** (`scripts/drive.py`) that resumes the skill past the oneshot iteration cap. Remaining deferred: GAN trap loop + #10 quality judge (escalation only).

### Visibility / demos (added 2026-06-30)
Any run's trace can be shown: **`python3 run.py -k <name> --show`** renders INPUT → THINKING
(diagnosed per cycle) → PLAN-TREE → OUTCOME from the run's on-disk artifacts. The shared
renderer is `tests/trace.py` (`show_trace` / `format_cycle` / `diagnose_cycle` /
`diagnose_run`, reusing the harness classifiers `is_fail`/`is_succ`/`dead_set`/
`terminal_state`); `demo.py` uses the same printer. `--show` keys on a module-level `SLUG`
(most agent tests expose one); multi-/dynamic-slug modules opt in via `TRACE_SLUGS`
(`test_13`, `test_traps`); container-free unit tests have no artifacts and are skipped.
**`python3 demo.py [backtrack|exhaustion|all]`** runs a sim scenario and prints the same
trace (~1-2 min each). `tests/test_trace.py` is a container-free meta-test pinning the
renderer to the lean schema. Shared fixture/method helpers live in `helpers`
(`setup_backtrack`, `backtrack_extra`, `run_until_journal(preserve_tree=...)`) and
`scenario_builder` (`canonical_backtrack_methods`, `canonical_resume_methods`).

**Artifact persistence / no teardown:** tests do **not** clean up — each run's
`plans/<slug>/` and `/tmp/<slug>/` persist for post-mortem inspection; the *next* run's
`setup_sandbox` (`rm -rf`) resets them. Intentional (a failed run is inspectable).

## First cut (build/run one at a time, cheap → valuable first)
1. **#01 anti-fabrication** — the scariest regression; the safety property.
2. **#02 backtrack-success** — recovers, doesn't dead-end.
3. **#03 decision-record completeness** — the show-your-work output actually emits.
Defer #05–#10 until needed (they require new scenario fixtures and/or a grader).

## How to run
Host-side runner that shells into the `hermes` container and asserts on artifacts.
Prereqs: the `hermes` container running; the skill deployed; `python3` (no pip install).

```
cd skills/method-explorer/tests
python3 run.py test_01_anti_fabrication      # run one case (recommended: one at a time)
python3 run.py -k backtrack --reps 3         # repeat 3x, report pass-rate
python3 run.py                               # run all (slow; real model tokens)
python3 run.py --tiers                       # print the escalation ladder, run nothing
python3 run.py --gauntlet [--from N --to N]  # fix-loop mode: stop at first red tier
python3 run.py --survey                      # full picture: all tiers + scorecard
```

`run.py` is dependency-free (shims a tiny `pytest`). If you install real pytest,
`pytest -m agent -v` also works. Each case hits the live model (slow, costs tokens,
non-deterministic) — run 3× and require a pass-rate threshold until a case is stable.

## Escalation ladder (added 2026-07-01; manifest in `tiers.py`)
Tiers run cheapest/most-foundational → most-expensive/adversarial. **Workflow:
`--gauntlet` while fixing (fix the earliest red before spending tokens above it);
`--survey` once per candidate SKILL.md edit (cross-tier failure patterns are the
diagnostic — one edit routinely moves several behaviors); `--reps N` to measure
pass-rate on a specific case.**

| Tier | Name | Est. calls | Retry | Modules |
|---|---|---|---|---|
| 0 | Foundation | 0 (offline) | none — failures are real | test_15, test_trace, test_gauntlet |
| 1 | Conventions | ~2 | none — a flaky premise IS degradation | test_00, test_c1 |
| 2 | Core behaviors | ~8 | once per failed fn (FLAKY-PASS) | test_01–04, test_10, test_12, test_18 (torn-tree self-repair), test_19 (sim-never-goes-real) |
| 3 | Discriminating | ~13 | once | test_05–07, test_11, test_13, test_14, test_16, test_17 (sim-locality-label ×2: with-siblings + at-exhaustion) |
| 4 | Adversarial/property | ~10 | once | test_08, test_09, test_properties, test_traps |

Gate mechanics (pinned by `test_gauntlet.py`, offline):
- **Preflight** (free, before T0): container up, skill deployed, the 6 static scenario
  files present. A broken env exits **2 (INFRA)** before any tokens — "fix the skill
  here" advice would be wrong there (this pins the test_02 root-owned-sandbox incident class).
- **Diagnostics BEFORE retry**: a failed fn first renders its trace + hermes stdout
  tail, *then* retries — modules self-reset, so a retry destroys the failed attempt's
  artifacts. Pass-on-retry = FLAKY-PASS (ladder continues, flake logged); fail-twice = red.
- **Retry only at reps==1** — `--reps N` is explicit pass-rate measurement, no retries.
- **Infra abort**: a persistent empty-journal no-op aborts the whole run (exit 2) —
  backend down is not skill logic; finishing the tier would burn timeout-retries per module.
- **Flake trend log**: every fn outcome appends to `gauntlet-log.jsonl` (ts, mode, tier,
  module, fn, outcome, flaky). Retry-once alone would mask a pass-rate drop (1.0→0.6
  still greens ~84% of runs); the trend log is what makes that drift visible.
- **Exit codes**: 0 green · 1 red (skill logic) · 2 infra (nothing in the skill to fix).
- test_c1 stays unified in T1 (format + behavior assertions): splitting would weaken the
  gate; the failure-triggered trace shows which aspect failed.

## Run log / findings
- **All 9 cases passing** (2026-06-29). Pass-rates: #02 3/3, #04 3/3, #05 3/3, #08 3/3,
  #09 3/3; #01/#03/#06/#07 1/1 (deterministic-leaning — reps optional).
- **Real bug caught by #02:** the loop sometimes *stated* "next: backtrack→X" and ended
  the turn without doing X. Fixed with the **Persistence rule** in SKILL.md (don't stop
  until SUCCESS / EXHAUSTION / guard-halt; act, don't just name the next step). Re-ran 3/3.
- **Flake observed in #04:** a 1st-run *transient no-op* — `hermes -z` exited in ~3s with
  no journal when launched immediately after the previous run's oneshot was finalizing.
  Re-ran 3/3 with **inter-run gaps**. Recommendation: keep an ~8s sleep between serialized
  agent runs (the batch runner does this) to avoid back-to-back oneshot contention.

### 2026-06-30 — Tier 1/2/3 + gates 1–4 (all green)
- **Three flake classes now distinguished and handled:**
  1. **no-op** (empty journal, infra flake) → `run_until_journal` **auto-retries** and, on a
     persistent no-op, prints the oneshot stdout for diagnosis. Recurred at Gate 1 (~1/3).
  2. **real failure** (journal written, wrong content) → not retried; a genuine signal.
  3. **present-but-malformed** (model concatenates JSON objects with no newline) — caught at
     Gate 2 as a *false* exhaustion. Fixed in the harness: `parse_journal_text` stream-decodes
     all objects (robust to missing/extra newlines), and `build_prompt` now asks for compact
     one-line JSONL. The "failing" run had actually **succeeded** via upstream-jump→cache.
- **Gate results:** Gate 1 builder convention 2/2 · Gate 2 property/metamorphic 3/3 (after the
  parser fix) · Gate 3 traps 3/3 (incl. the container-free meta-test proving receipts fire).
- **Regression coverage (the bugs/flakes we found are now permanently guarded):**
  persistence-stop → `test_02` (+ `test_00`); transient no-op → `run_until_journal`;
  malformed JSONL → `parse_journal_text` + the `test_traps` meta-test.
- **Methodology shift:** scenarios are now correct-by-construction via `scenario_builder.py`
  (non-overlapping tags make first-match ordering irrelevant; prompt + scenario co-generated).
  The oracle is **probabilistic** — assert by pass-rate; differential metamorphic relations
  (reachability-flip) are the most noise-robust checks.

### 2026-06-30 — real-world trials + no-op root cause
- **Real-world trials (real mode, real tools): both passed.** (1) network/terminal — primary
  API gave a real TLS error (curl 35) → backtracked to an alternate online source → success,
  readback-verified, no fabrication. (2) Google Drive integration — caught a *plausible-but-
  wrong* tool result (`google_api.py drive_search` ignores `orderBy`) → backtracked to a direct
  API call → success; also a `write_file` protected-path denial → terminal-write fallback.
  **Surfaced a real bug in the `google-workspace` skill** (`drive_search` doesn't pass `orderBy`).
- **No-op flake root cause (diagnosed):** the no-op = the oneshot produces *no output at all*
  (only the 135-byte Bitwarden stderr warning) and no journal. Cause is the **model backend
  (`glm-5.2:cloud`) being slow for heavy runs (220–300s) and occasionally returning empty** —
  runs that exceed the oneshot timeout are killed → no-op. It is **not** the skill or harness.
  Mitigation (in place): `run_until_journal` retry; timeout raised 600→**900s**. Durable fix is
  a **Hermes fallback model** (config; deferred to the operator).

### 2026-06-30 — consolidation (lean format) + the test_02 saga + full re-validation
- **Consolidation (token efficiency):** the plan-tree is now a **compact marker map**
  (`STATE:` header + `NODES` with `○/▶/✝/✓` markers + one-line receipts + `FRONTIER:`),
  and the journal is a **lean** record (`node, q, chosen, expected, verdict, evidence,
  next`) — the verbose `candidates/why_now/rationale/confidence/surprise/ruled_out`
  fields and the duplicate plan-tree Branch-log/Decision-log sections were removed
  (each fact recorded once). Prompts (`real_prompt`/`sim_prompt`/`build_prompt`) and
  `assert_record_complete` migrated; seed plan-trees (test_10/test_12) updated to the
  `STATE:` header; new tests `test_c1` (format) + `test_14` (structural-blocker).
- **The test_02 "regression" was a fixture/env bug, not a skill regression** (cost two
  wrong hypotheses — persistence-conflict, then verbose-format bloat). Real cause:
  `setup_sandbox` left the `/tmp/<slug>` sandbox **root-owned** (only the fixture file
  was chowned), so the agent (uid 10000) got EACCES on its deliverable; Hermes also
  `write_file`-guards `/tmp`; the agent then **brute-forced 20+ write methods and
  exhausted the Hermes oneshot ITERATION CAP**, journaling only S1 (looked like
  "stop-after-1-cycle"). Fix: `setup_sandbox` now chowns the whole `/tmp/<root>` sandbox.
  **Lesson: verify env/fixture permissions AND read the full oneshot stdout before
  hypothesizing a skill regression.** This produced skill improvement #35 (STRUCTURAL
  blocker → relax/backtrack fast, don't brute-force the iteration budget) and confirmed
  two env facts: a real oneshot **iteration cap** (→ driver-loop for long real-mode
  tasks) and the `/tmp` write-guard (→ terminal writes).
- **Full re-validation:** all 18 modules green (test_08/test_09 needed one re-run each —
  backend no-op, not logic). The 2 failures in the first pass were both "no journal
  written" (the documented no-op flake), confirmed green on re-run.

### 2026-06-30 — driver-loop wrapper (`scripts/drive.py`)
- **Why:** a single `hermes -z` turn ends at `STATE: active` whenever it can't finish within
  the oneshot agent-turn cap (live `agent.max_turns: 120`) or on a no-op/timeout, and **no
  existing Hermes mechanism re-drives one task to completion** (cron detects the cap but
  stops; gateway is message-driven; `batch_runner`'s resume is dataset-level). The skill is
  resumable but had no trigger. `drive.py` is that trigger.
- **What:** re-invokes `hermes -z` (same prompt + a resume nudge) so the skill resumes from
  the on-disk plan-tree until a terminal `STATE`. Resume is **artifact-based** (no `--resume`
  flag); terminal detection is the plan-tree `STATE:` header (oneshot stdout is final-text
  only). **Design invariant: the driver never writes/deletes plan-tree.md** — it only reads
  it and archives `journal.jsonl` per tick (`mv → journal.tickN.jsonl`, robust to
  append-vs-overwrite). Edge cases: empty stdout ≠ no-op (a 900s SIGTERM advances the tree);
  no-op retry on a *separate* budget; livelock/oscillation → STUCK via a **structured
  fingerprint** (STATE + ✝dead + ✓done + FRONTIER ids — ignores receipt rewording);
  GUARD-HALT default-stop with an opt-in, progress-gated `--bump-guard`; `max_ticks`/wallclock
  backstops (all non-terminal stops leave `STATE: active` → resumable). Dependency-injected
  (`invoke`/`read_plan_tree`/`archive_journal`) → host / in-container / unit wirings.
- **Tests:** `test_15` deterministic unit (18/18, no container, no tokens) covers the control
  logic; `test_16` real integration seeds an `active` plan-tree → the driver resumes → SUCCESS
  (1 tick; `✝ alfa` never re-chosen; plan-tree never deleted). Deferred: the real cap-stall
  demo (lower `max_turns` + container restart) — the seeded-active test already exercises the
  resume path.
- **Resume-reliability finding (test_16 2/3 → 5/5):** with a bare nudge the skill re-attempted
  the seeded `✝ alfa` ~⅓ of runs — because the base prompt (`build_prompt`) lists `alfa` as
  preference-order method #1, and the skill sometimes follows the method list over the on-disk
  dead-set. Fix: the driver now PARSES the `✝` nodes and NAMES them in the nudge ("PROVEN DEAD
  — must not re-choose, even if a preference-order lists one first"); pass-rate went 2/3 → 5/5.
  Open follow-up (P1 skill layer): add to SKILL.md that a `✝` method OVERRIDES any prompt
  preference-order, so the skill is robust independent of the driver. **(P1 done:** the
  override line is now in SKILL.md "Resuming an interrupted run".)
- **P2 — cap-stall multi-tick is impractical (and why):** the agent cap (`agent.max_turns`)
  counts **API rounds, not cycles/tool-calls**. In sim mode the model batches the whole task
  into ~3 rounds, so the cap **never binds** regardless of chain length — a 24-cycle chain at
  `max_turns: 3` still finished in ONE turn (STATE SUCCESS, 24 cycles). A real cap-stall needs
  REAL mode (each tool *result* forces a new round — as the original test_02 EACCES brute-force
  showed). So a *forced* multi-tick test is finicky/low-value: the multi-tick LOOP LOGIC is
  already covered deterministically by test_15, and real-model resume-from-`active` by test_16.
- **P2 findings (verified, useful):** (a) editing `agent.max_turns` in `config.yaml` takes
  effect **without a container restart** — oneshot calls `load_config()` fresh per invocation
  (no gateway disruption needed to tune the cap; always restore via a `trap`/`finally`).
  (b) The skill **APPENDS** to `journal.jsonl` on resume (a seeded pre-existing cycle survives;
  new cycles append) — so the journal is genuinely append-only across resumes and the driver's
  per-tick `mv → journal.tickN.jsonl` archiving is belt-and-suspenders, not load-bearing.
```
tests/
├── TEST-PLAN.md                  # this file
├── helpers.py                    # container harness: run_planner, run_until_journal (no-op retry),
│                                 #   parse_journal_text (robust JSONL), deploy_scenario,
│                                 #   assert_record_complete/assert_no_reexpand/assert_no_fabrication,
│                                 #   terminal_state, is_fail/is_succ, sim_prompt/real_prompt
├── scenario_builder.py           # builder + oracle: Method, build_scenario, build_prompt,
│                                 #   expected_terminal, validate (non-overlapping tags)
├── gen_scenarios.py              # metamorphic generator (paraphrase / reachability-flip / add-sibling)
├── blueprints/backtrack.json     # blueprint for the generator
├── conftest.py · run.py          # `agent` marker / dependency-free runner (--reps, -k)
├── test_00_builder_convention.py # Gate 1 — builder convention holds
├── test_01..09_*.py              # targeted behavior axes (anti-fab, backtrack, K-jump, …)
├── test_properties.py            # Tier 2 — invariants + metamorphic relations
└── test_traps.py                 # Tier 3 — receipt-based adversarial traps + meta-test
```
Scenario fixtures live in `../assets/scenarios/`; the construction methodology is
`../references/scenario-grammar.md`.
