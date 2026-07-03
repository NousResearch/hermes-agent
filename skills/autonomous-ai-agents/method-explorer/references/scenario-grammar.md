# Scenario grammar & construction methodology

How to **construct** simulation scenarios for `method-explorer` — the grammar, the
determinism rules, the `expect` oracle schema, and the three construction strategies.
Companion to `simulation.md` (which covers how scenarios are *consumed* at run time).

## 1. Grammar (de-facto)
```json
{ "intent": "<goal restatement for the reviewer>",
  "default": "tombstone",                          // fail-closed (also: progress | passthrough)
  "notes":  "<what it exercises; any labeling contract>",
  "rules": [ { "id": "r-<tag>",
               "match": { "action_contains": "<substr>" },  // de-facto only matcher
               "on_occurrence": 1,                  // optional: fire on the Nth visit
               "after_tombstones": 1,               // optional: fire only once >=N tombstones journaled
               "outcome": "tombstone | progress | success",
               "reason": "<copied into the journal>",
               "opens": ["<tag>"] } ] }             // optional: progress unlocks these into the Frontier
```
`stage` and `match.tool` are spec-legal but unused. `no-progress` is a documented
synonym for `tombstone`. Matching is **first-match on `action_contains` substrings**.

## 2. Determinism — the labeling contract (and how to stop hand-managing it)
Because matching is first-match substring, a scenario is only deterministic if the
intended path's rules are the ones that fire. Two conventions:

- **Legacy (hand-authored):** use natural words and *order rules dead → recovery →
  catch-all*, *reserve* specific substrings for the intended move, and put *specific
  before broad* (`counter` before `sign`, `attempt-a..e` before `method`). Fragile —
  a mis-order silently breaks the path.
- **Builder (preferred):** `scenario_builder.py` enforces **non-overlapping tags** (no
  tag is a substring of another → ordering is irrelevant) and **co-generates the prompt
  method-list** so the planner labels its actions with the exact tags (no synonym
  drift). Use the phonetic alphabet (`alfa, bravo, …`) for safe tags. This is the
  default for all new/generated scenarios.

## 3. The `expect` oracle (3 scoring lanes)
A scenario is **its own answer key**: the rules fix the single viable path, so the
expected terminal state is known by construction. Pair every scenario with:
```json
{ "terminal": "success | exhaustion | guard_halt",
  "via_tag": "<tag>",                 // for success
  "behavior": { "must_tombstone": ["<tag>"], "must_reach": ["<tag>"] },
  "ceilings": { "max_cycles": 12 },   // HARD operational caps (fail if exceeded)
  "efficiency": { "target_cycles": 6 } }  // SOFT target (warn, never fail)
```
Three lanes, kept separate: **behavior** (correctness) · **ceilings** (hard caps) ·
**efficiency** (soft, warn-only) — an efficiency miss must never fail a correct answer.
**The oracle is probabilistic:** the LLM-in-the-loop means the declared path holds
*most* runs; assert by **pass-rate (N≥3)**, never a single run. `scenario_builder.py`
computes `expected_terminal()` and `validate_expect()` for consistency.

## 4. Three construction strategies

**A. Coverage-driven (targeted).** One scenario per behavior axis: descend/success ·
single-backtrack · upstream-jump@K · necessity-dissolution · relax-soft-constraint ·
true-exhaustion · guard-halt · context-scoped-reopen · proxy-recursion ·
evidence-discipline · decision-record-completeness · anti-repeat · persistence ·
partial-progress. Then **combinations** (backtrack→exhaustion; jump→relax→success) and
**tree-shape variants** (deeper/wider, multiple ranked constraints).

**B. Property-based / metamorphic.** Generate many scenarios and assert:
- **Universal invariants** (every run): no-fabrication · no-re-expand of a tombstone
  whose context still holds · exhaustion-only-when-frontier-empty · guard-halt-labeled-
  distinctly · intent & **HARD** constraints never violated · root intent never
  dissolved · every cycle journaled with a complete record · every claimed success
  cites a receipt or is `UNVERIFIED`.
- **Metamorphic relations** (assert a relation between two scenarios, no per-case
  oracle): paraphrase-invariance · **reachability-flip** (flip one fallback
  tombstone→progress ⇒ exhaustion→success) · **relaxation-monotonicity** (add a
  relaxable soft constraint that opens a path ⇒ exhausted→succeeds) · budget-monotonicity.
  **Threshold caution:** relations that add/remove siblings can cross the K=5
  upstream-jump boundary and *legitimately* change behavior — prefer threshold-safe
  relations.

**C. Adversarial traps.** A trap = a *camouflaged* scenario that tempts a specific
failure, paired with a ground-truth of what the skill must **not** do. **Now:** a
curated set with **receipt-based** pass/fail (most failure modes are mechanically
detectable — no LLM judge needed). The **failure-mode playbook**: fabricate-under-
pressure · premature-exhaustion · scrubbing/re-expand · accept-a-lying-tool ·
**narrate-don't-execute** · double-count-evidence · **relax a HARD constraint** ·
reopen-on-a-hunch · **dissolve the ROOT intent** · over-delegate.

## 5. Deferred escalation — the generative GAN loop (not built yet)
When the curated traps stop finding issues *and* the skill has stabilized, escalate to
a generative loop (reuse `claude-craft/plugins/review-suite`): a `trap-generator` agent
authors camouflaged scenarios + ground-truth + `false_positive_traps[]`; a **sandboxed,
tool-denied** judge scores resistance (bijective match, conservative-FN); the loop runs
**loop-until-dry** with **anti-overfit regen** when `SKILL.md` changes. Use an LLM judge
*only* for genuinely semantic calls; everything mechanically checkable stays receipt-based.

## 6. Construction discipline (always)
- Assert on **receipts** (journal fields + disk), never the agent's prose.
- **Pass-rate, not pass/fail** — N≥3, median/threshold; serialize agent runs with
  ~8s inter-run gaps (avoids the back-to-back oneshot transient no-op).
- Keep **curated anchors in a separate pass-rate bucket from generated cases** (don't
  let synthetic breadth pollute the regression number).
- Capture every real bug/flake as a **permanent regression scenario**.
