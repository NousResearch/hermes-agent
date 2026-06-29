# Trial — The Eight Gates (detail)

Reference for the `trial` skill. The orchestrator reads this when it needs the
full procedure for a gate. Keep every artifact in the run dir
`.hermes/trial/<timestamp>-<slug>/`.

## Gate 1 — Mission framing (orchestrator)

Produce `mission-brief.md`. The hardest, most-skipped gate. Fill:
- Mission: one paragraph — the real job to be done, not the feature list.
- Non-goals: what this explicitly will NOT do.
- Constraints: tech, time, budget, platform, compliance.
- Acceptance criteria: testable bullets. "A trader can run 1,200 operations in a
  day with no data error" beats "works well".
- Rejection criteria: what makes the result unacceptable.
- Must-not-assume: unknowns that, if wrongly assumed, sink the build.

If any of these can't be filled, ask the user 1–3 short questions in their
language. Do not proceed on guesses.

## Gate 2 — Triple research (parallel batch)

Three independent lenses, each a leaf subagent that writes a findings file and
returns a 5-line summary. Pick the lens set by domain (matrix below). Generic
software/product set:
- `research/source.md` — external: market, competitors, prior art, best practice.
- `research/internal.md` — internal: existing code, constraints, what's there.
- `research/adversarial.md` — critical: what would make this plan fail? the
  strongest case against it.

Each findings file ends with: facts, risks, unknowns, one-line recommendation.

## Gate 3 — Decision brief (orchestrator)

Synthesize Gate 2 into `decision-brief.md`: proposed path, rejected
alternatives + why, risks, and the acceptance criteria the build must hit
(carried from the mission brief, sharpened by research). This is the contract the
builders are bound to — they may not invent a new goal.

## Gate 4 — Council approval (parallel batch)

Three judges review the decision brief, each from one angle:
- `council/strategy.md` — does this serve the real mission? right problem?
- `council/feasibility.md` — buildable without fragility? hidden complexity?
- `council/criteria.md` — are the acceptance criteria clear, complete, testable?

Each returns accepted / accepted-with-conditions / rejected + reasons. Proceed
only on accept; fold conditions into the build brief. On rejection, return to
Gate 3 (or Gate 1 if the mission itself is wrong).

## Gate 5 — Execution team (workers)

Spawn builders sized to the task — not fixed roles. Common roles: architecture,
backend, frontend, integration, docs, repair. Give each: the decision brief, its
exact slice, the run dir, and the boundary of what it owns. Builders use
`terminal`, `file`, `patch` (not `execute_code`). They write `build/<role>.md`
(what changed, where, assumptions, open issues) and return a summary. Keep
batches <= `delegation.max_concurrent_children`.

## Gate 6 — Triple judging (parallel batch)

Fresh judges — never a builder. Three DIFFERENT methods (not three reads):
- `judging/requirements.md` — match output line-by-line to acceptance criteria.
- `judging/execution.md` — actually run it: build, test, click the real path,
  hit the edge case (the 1,200-operation day, not a toy).
- `judging/adversarial.md` — try to break it: malformed input, concurrency, the
  unhappy path, the security angle.

Each writes a `judge-verdict.md`-shaped file with method, evidence, findings,
blocking vs non-blocking, and a verdict. See `verdicts.md`.

## Gate 7 — Targeted rework (loop, bounded)

Collect blocking findings. Route each to the responsible role only, with a
precise rework order in `rework/<n>-<role>.md`: the defect, the evidence, the fix
expected, the acceptance bar. Re-spawn that builder; then re-run the relevant
judge(s). Bound by the mode's rework rounds. On deadlock (judges keep
disagreeing, or a fix breaks another part), stop and escalate to the user with
the open verdicts — do not loop.

## Gate 8 — Final review (orchestrator, independent)

Do not trust summaries. Independently:
- re-run the key tests yourself (`terminal`),
- read the actual diff (`read_file` / `search_files`),
- check every acceptance criterion in `mission-brief.md` against real evidence,
- delete dead weight and contradictions.

Write `final-delivery.md`: each acceptance criterion -> its evidence, known
limits, and what was cut. Then report to the user in their language.

## The triple-lens matrix (pick by domain)

| Domain | Lens 1 | Lens 2 | Lens 3 |
|---|---|---|---|
| Research / plan | source (data) | logic (does it follow?) | adversarial (why wrong?) |
| Software | static read | run + test | does it do what the user needs? |
| Design | visual consistency | usability | does it sell / serve the goal? |
| Writing | meaning correct | voice right | structure serves the argument |

Three identical reviews catch the same bug three times. Three different lenses
catch three classes of bug. Always vary the lens.
