# Trial — Verdicts, Evidence, Termination

Reference for the `trial` skill (Gates 4, 6, 7).

## A verdict is evidence-bound

Every judge returns the `judge-verdict.md` shape:
- Lens / angle — which lens this is.
- Method used — exactly how it was checked.
- Evidence checked — commands run + output, files + lines read, criteria
  compared. Concrete and reproducible.
- Findings — what was found.
- Blocking issues — defects that prevent acceptance.
- Non-blocking issues — notes that don't block.
- Required fixes — precise, per finding.
- Verdict — one of: accepted / accepted-with-conditions / rejected.

A verdict with no "evidence checked" section is invalid — treat it as
"rejected, re-run". This single rule is what prevents the model from *performing*
approval instead of *doing* it.

## Blocking vs non-blocking

- Blocking: violates an acceptance criterion, breaks a real user path, loses or
  corrupts data, a security hole, a crash on normal input.
- Non-blocking: style, minor polish, a nice-to-have, a future refactor.

Only blocking issues trigger Gate-7 rework. Non-blocking issues are recorded in
`final-delivery.md` as known limits.

## Voting (multiple judges)

- standard (2 judges): both must reach accepted / accepted-with-conditions. Any
  reject -> rework.
- strict / maximum (3 judges): majority (>= 2) accept to pass; but any blocking
  finding from any judge must still be resolved or explicitly waived by the user.
- Disagreement is data: if judges split on whether something blocks, treat it as
  blocking until resolved.

## Termination (no infinite loops)

- Rework rounds are capped by mode (light 1, standard/strict 2, maximum 3).
- A fix that introduces a new blocking defect counts toward the cap.
- On reaching the cap with blocking issues still open, STOP and escalate to the
  user: present the open verdicts, the attempted fixes, and a recommendation.
  Never silently keep looping, and never silently ship with known blockers.

## Maximum-reasoning judges

For strict / maximum, judges reason adversarially and thoroughly. Ask the user
to set `delegation.reasoning_effort: high` so subagents think hard, and phrase
each judge's `goal` to "default to rejected when uncertain" — skepticism is the
job, not politeness.
