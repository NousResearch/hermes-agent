# Outside-Family Secondary Dialectic Review (Claude Opus 4.7)

Findings:

- **M-1 — `pr_url` JSON key is the noisiest signal in the ownership hint alternation.** In real handoff JSON, `pr_url` / `pull_request_url` may describe upstream or evidence PRs, not a task-owned PR. Add a negative test and avoid treating bare `pr_url` as ownership proof.
- **M-2 — Narrowing is English-only and could over-fix.** Legitimate human comments may use phrasings not covered by the regex, but this fails in the less harmful direction: duplicate-PR prevention may be weaker, while dispatcher wedging is avoided. Canonical `PR created: <url>` remains covered.
- **L-1 — Cross-clause matching.** URL and ownership hint currently search the whole body independently; a long mixed comment can still be ambiguous.
- **L-2 — No diagnostic for dropped non-owned PR references.** Low priority.
- **L-3 — Spawn-positive test could assert no guard more tightly.** Low priority.

Strongest case for acceptance:

- Direction is correct: moving from “any PR URL” to “PR URL plus ownership language” fixes the operator-hostile wedge.
- Canonical worker path `PR created: <url>` remains guarded.
- Tests reproduce the user-reported upstream/reference PR failure at both direct guard and dispatcher levels.

Strongest case against acceptance:

- Regex-on-prose is still a heuristic. Longer-term, task-owned PRs should be structured state or tagged events, not inferred from free-form comments.
- Bare `pr_url`/`pull_request_url` keys are ambiguous and should not be sufficient ownership evidence.

OUTSIDE_FAMILY_SECONDARY_VERDICT: APPROVE

Host resolution:

- Accepted the `pr_url` critique.
- Added a negative regression test for bare upstream `pr_url` JSON.
- Removed ambiguous bare `pr_url` and `pull_request_url` from the ownership-hint regex; retained explicit `created_pr`, `opened_pr`, `submitted_pr`, and `task_pr` keys.

CLAUDE_SUBAGENT_DONE
