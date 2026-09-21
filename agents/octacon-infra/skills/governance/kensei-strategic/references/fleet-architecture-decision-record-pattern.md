# Fleet Architecture Decision Record Pattern

Use this pattern when Sahil asks to validate, revise, or implement a multi-agent fleet architecture decision.

## Core lesson

Do not jump from architecture analysis straight into runtime wiring. First turn the analysis into an execution-ready decision record and a read-only audit surface.

## Required sequence

1. Preserve the original direction and conclusions unless evidence contradicts them.
2. Correct overclaims:
   - broken items are open risks, not resolved
   - unverified live-state claims are gaps or follow-ups
   - profile counts must be checked, not repeated from prior analysis
3. Write an execution-ready decision record with:
   - decision
   - current state
   - open risks
   - implementation plan
   - ownership
   - verification steps
   - next actions
4. Prefer governance/reference files first.
5. Build read-only audit scripts before making config/profile changes.
6. Run incremental checks after each small batch.
7. Avoid service restarts, provider/auth/fallback edits, deletions, or broad config sweeps without Sahil approval.

## Fleet hygiene pattern

For multi-profile cleanup, classify before editing:

| Profile | Type | Owner lead | Verdict | Issues / notes |
|---|---|---|---|---|

Useful verdicts:

- KEEP
- KEEP/REVIEW
- FIX
- MERGE CANDIDATE
- RETIRE CANDIDATE

Do not delete/deactivate profiles during classification.

## Safe small-fix batch

If Sahil approves proceeding after classification, acceptable small fixes include:

- add missing SOUL.md files for clearly named sub-agent profiles
- add `kanban-worker` to clearly dispatchable worker profiles
- add missing profile-local skill files when the profile SOUL/config already depends on them

Still avoid:

- provider/auth/fallback changes
- service restarts
- profile deletion/deactivation
- broad unreviewed config rewrites

## Reporting style

When Sahil asks what has been done, answer in short bullets:

- whether KenseiAgent changed
- what completed
- what did not complete
- what needs approval

Do not bury the answer in architecture prose.
