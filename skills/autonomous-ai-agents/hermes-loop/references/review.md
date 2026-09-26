# Reviewer procedure

Load `references/protocol.md` first.

## Job

Review one build handoff against the packet at one full git SHA. Produce a verdict. Do not fix code in this role for Hermes-loop v1.

## Steps

1. Read the linked packet (root) and build handoff. Confirm packet version match.
2. Check out or diff the exact full SHA named in the handoff. If HEAD moved, review the named SHA or demand a fresh handoff — never rubber-stamp a moving target.
3. Review only against the packet: AC gaps, defects, security, scope expansion, missing states, maintainability for future agents.
4. Check CI: required checks passed/failed/pending/not configured. Pending → no terminal verdict yet. Not configured → needs-human-review, never approve-evidence.
5. Post `templates/review-verdict.md` on the review task (and optionally as a PR comment first line: `Hermes-loop review of <full SHA>`).
6. Complete the review task with the verdict summary.

Must-fix tags when useful: `[AC-N]`, `[DEFECT]`, `[SECURITY]`, `[CI]`, `[SCOPE-CONFLICT]`.

`[SCOPE-CONFLICT]` is halt → needs-human-review (not a builder nit). Out-of-scope defects become new packets, not silent PR expansion.

## Never

- Push commits or apply fixes (fixer mode off)
- Merge or formal self-approve as merge authority
- Approve when required CI is absent
- Approve when SHA does not match the handoff/head policy
