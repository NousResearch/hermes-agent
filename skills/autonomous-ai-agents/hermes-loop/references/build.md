# Builder procedure

Load `references/protocol.md` first.

## Job

Implement exactly one unit: the acceptance criteria on your task card / parent packet. Return a structured handoff with a full git SHA.

## Steps

1. Preflight: correct repo from packet, clean or isolated worktree, default branch detected (never assumed).
2. Prefer fixing open `changes-requested` units for this packet before starting new AC work when both exist.
3. Read packet version and every AC-N / NG-N. If ambiguous or conflicting, block with one concrete question — do not guess.
4. Implement only AC-N. Preserve NG-N. No opportunistic refactors.
5. Run the narrowest useful lint/typecheck/tests for the change. Disclose pre-existing failures separately.
6. Before opening/updating a PR: re-check packet version unchanged and freeze still valid.
7. Open or update PR if that is the delivery mode. Description includes scope ledger (AC evidence, NG preservation, Other behaviour changes: None).
8. Complete the kanban task with `templates/build-handoff.md` filled, including **full commit SHA**.
9. If you push new commits after a prior review, say so explicitly — prior verdicts are invalid.

## Never

- Merge or enable auto-merge
- Expand scope
- Change packet AC/NG yourself (ask orchestrator/human)
- Use kernel `review` status
