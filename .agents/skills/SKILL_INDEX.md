# SKILL_INDEX

## Purpose
- Show which repo skills exist, when to call them, and how to combine them.

## Skills
| skill | use when | pair well with |
|---|---|---|
| `$task-kickoff` | work is starting and scope must be fixed first | `$issue-to-fix`, `$change-review`, `$beginner-report` |
| `$issue-to-fix` | a bug, regression, or broken behavior must be repaired | `$task-kickoff`, `$change-review`, `$beginner-report` |
| `$change-review` | edits are already made and must be checked | `$beginner-report` |
| `$beginner-report` | final answer must be easy for a non-developer | any skill |
| `$doc-governance` | handoff, index, policy, or rollover docs must be maintained | `$beginner-report` |

## Default combos
1. Start + fix: `$task-kickoff` + `$issue-to-fix`
2. Start + fix + easy report: `$task-kickoff` + `$issue-to-fix` + `$beginner-report`
3. Post-change check: `$change-review`
4. Documentation pass: `$doc-governance` + `$beginner-report`

## Rule of thumb
- no change yet -> start with `$task-kickoff`
- bug or failure -> add `$issue-to-fix`
- edits already done -> use `$change-review`
- easy Korean report needed -> add `$beginner-report`
- docs/handoff/index touched -> add `$doc-governance`
- non-trivial work with safe independent side work is available -> add `$subagent-orchestration`
- the user says `이번 작업은 메인 lane 하나로만 진행해줘.` -> stay single-lane

## Auto routing note
- Manual skill naming is optional.
- Codex should auto-select the smallest matching skill set before non-trivial work.
- Default auto route examples:
  1. broad work -> `$task-kickoff` + `$subagent-orchestration`
  2. bug fix -> `$task-kickoff` + `$issue-to-fix`
  3. bug fix + review + easy report -> `$task-kickoff` + `$issue-to-fix` + `$change-review` + `$beginner-report`
  4. doc maintenance -> `$doc-governance` + `$beginner-report`

## Subagent note
- Non-trivial and non-risky work should default to subagent consideration.
- If safe independent side work exists, prefer `main 1 + helper 1~3` after the eligibility check passes.
- If the runtime still requires stricter behavior, keep that note in the project KB or subagent playbook instead of the default rule text.
- Users can opt out with `이번 작업은 메인 lane 하나로만 진행해줘.`
