# CODEX_SKILLS_GUIDE

## Purpose
- Explain how to use Codex skills in this project without guessing.

## Skill sources
1. Repo skills:
   - Path: `.agents/skills/`
   - Purpose: workflows reused inside this project
2. Global skills:
   - Path: `~/.agents/skills/`
   - Purpose: tools or workflows usable across many projects

## Starter repo skills in this project
| skill | use when | pair well with |
|---|---|---|
| `$task-kickoff` | work is starting and scope must be fixed first | `$issue-to-fix`, `$change-review`, `$beginner-report` |
| `$issue-to-fix` | a bug, regression, or broken behavior must be repaired | `$task-kickoff`, `$change-review`, `$beginner-report` |
| `$change-review` | edits are already made and must be checked | `$beginner-report` |
| `$beginner-report` | final answer must be easy for a non-developer | any skill |
| `$doc-governance` | handoff, index, policy, or rollover docs must be maintained | `$beginner-report` |

## Common global skills
| skill | use when |
|---|---|
| `$codex-harness-health` | check Codex harness health in one flow |
| `$openai-docs` | fetch up-to-date official OpenAI docs |
| `$subagent-orchestration` | split non-trivial safe work into bounded subagent lanes when independent reader/checker work exists |
| `$playwright` | automate a browser or capture UI evidence |
| `$obsidian-markdown` | edit Obsidian markdown safely |

## Rule of thumb
1. No change yet -> start with `$task-kickoff`
2. Bug or broken behavior -> add `$issue-to-fix`
3. Edits already done -> use `$change-review`
4. Easy Korean report needed -> add `$beginner-report`
5. Handoff or long docs changed -> add `$doc-governance`

## Subagent baseline
- Non-trivial and non-risky work should default to subagent consideration.
- If safe independent side work exists, prefer `main 1 + helper 1~3` after the eligibility check passes.
- Single-lane is the exception path for trivial, risky, blocked, overlapping-write, or runtime-limited work.
- If the runtime still requires stricter behavior, keep that note in the project KB or subagent playbook instead of the default rule text.
- Users can opt out with `이번 작업은 메인 lane 하나로만 진행해줘.`

## Auto-Skill Mode
- In this project, manual skill naming is optional.
- Codex should auto-select the smallest matching skill set before non-trivial work.
- Default flow:
  1. broad work -> `$task-kickoff` + `$subagent-orchestration`
  2. bug/failure -> `$issue-to-fix`
  3. edits already made -> `$change-review`
  4. easy report -> `$beginner-report`
  5. docs/index/handoff -> `$doc-governance`
  6. non-trivial work with safe independent side work -> `$subagent-orchestration`

## Copy-paste examples
### 1. Start work
```text
이번 작업은 `$task-kickoff` 방식으로 진행해줘.
목표: [끝낼 작업 1개]
완료조건: [성공 기준]
제약: [건드리면 안 되는 것]
검증방법: [테스트/확인 명령]
```
