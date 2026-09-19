# Codex Lane Prompt Template

Use this template when constructing prompts for Codex CLI within a Kanban workflow.

## Required Fields

```markdown
## Task
- **task_id**: {kanban_task_id}
- **Title**: {task_title}
- **Acceptance Criteria**:
  - {criterion_1}
  - {criterion_2}

## Context
- **Repo**: {repo_path}
- **Worktree**: {worktree_path}
- **Branch**: {branch_name}
- **Allowed Scope**: {file_patterns_or_directories}

## Safety Block
- Hermes owns Kanban lifecycle; Codex is an input lane only.
- Do NOT access secrets, send external messages, or mutate the board.
- No unrelated refactors or dependency upgrades unless required by the task.

## Output Required
- Concise summary of changes
- Files changed (list)
- Commits made (hash + message)
- Tests run and results
- Known risks or follow-up items

## Verification
- Commands Codex may run: {commands}
- Commands Hermes will run after: {commands}
```

## For Prediction-Market-Bot

Do NOT use `--yolo` for PMB. Prefer `--full-auto` inside the isolated worktree, then rely on Hermes reconciliation.
