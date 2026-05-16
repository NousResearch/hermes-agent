# Agent Harness Workflow

This repository includes a lightweight phase/step harness adapted from `jha0313/harness_framework`.

## Purpose

Use the harness when a feature is large enough to split into ordered implementation steps and you want an external coding agent to execute each step with guardrails, status tracking, retry context, and commits.

## Directory layout

```text
phases/
├── index.json
└── 0-mvp/
    ├── index.json
    ├── step1.md
    └── step1-output.json   # generated after execution
```

Top-level `phases/index.json`:

```json
{
  "phases": [
    {"dir": "0-mvp", "status": "pending"}
  ]
}
```

Phase `phases/0-mvp/index.json`:

```json
{
  "project": "Hermes Agent",
  "phase": "mvp",
  "allowed_paths": [
    "phases/0-mvp/",
    "scripts/harness_execute.py",
    "tests/scripts/test_harness_execute.py"
  ],
  "steps": [
    {"step": 1, "name": "first-task", "status": "pending"}
  ]
}
```

Each `stepN.md` should contain:

```markdown
# Step N: short title

## Objective
What this step changes.

## Scope
- Files or components that may be touched.
- Explicit non-goals.

## Acceptance Criteria
- Exact commands/tests that must pass.
- Expected observable behavior.
```

## Running

Default agent command is Claude Code style:

```bash
python scripts/harness_execute.py 0-mvp
```

To use another agent, pass a safe command prefix. The harness appends the generated prompt as the final argument:

```bash
python scripts/harness_execute.py 0-mvp \
  --agent-command "codex exec"
```

Dangerous/full-auto flags such as `--full-auto`, `--yolo`, and `--dangerously-skip-permissions` are refused by default. Use `--allow-unsafe-agent` only inside an isolated worktree after explicit approval.

Or set it once:

```bash
export HARNESS_AGENT_COMMAND="codex exec"
python scripts/harness_execute.py 0-mvp
```

Optional push after all steps complete requires a second explicit confirmation after reviewing the diff and secret scan:

```bash
python scripts/harness_execute.py 0-mvp --push --yes-push
```

### Commands harness

Automation target: while a harness phase is running, update the current status snapshot every 1 minute so operators can see which phase/step is active, when it started, and whether it is pending, running, blocked, completed, or errored. The interval is configurable with `--status-interval` and must be greater than 0.

## Step status contract

The agent must update `phases/<phase>/index.json` for the current step:

- success: `"status": "completed"` and `"summary": "one-line result"`
- needs human input: `"status": "blocked"` and `"blocked_reason": "..."`
- cannot complete: `"status": "error"` and `"error_message": "..."`

The harness records timestamps, writes a redacted/bounded `stepN-output.json` artifact, refreshes `status.json` while a step runs, updates top-level phase status, and commits only phase-allowlisted paths. `allowed_paths` are frozen when the run starts; if an agent mutates them or changes out-of-scope files, the run fails closed. Output artifacts are excluded from staging by default.

## Guardrails loaded into every step

The prompt includes:

- `AGENTS.md`
- `CLAUDE.md` when present
- top-level Markdown docs under `docs/*.md`
- summaries from completed previous steps

## Undersea Friends operating model

Use harness phases for material changes to the Undersea Friends structure: automation, file/config edits, gateway/provider/process work, cross-profile handoffs, and long-running refactors. Casual conversation can still happen directly with any friend; not every message needs ticketing.

Role ownership:

- `nemo` (`니모`): execution and verification gate for harness phases, Hermes settings, file edits, tests, and gateway-safe runbooks.
- `manta` (`만타`): front-door planner for AX/research/automation framing, options, and decision criteria.
- `whale` (`고래`): canonicalization, LLM_WIKI/data quality, schema/index/log hygiene, and long-term structure checks.
- `shark` (`상어`): security, permissions, secret exposure, public/private boundary, and risky-operation review.
- `octopus` (`문어`): multi-tool automation/prototype helper across Slack, Notion, docs, mail, browser, and files.

For cross-profile work, the handoff packet must include: `goal / why / current state / paths / risks / execution steps / verification / report format`. Risky live operations must begin with read-only diagnosis and require explicit user approval before changing gateways, processes, providers, or credentials-adjacent files.

Current structure-change phase:

```bash
python scripts/harness_execute.py 0-undersea-friends-harness-structure \
  --agent-command "codex exec"
```

This phase declares `allowed_paths` in its phase index so the harness can stage only the intended harness/docs/test files.

## Safety notes

- Run only from a clean worktree or a dedicated isolated worktree; the harness refuses dirty worktrees by default.
- The harness never uses blanket `git add -A`; it stages only `allowed_paths` from the phase index plus the phase directory, excluding `stepN-output.json` artifacts.
- `allowed_paths` are captured at startup. If an agent edits the allowlist or leaves out-of-scope files dirty, the harness refuses to commit.
- Keep step scopes small; do not put multiple unrelated features in one step.
- Avoid secrets in phase files or step outputs. Captured output is redacted and truncated, but agents should still avoid printing secrets.
- Do not restart/stop gateways or kill processes inside a harness step unless the step explicitly states approval and rollback conditions.
- Use `--push --yes-push` only after reviewing the staged diff and confirming no secret/config/gateway-adjacent changes are included.
