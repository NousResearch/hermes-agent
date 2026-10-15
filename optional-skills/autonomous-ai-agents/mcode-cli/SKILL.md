---
name: mcode-cli
description: Delegate coding tasks to the MiniMax Code CLI.
version: 0.1.0
author: DanielWalnut (hetaoBackend), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [mcode]
metadata:
  hermes:
    tags: [Coding-Agent, MiniMax-Code, CLI, Delegation, Code-Review]
    related_skills: [antigravity-cli, claude-code, codex, grok, hermes-agent, openhands]
    requires_toolsets: [terminal]
---

# MiniMax Code CLI Skill

Delegate bounded coding tasks to the `mcode` headless runner from Hermes. Use
the bundled wrapper for deterministic JSON output and safe prompt transport;
keep Hermes responsible for scope, review, and final verification.

## When to Use

- The user explicitly asks to use MiniMax Code or MCode.
- A feature, fix, refactor, or review benefits from an independent coding agent.
- A task should run in a separate workspace or continue an existing MCode session.
- You need a structured `ExecResultV1` result instead of an interactive TUI.

Use Hermes-native `delegate_task` for ordinary subagents. Use this skill when
MCode itself is the requested execution backend or a useful independent
second opinion.

## Prerequisites

1. Verify Node.js satisfies the package's current `engines` declaration.
2. Install the public package and verify the active command:

   ```text
   npm install --global @minimax-ai/code@latest --registry https://registry.npmjs.org/
   command -v mcode
   mcode --version
   ```

3. Authenticate once with `mcode login --region global` or
   `mcode login --region cn`. For BYOK, configure a provider with
   `mcode provider` instead.
4. Confirm the headless contract with `mcode exec --help`.
5. Use the platform's Python launcher for the wrapper: normally `python3` on
   Linux/macOS, or `py -3` / `python` on Windows.

If npm blocks native install scripts, follow npm's prompt or reinstall with
the narrow allow-list for `@minimax-ai/code` and `better-sqlite3`; do not
enable arbitrary package scripts globally.

## How to Run

`SKILL_DIR` is the directory containing this `SKILL.md`. Run the wrapper
through the Hermes `terminal` tool:

```text
terminal(
  command="python3 SKILL_DIR/scripts/run_mcode.py --cwd /path/to/repo --timeout 10m 'Fix the failing focused test and verify it.'",
  workdir="/path/to/repo",
  timeout=660
)
```

For prompts containing shell syntax, secrets, or long specifications, write a
UTF-8 task file with `write_file` and pass `--prompt-file`; do not interpolate
untrusted text into a shell command:

```text
terminal(
  command="python3 SKILL_DIR/scripts/run_mcode.py --cwd /path/to/repo --prompt-file /path/to/task.md --timeout 20m",
  workdir="/path/to/repo",
  timeout=1260
)
```

The wrapper sends the prompt as UTF-8 JSON to
`mcode exec --input - --input-format json`, requests `--output-format json`,
validates the final `ExecResultV1`, writes one JSON object to stdout, and
preserves MCode's nonzero exit code and stderr.

## Quick Reference

| Wrapper option | Purpose |
| --- | --- |
| `--cwd PATH` | Workspace MCode may inspect and modify. |
| `--prompt-file PATH` | Read the task without shell interpolation. |
| `--model PROVIDER/MODEL` | Override the model for this run only. |
| `--session ID` | Continue a specific active MCode session. |
| `--continue` | Continue the latest active session in `--cwd`. |
| `--permission smart` | Default; auto-handle low-risk actions. |
| `--permission full` | Allow all actions only with explicit authority. |
| `--timeout 10m` | Bound the inner MCode run. |
| `--max-steps N` | Bound assistant steps. |
| `--mcode PATH` | Use an explicit MCode executable. |

`--session` and `--continue` are mutually exclusive. The wrapper defaults to
`smart`; headless runs exit as blocked when an action still requires a human.
The final statuses and public exit codes are `succeeded` (0), `failed`
(3, 4, or 70 according to error category), `blocked` (5), `timeout` (6),
`limit_exceeded` (7), and `cancelled` (130).

## Procedure

1. Inspect repository instructions, status, and the requested scope before
   delegating. Create any required branch or worktree first.
2. Give MCode a concrete task, constraints, expected files, and focused
   verification commands. Do not hand it an ambiguous product decision.
3. Choose a bounded inner `--timeout` and set the Hermes terminal timeout at
   least 30-60 seconds longer. The wrapper forwards MCode's run timeout; the
   terminal timeout is the independent outer deadline. Use background execution
   with `notify_on_complete=true` for long tasks, then inspect it with the
   `process` tool.
4. Read the JSON result. Treat `status: "succeeded"` as the worker's report,
   not proof that the change is correct.
5. Inspect the actual diff and run the repository's required tests yourself.
   Fix or revert out-of-scope changes before presenting the result.
6. Report the MCode session ID when continuity may be useful; use `--session`
   or `--continue` for a follow-up in the same workspace.

## Pitfalls

- Do not run the interactive `mcode` TUI without a PTY. Prefer this headless
  wrapper for delegation.
- Do not parse `stream-json` when only the final outcome is needed. The wrapper
  deliberately uses the single-document JSON contract.
- Do not treat exit code 5 / `status: "blocked"` as a model failure. A pending
  permission or questionnaire needs a different permission policy or a human.
- Do not use `--permission full` unless the user's authority covers the
  resulting writes and commands.
- Do not expose tokens, local auth files, full environment dumps, or sensitive
  prompts in logs or summaries.
- Do not let two agents edit the same checkout concurrently. Use isolated git
  worktrees for parallel tasks.
- Do not trust `command -v mcode` alone after upgrades. Check `mcode --version`
  and run a small headless smoke task.

## Verification

Run a read-only smoke task in a temporary or disposable workspace:

```text
python3 SKILL_DIR/scripts/run_mcode.py \
  --cwd /path/to/disposable/repo \
  --permission off \
  --timeout 2m \
  'Reply with exactly MCODE_HERMES_OK. Do not modify files.'
```

Verification passes only when the process exits 0 and the emitted object has
`type: "exec.result"`, `schemaVersion: 1`, `status: "succeeded"`, and an answer
containing `MCODE_HERMES_OK`. For real coding tasks, also inspect `git diff`
and run the repository's focused tests after MCode exits.
