# PROJECT_RULES.md

Shared execution guardrails for AI assistants working in this Hermes checkout.

## Primary Goals

- Build Hermes toward this north star:
  **Hermes is a persistent, cost-aware AI orchestration system that turns
  high-level goals into verified outcomes by coordinating replaceable AI
  workers, long-running processes, isolated Git workflows, bounded retries,
  evidence-based verification, and selective escalation to expensive expert
  models.**
- Keep scope tight and evidence-based.
- Protect secrets, local Hermes state, generated runtime files, and operator setup.
- Follow upstream Hermes architecture: narrow core, capabilities at edges, prompt-cache stability.
- Stop cleanly when progress stalls instead of looping.

## Hermes North Star

Hermes is the boss above replaceable AI workers, not a new coding agent to
replace Codex, Claude Code, or DeepSeek. A user gives a high-level goal and
policy; Hermes should manage the work lifecycle: plan, split tasks, choose the
right model by cost and capability, run long jobs, capture logs/artifacts,
classify failures, retry within bounds, isolate branches/worktrees, verify,
commit, merge, audit, report, and recover next actions after context loss,
process death, API timeout, restart, or the user going away.

Default economic policy: cheap models do the routine labor; expensive models
act as experts, auditors, and escalation lanes. Do not hard-code workflows
around any one provider or agent. DeepSeek, OpenCode, Codex, Claude, local
models, and future cheaper models are interchangeable worker lanes behind the
orchestration policy.

Decision filter for every new task or feature: does it improve persistent
goal/workflow state, model routing, long-running process supervision,
evidence capture, root-cause classification, bounded retry, isolated Git
execution, verification, commit/merge control, expert audit, cost accounting,
or final reporting? If not, it is probably not on the main path.

## Required Read Order

1. `AGENTS.md`
2. `HANDOFF.md`
3. `PROJECT_STRUCTURE.md` for large, cross-file, risky, or architecture-sensitive work
4. The files directly involved in the task

## Default Execution Loop

For non-trivial changes:

1. Derive target outcome, scope, and acceptance criteria.
2. Inspect the relevant implementation and tests before editing.
3. Make the smallest safe change.
4. Run the smallest meaningful validation.
5. Review diff for unrelated churn, secrets, generated files, and Windows portability issues.
6. Report changed files, validation, skipped checks, and remaining risk.

## Hermes-Specific Rules

- Preserve prompt caching and strict user/assistant/tool role alternation.
- Do not mutate old conversation context or rebuild the system prompt mid-session unless implementing compression or an explicitly approved lifecycle change.
- Do not add new core model tools unless the capability cannot reasonably live as existing code, CLI plus skill, gated tool, plugin, or MCP server.
- User-facing behavioral settings belong in `~/.hermes/config.yaml`, not `.env`. `.env` is for secrets only.
- Use `get_hermes_home()` / profile-aware helpers instead of hardcoding `~/.hermes`.
- On Windows, avoid POSIX-only process, signal, permission, symlink, and shell assumptions. Read the Windows section in `CONTRIBUTING.md` before touching cross-platform code.
- Use explicit text encodings for file I/O.

## Validation Rule

- Use `scripts/run_tests.sh` for Python tests because it matches CI isolation.
- For narrow changes, run a targeted test path first, for example:
  `bash scripts/run_tests.sh tests/agent/test_example.py::test_case`
- For packaging/config changes, prefer metadata tests and import smoke checks.
- For TypeScript work, use the package/workspace scripts already present in `package.json` files.
- If validation cannot run on native Windows due to shell assumptions, say exactly what was skipped and why.

## Safety Boundaries

- Do not read, print, edit, stage, or commit real API keys, OAuth tokens, bot tokens, private keys, local auth files, session DBs, or user memories unless the user explicitly asks and the action is safe.
- Do not run live gateway, messaging, cron, browser, billing, account, or external-provider actions unless explicitly requested.
- Do not run broad destructive cleanup commands.
- Do not stage `.env`, `.ai-runs/`, `.hermes/`, logs, caches, build outputs, node_modules, venvs, or generated dist artifacts.

## Repeated-Blocker Rule

Make at most two meaningful attempts against the same error signature. If the same signature remains, stop and report `BLOCKED` with:

- task and phase
- commands tried
- changed files
- newest relevant logs/artifacts
- root-cause hypothesis
- risk of continuing
- recommended next owner/action

## Handoff Rule

Update `HANDOFF.md` when local state changes in a way the next session needs to know: setup status, blocker, artifact path, validated command, chosen provider/backend, or an active work boundary.
