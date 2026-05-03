# Codex Job Profiles MVP

## Goal

Make `codex_job` jobs transparent orchestration units for Discord control-room
workflows. Each job should clearly show which model, effort, workspace, profile,
launch policy, and notable capability bundle was selected, while preserving the
existing tmux/worktree runner.

## Scope

- Add first-class `profile` metadata to `codex_job` with generous built-in
  profiles:
  - `base-rich`
  - `review-rich-readonly`
  - `ios-full`
  - `web-full`
  - `ops-full`
  - `hermes-full`
- Keep profile behavior honest:
  - `codex_job` can set coarse Codex launch flags such as `-m`, `-a`, `-s`,
    `--search`, `--profile`, and `-c key=value`.
  - It does not synthesize or hard-enforce per-MCP allowlists. Actual MCP and
    plugin availability still comes from the Codex CLI configuration unless an
    existing Codex config profile or explicit config override is supplied.
- Persist and render:
  - selected profile and summary
  - notable included and omitted capabilities
  - runner limitations
  - Codex launch flags used
  - workspace/repo/worktree/branch
  - phase, latest activity, key findings, tests, blockers, completion handoff
- Append a lightweight final worker handoff template to launched prompts unless
  disabled by the caller.
- Add a small retrospective/distillation marker so Hermes can later decide
  whether a long or complex job deserves deeper learning extraction.
- Add an automatic launch health check in the monitor/status path so obvious
  repo-trust prompts, permission prompts, and MCP startup stalls are surfaced
  quickly instead of letting a Codex job sit silently idle.

## Non-goals

- No gateway-specific restart loop.
- No private ops-doc writes from this worktree.
- No new MCP startup dependency.
- No claim that profile metadata equals hard MCP enforcement.

## Validation Plan

- Focused tests for:
  - default profile resolution and backward compatibility
  - read-only profile launch defaults
  - Codex CLI profile/config/search flag rendering
  - prompt handoff injection
  - status serialization and Discord status rendering
  - launch health detection/alert rendering
- Syntax check for `tools/codex_job_tool.py`.
- Use `scripts/run_tests.sh` rather than direct `pytest`.
