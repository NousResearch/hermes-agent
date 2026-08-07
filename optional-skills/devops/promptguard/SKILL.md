---
name: promptguard
description: Audit prompt contracts before agent execution.
version: 0.4.2
author: Mehmet Turac (@mturac), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [prompt, guardrails, safety, coding, contracts, pre-write, devops]
    category: devops
    related_skills: []
---

# PromptGuard Skill

PromptGuard audits prompts as executable contracts before an agent acts on
them. It runs offline, reports missing ownership, verification, safety, and tool
schema requirements, and does not call model APIs or modify audited files.

## When to Use

- Before writing or changing a system, agent, router, or evaluator prompt.
- When a coding task omits ownership, tests, acceptance criteria, or safety
  boundaries.
- Before publishing changes to `AGENTS.md`, `SKILL.md`, `CLAUDE.md`, or prompt
  configuration.
- When CI or an agent hook should reject incomplete instruction contracts.

## Prerequisites

- Linux or macOS with Python and `pipx`.
- The Hermes `terminal` tool for running PromptGuard commands.
- Install the CLI once:

  ```bash
  pipx install "git+https://github.com/mturac/promptguard.git"
  ```

- Optional Hermes adapter installation:

  ```bash
  work_dir="$(mktemp -d)"
  git clone --depth 1 https://github.com/mturac/promptguard.git \
    "$work_dir/promptguard"
  "$work_dir/promptguard/install-agent-adapters.sh" hermes
  ```

  Restart Hermes after installing an adapter.

## How to Run

Use `terminal` to audit a file, standard input, or an entire repository:

```bash
promptguard audit path/to/prompt.md \
  --profile coding-agent \
  --fail-on high \
  --format markdown

printf '%s\n' 'Fix this bug and write code.' |
  promptguard audit - --profile coding-agent --fail-on high

promptguard audit-repo . --profile coding-agent --fail-on high
```

## Quick Reference

| Profile | Use |
|---|---|
| `coding-agent` | Implementation ownership, risk, and verification |
| `system` | System, router, and policy prompts |
| `security` | Instruction override and exfiltration patterns |
| `general` | Full core rule catalog |

| Command | Purpose |
|---|---|
| `promptguard audit <file>` | Audit one prompt file |
| `promptguard audit -` | Audit text from standard input |
| `promptguard audit-repo .` | Audit prompt-bearing files in a repository |
| `promptguard tui <file>` | Open an optional interactive review |

The report shape is:

```text
Severity | Evidence | Impact | Missing Contract | Questions | Approval | Fix Draft
```

## Procedure

1. Identify the prompt-bearing file or text and choose the narrowest matching
   profile.
2. Run the audit through `terminal` with an explicit `--fail-on` threshold.
3. Preserve the command exit status and report every high or critical finding
   with its evidence.
4. Do not perform the requested write while blocking findings remain. Present
   the approval criteria or a grounded fix draft first.
5. Re-run the same command after revising the prompt. A clean rerun is the
   completion signal.
6. Use `promptguard tui <file>` only when an interactive review adds value; it
   is not required for deterministic validation.

## Pitfalls

- PromptGuard is intentionally offline; do not describe its output as a remote
  model review.
- Community installation can flag instruction-related filenames. Inspect the
  repository before using an override such as `--force`.
- Adapter settings use `PROMPTGUARD_PROFILE` and `PROMPTGUARD_FAIL_ON`.
  `PROMPTGUARD_HERMES_DISABLE=1` disables adapter blocking for recovery.
- A successful process launch is not a clean audit. Use the exit status and
  finding severity to determine the result.
- Do not broaden a task merely because the audit supplies a possible fix draft.
  Keep changes within the user's approved scope.

## Verification

Run a deliberately incomplete prompt and confirm that the configured threshold
rejects it:

```bash
set +e
printf '%s\n' 'Fix this bug and write code.' |
  promptguard audit - --profile coding-agent --fail-on high
status=$?
set -e
test "$status" -ne 0
```

Expected result: a non-zero exit status with `PG012` or another relevant
contract finding. Then audit a complete prompt and confirm the same command
returns zero before treating the workflow as verified.
