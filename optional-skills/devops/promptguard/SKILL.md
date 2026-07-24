---
name: promptguard
description: >
  Offline prompt-contract auditor for agent workflows. Use before writing or
  seeding system prompts, agent instructions, or vague coding tasks ("fix this
  and write code"). Checks ownership, verification, safety, and tool schema;
  returns findings with approval criteria and a grounded fix draft. No model API.
version: 0.4.2
author: mturac
license: MIT
metadata:
  hermes:
    tags: [prompt, guardrails, safety, coding, contracts, pre-write, devops]
    category: devops
    related_skills: []
---

# PromptGuard

Audit prompts as **executable contracts** — offline, deterministic, zero dependencies.

Upstream project: [mturac/promptguard](https://github.com/mturac/promptguard)

## When to Use

- User asks to write, seed, or edit a system / agent / router / evaluator prompt
- Coding task is vague (missing owner, surface, tests, acceptance criteria)
- Pre-write review of `AGENTS.md`, `SKILL.md`, `CLAUDE.md`, or prompt configs
- CI or agent hook should fail closed on incomplete instruction contracts

## Install

```bash
# CLI
pipx install "git+https://github.com/mturac/promptguard.git"

# Hermes skill + pre_tool_call plugin (from the repo)
git clone --depth 1 https://github.com/mturac/promptguard.git /tmp/promptguard
/tmp/promptguard/install-agent-adapters.sh hermes
```

Or install the skill bundle from GitHub (may require `--force` under community scan):

```bash
hermes skills install mturac/promptguard/skills/promptguard --force
```

Restart Hermes after adapter install. Recommended env:

```bash
export PROMPTGUARD_PROFILE=coding-agent
export PROMPTGUARD_FAIL_ON=high
```

## Procedure

1. Prefer package CLI when available:
   ```bash
   promptguard audit <file> --profile coding-agent --fail-on high --format markdown
   printf '%s' '<prompt>' | promptguard audit - --profile coding-agent --fail-on high
   ```
2. For a whole repo:
   ```bash
   promptguard audit-repo . --profile coding-agent --fail-on high
   ```
3. If high/critical findings appear before a write: **do not write yet**. Show findings, ask for approval, or apply a grounded fix draft.
4. Optional interactive review:
   ```bash
   promptguard tui <file> --profile coding-agent
   ```

## Profiles

| Profile | Use |
|---------|-----|
| `coding-agent` | Implementation prompts (responsibility, risk, verification) |
| `system` | System / router / policy prompts |
| `security` | Instruction-override and exfil patterns (PG016–018) |
| `general` | Full core catalog |

## Report shape

`Severity | Evidence | Impact | Missing Contract | Questions | Approval | Fix Draft`

## Pitfalls

- Community hub install may flag scripts as dangerous (false positives on instruction filenames). Prefer `install-agent-adapters.sh hermes` or package CLI.
- Plugin hard-block uses `PROMPTGUARD_PROFILE` / `PROMPTGUARD_FAIL_ON`. Disable with `PROMPTGUARD_HERMES_DISABLE=1`.
- Product is intentionally offline — no remote model judge.

## Verification

```bash
promptguard audit - --profile coding-agent --fail-on high <<'EOF'
Fix this bug and write code.
EOF
# expect non-zero exit and PG012 / related findings
```
