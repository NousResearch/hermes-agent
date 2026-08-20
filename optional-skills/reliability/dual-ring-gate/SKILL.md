---
name: dual-ring-gate
description: >-
  Force AI self-checks with dual-layer enforcement and dynamic
  rule lifecycle.
version: 1.0.0
author: Ghl211
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Reliability, Guardrails, Self-Check, Meta, Error-Prevention]
    requires_toolsets: [terminal]
    related_skills: [session-startup]
    category: reliability
---

# Dual-Ring Gate

Every self-check mechanism depends on the AI remembering to load it.
But "remembering to load the self-check" has no self-check of its own.
Dual-Ring Gate closes this gap by moving self-checks from "the AI
decides to do them" to "the system forces them before the AI speaks."

## When to Use

Use this skill when your Hermes agent:

- Repeatedly makes the same mistake despite documented rules
- Skips startup checks (time confirmation, gateway status) in new
  sessions
- Fixes problems without updating its error rule database
- Has a growing pile of static rules that never get pruned
- Costs you more time in corrections than it saves

Do not use for one-shot tasks, simple queries, or agents that already
follow their rules perfectly.

## Prerequisites

- `terminal` tool (required for outer ring shell gate)
- `~/.hermes/SOUL.md` exists (created by Hermes setup)

## How to Run

### Quick install

```bash
hermes skills install \
  https://raw.githubusercontent.com/Ghl211/skills-introduction-to-github/main/AI-skill/dual-ring-gate/SKILL.md
```

### Manual setup

1. Copy `SKILL.md` to `~/.hermes/skills/knowledge/dual-ring-gate/`
2. Append inner ring instructions to `~/.hermes/SOUL.md`
3. Place `hot-rules.json` in `~/.hermes/flywheel/`

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| Inner ring | `~/.hermes/SOUL.md` | 3 pinned instructions, auto-injected every session |
| Outer ring | `~/.hermes/scripts/pre-session-check.sh` | Shell gate run before AI speaks |
| Hot rules | `~/.hermes/flywheel/hot-rules.json` | Dynamic rule lifecycle state |

## Procedure

### 1. Inject inner ring into SOUL.md

Run `read_file` on `~/.hermes/SOUL.md` to check existing content,
then append:

```
## 🔴 Dual-Ring Gate · Inner Ring (auto-injected · cannot skip)
- **Time check**: terminal('date') before every response
- **Gateway check**: verify gateway status on first response
- **Rule update**: every fix must also update the error rule database
```

### 2. Create hot rules file

Create `~/.hermes/flywheel/hot-rules.json`:

```json
{
  "updated": "<today>",
  "hot": [
    {
      "rule": "terminal('date') before every response",
      "reason": "Most frequent error",
      "days_active": 1,
      "last_correction": "<today>"
    }
  ],
  "warm": [],
  "meta": {
    "hot_promote_if": "Corrected within last 3 days",
    "warm_cooldown": "7 days no recurrence → warm → cold",
    "retire_after_days": 30,
    "retire_reentry": "Penalty bounce: direct to hot"
  }
}
```

### 3. (Optional) Set up outer ring

Create `~/.hermes/scripts/pre-session-check.sh` (see
`scripts/pre-session-check.sh` in this skill directory) and add a shell
alias in `~/.bashrc`:

```bash
alias hermes='~/.hermes/scripts/pre-session-check.sh && hermes'
```

### 4. Maintain hot rules

When the agent makes a mistake, say:

> "I keep forgetting to check the time. Put it in hot rules."

The agent runs `write_file` or `patch` on `hot-rules.json` to add the
rule. Rules that go 30 days without recurrence auto-retire. Retired
rules that recur get a penalty bounce back to hot.

## Pitfalls

1. **Rules pile up** — keep `retire_after_days` at 30 or lower. Rules
   that haven't triggered in a month are noise.
2. **Inner ring too long** — limit to 3 rules max. A 10-item checklist
   becomes white noise the agent ignores.
3. **Outer ring blocks everything** — warn on non-critical failures;
   only hard-fail on P0 items like gateway being down.
4. **Fixes without rule updates** — this is the most common failure
   mode. Every fix must touch `hot-rules.json` or the error DB.
5. **Multiple agents writing simultaneously** — use sequential writes;
   `hot-rules.json` is not concurrency-safe.
6. **Assuming hot rules stay hot** — check `hot-rules.json` weekly.
   No changes means the agent isn't learning new patterns.

## Verification

After setup, start a fresh Hermes session and confirm:

- [ ] `terminal('date')` is called before the first response
- [ ] Gateway status is checked on first response
- [ ] `hot-rules.json` exists and has valid JSON
- [ ] Inner ring instructions are visible in the system prompt
- [ ] Outer ring script (if configured) runs without error
- [ ] A test correction ("I keep making mistake X") updates hot rules
- [ ] Old rules auto-demote after the configured cooldown
