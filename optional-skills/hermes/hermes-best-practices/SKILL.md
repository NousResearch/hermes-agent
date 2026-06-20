---
name: hermes-best-practices
description: Practical habits for reliable Hermes Agent operation.
platforms: [linux, macos]
category: hermes
triggers:
  - "好习惯"
  - "best practices"
  - "检查一下有没有违规"
  - "audit my config"
  - "safety check"
  - "compliance review"
toolsets:
  - terminal
  - file
metadata:
  hermes:
    tags: [hermes, best-practices, safety, memory, telegram, configuration]
---

# Hermes Agent Best Practices

> Core philosophy: config files are supplementary guardrails, not absolute guarantees.
> Especially on Telegram (known SOUL.md loading instability — Issue #5200), the core habit
> is "verify regularly, persist proactively, human-in-the-loop as last defense."

## When to Use

Load this skill when:
- Self-auditing agent compliance or reviewing operational habits
- Reviewing past mistakes and hardening workflows
- User asks "check if I've violated any rules" or requests a safety review
- Configuring a new Hermes profile and want to avoid common pitfalls

## Prerequisites

- Hermes Agent installed and operational
- Read access to `~/.hermes/config.yaml`
- Familiarity with SOUL.md / AGENTS.md / MEMORY.md conventions

## Quick Reference

| Category | Top Habit |
|----------|-----------|
| Config | Verify SOUL/AGENTS loaded after edit (new session) |
| Memory | Monitor MEMORY.md budget — 50% safe, 70% watch, 85% clean |
| Delegation | Verify dangerous ops (delete/send/funds) actually trigger confirmation |
| Telegram | Restart gateway weekly; review allowed users periodically |
| Trust | Don't treat agent confirmations as absolute safety net |
| Anti-hallucination | Verify facts (date/price/news) with tools before output |

## Procedures

### 1. Configuration Maintenance

**1.1 Verify SOUL.md / AGENTS.md after changes**
After editing, start a new session and ask the agent to recite its current SOUL.md
path and key content. Never assume changes took effect — especially on Telegram
where auto-loading is unstable (Issue #5200).

**1.2 Change one config category at a time**
Don't modify multiple config files simultaneously then verify all at once. Change
one, verify, then move to the next.

**1.3 Backup before changes**
Before editing config files, save the current full content locally so you can
roll back directly rather than reconstructing from memory.

**1.4 Monitor MEMORY.md / USER.md character budget**
Let the agent self-report character usage regularly (weekly):
- MEMORY.md cap: 8,000 chars (50% safe / 70% watch / 85% clean)
- USER.md cap: 3,000 chars (50% safe / 70% watch / 85% clean)

⚠️ **memory replace truncation risk**: `memory action=replace` replaces the
*entire* matching entry, not just the old_text portion. If old_text only matches
part of an entry, the rest is lost. Safer: delete old entry then `action=add` rebuild.

**1.5 Harness consistency audit**
Audit SOUL/AGENTS/MEMORY/config.yaml against external frameworks (Rahul Harness
Engineering / Karpathy CLAUDE.md) for version drift, missing constraints, and
scene-mismatched rules. See `references/harness-audit-checklist.md`.
Key check: config.yaml embedded personality version ≠ SOUL.md version.

**1.6 Agent proactive reminders**
The agent should proactively remind when: SOUL/AGENTS edited → verify loading;
MEMORY >80% → clean; dangerous operations → verify rules active;
long tasks → progress checkpoints; ambiguous instructions → clarify boundaries;
weekly → restart gateway; monthly → audit memory.

**1.7 `tool_use_enforcement` scope**
`tool_use_enforcement: auto` matches model names against `TOOL_USE_ENFORCEMENT_MODELS`
in `agent/prompt_builder.py`. Current list (as of v0.13.0):
```
gpt, codex, gemini, gemma, grok, glm, qwen, deepseek
```
Claude models are NOT in this list. Verify with:
```bash
grep "tool_use_enforcement" ~/.hermes/config.yaml
grep -n "TOOL_USE_ENFORCEMENT_MODELS" ~/.hermes/hermes-agent/agent/prompt_builder.py
```

### 2. Delegation & Daily Tasks

**2.1 First-time dangerous op verification**
The first time a delete/send/funds/production operation triggers, manually observe
whether the agent actually stops to confirm. Trust rules only after seeing them
work in a real scenario.

**2.2 Phased progress for long tasks**
Multi-step, long-running tasks should report progress in phases. Catch errors
early rather than at the end.

**2.3 Clear instruction boundaries**
Give explicit scope: "delete temp files under test/" is safer than "clean up."

**2.4 Post-refactor chain verification**
After restructuring (script migrations/path changes/file renames), verify the
ENTIRE call chain, not just file existence. A filename mismatch in one link
(e.g., `optical_share` vs `optical_module_share`) only surfaces when running the
full pipeline.

**2.5 Scan first, then advise; verify first, then act**
- **Scan before advising**: When asked to "plan" or "suggest," scan the current
  state comprehensively first. Skipping this leads to unrealistic proposals.
- **Verify before acting**: When the user says "double-check," that means go through
  it a second time — not just a courtesy pause. Issues found during verification
  (e.g., rsync `--delete` traps, misspelled directory names) must be addressed
  before execution.

**2.6 Commands suggested to users must be verified first**
Any CLI command the agent recommends (e.g., `hermes config get`) must be tested
in its own environment before suggesting. Recommending unverified commands leads
to `invalid choice` errors. Similarly, before claiming "you can check with command X,"
confirm: (a) the command exists, (b) the feature exists, (c) the parameters are correct.

### 3. Telegram-Specific

**3.1 Sensitive ops in DM**
Sensitive or high-privilege commands should be sent via DM, not in group chats
where context is more easily misread and privacy mode may drop messages.

**3.2 Periodically review allowed users**
Regularly check TELEGRAM_ALLOWED_USERS — this list doesn't auto-sync with external changes.

**3.3 Periodic gateway restart**
Restart the gateway regularly (weekly recommended), especially after config.yaml
changes, to avoid runtime/disk config mismatch.

**3.4 Subdirectory AGENTS.md progressive discovery**
Subdirectory AGENTS.md files under Telegram gateway are NOT completely unreadable.
`subdirectory_hints.py` triggers scanning on any write operation (`terminal` with
writes, `write_file`). Pure reads and network calls don't trigger. Session startup
only loads CWD's AGENTS.md. See `references/subdirectory-agents-discovery.md`.

### 4. Memory & Long-Term Information

**4.1 Explicitly persist important info**
Don't assume the agent will "remember" one-off important info. If you want it
retained long-term, explicitly say "save this to memory."

**4.2 Periodic manual memory audit**
Monthly: have the agent output full MEMORY.md and USER.md content. Review for
stale/wrong/unimportant entries and manually clean rather than letting auto-accumulation continue.

**4.3 No sensitive info in memory**
Health, financial details, and other sensitive information should NOT be written
to MEMORY.md/USER.md. These files are read in full every session — the longer they
persist, the larger the exposure surface.

### 5. Trust Boundaries

**5.1 Confirmation is not an absolute safety net**
Don't treat agent confirmations as absolutely reliable — especially when SOUL.md/
AGENTS.md is known to sometimes not load correctly. The last line of defense for
critical operations is human judgment.

**5.2 Periodic Hard Constraints review**
Quarterly review of SOUL.md Hard Constraints — are they still appropriate for
current usage patterns? Rules written once aren't forever; adjust as needs evolve.

### 6. Anti-Hallucination & Fact-Checking

**6.1 Verify facts before output containing them**
Any output containing dates, weather, stock prices, news, file paths, or other
verifiable facts (TTS voice, message body, file content) MUST have the
corresponding tool executed BEFORE generating the text. Don't stuff unverified
info as "decorative filler."

- **Trigger**: Output contains specific numbers (time/date/price/quantity/version)
- **TTS specific**: Separate emotional content (greetings) from factual content —
  greetings handle mood, facts must be accurate.

**6.2 Mark sources item-by-item when asked to verify**
When the user explicitly asks "verify each item, answer separately, explain each
individually," distinguish every item as "executed/verified" vs "inferred/speculated."
Inferred items must be clearly labeled "this is speculation, no factual basis" —
don't pad with explanations to appear comprehensive. Don't replace actual execution
results with "it should..." If a command fails, report the error as-is.

**6.3 Confirm existence before claiming a feature or command**
CLI commands recommended to users must be run in the agent's own environment first.
Claims about config functionality must be backed by actual config.yaml fields or
official documentation. Never recommend based on "Hermes should have this feature."

**6.4 Config descriptions should match actual config.yaml**
Documentation summaries may be outdated or truncated. When describing current system
configuration, prefer actual `hermes config show` or `grep` output from config.yaml.
See `references/fact-verification-guide.md`.

## Pitfalls

- **SOUL.md / AGENTS.md auto-load on Telegram is unreliable** — known bug (Issue #5200).
  Verification method: ask agent directly "SOUL.md path + key content" rather than
  observing behavior.
- **memory replace destroys entries** — use delete+add pattern for safety.
- **vault-sync is an archive, not a pipeline config directory** — don't touch its
  subdirectories when refactoring push systems.
- **Scan before advising** — different scenarios need different harness patterns.
  Don't blindly copy coding-scene harness artifacts to content operations.

## Verification

After applying these practices:
1. Ask the agent to self-report MEMORY.md / USER.md character usage and percentage.
2. Start a new session after editing SOUL.md and verify the agent recites new content.
3. Test one dangerous operation to confirm the confirmation gate triggers.
4. Run a cron pipeline end-to-end after any restructuring.
5. Grep `TOOL_USE_ENFORCEMENT_MODELS` to confirm your current model's status.
