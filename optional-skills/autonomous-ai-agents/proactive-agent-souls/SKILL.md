---
name: proactive-agent-souls
description: Give an agent a tiered proactivity contract in SOUL.md.
version: 1.1.0
author: Pedro Relvas (PRELVAS94), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [proactivity, soul, identity, cron, heartbeat, notifications, quiet-hours, agent-design]
    category: autonomous-ai-agents
    related_skills: [hermes-agent, dynamic-workflow]
---

# Proactive Agent Souls Skill

Put a standing proactivity contract in the agent's identity file, so an agent that runs
unattended ticks acts on its own without turning into a pager. The contract is four tiers —
silent self-maintenance, record-but-don't-send, bounded self-heal, and may-send-only-on-a-rule
— with the escalation decision held by a rules file the operator owns, plus a quiet-hours
window, a send-suppression window, and an observations log for everything that is not worth a
message. This skill writes and validates that contract; it is not a scheduler, a monitor, or a
notification service — the ticks stay with `cron` and `heartbeat`, delivery stays with the
platform, and the rules stay with the person who owns the agent.

## When to Use

- An agent has unattended work (cron jobs, heartbeats, watch loops) and its messages should be
  decisions, not activity reports.
- A chatbot has started narrating progress, re-reporting the same finding, or pinging on quiet
  ticks — the fix is a contract in the identity file, not another prompt line.
- You want the agent to fix its own breakage without asking, but only inside limits you wrote
  down in advance.
- Several profiles run the same role (`hermes -p <profile>`), and you want one message
  discipline across all of them without repeating it in each task prompt.
- NOT for: a one-off instruction on a single task (that belongs in the task itself), project
  rules that apply in one checkout (`AGENTS.md` / `.hermes.md`), or a temporary tone change for
  one session (`/personality`).

## Prerequisites

- Write access to the agent's home. Identity file: `$HERMES_HOME/SOUL.md` (default
  `~/.hermes/SOUL.md`); each profile home carries its own `SOUL.md` at
  `~/.hermes/profiles/<name>/SOUL.md`.
- An operator-owned rules file (any plain file you keep, e.g. `~/notes/PROACTIVITY_RULES.md`).
  It is the tripwire: a rule matches → a message is earned; nothing matches → nothing is sent.
- One real delivery target, or "stay silent" has nothing to suppress: a `cron` job that
  delivers to a chat, or a heartbeat-enabled session.
- Python 3.10+ for the checker script (stdlib only, no install).

## How to Run

```bash
# 1. see the block the contract is made of
read_file ~/.hermes/skills/autonomous-ai-agents/proactive-agent-souls/templates/proactivity-section.md

# 2. append it to the identity file (approved write), then fill every <…> placeholder
#    (agents and editors: use the patch tool — never rewrite SOUL.md wholesale)

# 3. write the rules file it points at, and the observations log
read_file ~/.hermes/skills/autonomous-ai-agents/proactive-agent-souls/templates/observations-log.md

# 4. check the result
python3 ~/.hermes/skills/autonomous-ai-agents/proactive-agent-souls/scripts/check_proactivity_contract.py

# 5. check a profile home, or several at once, machine-readable
python3 ~/.hermes/skills/autonomous-ai-agents/proactive-agent-souls/scripts/check_proactivity_contract.py \
  ~/.hermes/profiles/work --json
```

Exit status is `0` when the contract is intact, `1` when a finding is reported, `2` on a usage
error — usable directly as a `cron` monitor script.

## Quick Reference

| Piece | Where it lives | What enforces it |
|---|---|---|
| Four tiers (silent / record / self-heal / send) | `## Proactivity Protocol` in `SOUL.md` | the model, every session — including cron ticks |
| Escalation contract | the rules file the section names | the operator, by editing rules — no match, no message |
| Observations log | any path the section names, e.g. `~/notes/OBSERVATIONS.md` | the agent appends; never a message |
| Self-heal limits | Tier 3 of the same section | the operator's written hard limits; verified + logged, never announced |
| Message quality bar | subsection of the contract | review before send: event, why it matters, source |
| Delivery silence | the job's prompt | `cron` and `heartbeat` suppress on `[SILENT]` (or empty output) |
| Structure check | `scripts/check_proactivity_contract.py` | exit code — CI, a monitor job, or a pre-edit habit |
| Sizes and precedence | `/context` | per-file listing: loaded, truncated, shadowed, blocked |

## Procedure

1. **Pick the identity file, not a task prompt.** `SOUL.md` in the agent's own home is injected
   as identity slot 1 in every session, and a cron session keeps that persona while skipping cwd
   instructions — that is why the contract belongs there. Project-scoped rules go to
   `AGENTS.md`/`.hermes.md`; a session-scoped voice change goes to `/personality`.

2. **Append the template block.** `read_file` the current `SOUL.md`, then `patch` the
   `templates/proactivity-section.md` content onto the end. Writes to `SOUL.md` are in the
   protected-instruction class: they need human approval even under yolo, so a headless agent
   cannot rewrite its own identity — ask the owner, or have them apply the block. Keep the file
   under `context_file_max_chars` (20K by default); past it the file is head+tail truncated and
   the contract's middle disappears. If the file is already large, put the block at the end so
   the tail survives truncation.

3. **Separate the policy that changes from the identity that does not.** The four tiers are
   stable; what earns a message is not. Write the rules file the section names, keep the trigger
   register in it, and let the operator edit it freely — that way a new rule costs one file edit
   instead of an approved identity rewrite. Each rule is an event plus the decision it unlocks
   for the recipient: "server is down" qualifies, "server was checked" does not. A rule with no
   decision attached becomes noise within a week; leave it at Tier 2.

4. **Write the Tier 3 limits before you need them.** Name what the agent may fix alone (retry a
   job once, repair stale references and drifted config, clean its own orphaned scratch) and what
   it may not touch without a human (cost, provider and model settings, the rules file, data and
   deliveries a human opted into, anything that leaves the machine). Every heal must be
   reversible, verified afterwards, logged as one `actioned` line in the observations log, and
   never announced — a heal is maintenance, not news.

5. **Name the observations log and create it from `templates/observations-log.md`.** Give the
   path on the same line that mentions observations so the checker can see it. The log's status
   ladder (`open` → `confirmed` → `actioned`/`dismissed`) is what keeps the agent from
   re-investigating the same thing every tick, and a `confirmed` observation is promotion
   material: it leaves the log as work in whatever tracker you use (a task, a card, an issue) or
   is deleted as noise. It is never a message on its own.

6. **Wire the silence markers into the jobs.** In a `cron` job's prompt: "if nothing changed,
   reply with exactly `[SILENT]`". Empty output suppresses delivery too, and a heartbeat turn
   ending in `NO_REPLY`/`[SILENT]` posts nothing at all. A job that *hard-fails* still delivers a
   `⚠️ Cron … failed` notice unless it sets `--failure-deliver local` — that failure path is
   deliberate, not a contract breach.

7. **Validate.** Run the checker on every identity file you maintain; add it as a monitor job if
   the fleet is large. Then confirm the behaviours by hand: a tick with nothing to report
   delivers nothing; a tick whose rule matched delivers exactly one message; two rule matches
   inside the suppression window produce one message, not two.

8. **Re-check after upgrades.** In-place config migrations have rewritten `SOUL.md` before (the
   40→41 migration stripped a legacy appended section from every profile's `SOUL.md`), and
   `hermes profile update` replaces a distribution-owned identity file outright. Re-run the
   checker after upgrading Hermes, and after any `hermes profile install`/`update`.

## Pitfalls

- **The contract guides, it does not enforce.** `SOUL.md` sets identity; nothing in the runtime
  checks that the agent obeyed a tier. Enforce mechanically where you can: `[SILENT]` in the job
  prompt, a script that only prints on change, a rules file short enough to read.
- **An unattended agent cannot edit its own identity.** Protected-instruction writes always need
  human approval, so "the agent will tighten its own contract" is false — the loop is: the agent
  records observations, a human edits a rule.
- **Do not park volatile policy in `SOUL.md`.** Every rule change would then need an approved
  identity write, and a growing identity file is the one that gets truncated first.
- **Approval is not the only gate.** `SOUL.md` is scanned for prompt injection as user-authored
  content: a hit is warned about but the file still loads. A `SOUL.md` owned by a profile
  distribution (`distribution.yaml` lists it) is not user-authored, so a hit blocks the content
  instead — and it gets overwritten on the next distribution update.
- **Old sessions keep the old prompt.** Edits take effect cleanly in a new session; an
  already-running session may still be carrying the previous identity.
- **Do not restate the tiers in every task prompt.** Duplicating the contract in job prompts is
  how three slightly different escalation policies end up in one fleet; the job prompt should
  only carry the job-specific silence rule.
- **A quiet agent is not an idle one.** Tier 1–3 work is invisible by design; if you cannot see
  what the agent did during a quiet week, read the observations log rather than un-muting the
  chat.
- **Tier 3 without written limits is not autonomy, it is drift.** An agent that decides for
  itself how far to go will eventually touch something a human owned.
- `hermes --ignore-rules` skips `SOUL.md` injection entirely, so a contract test run with that
  flag proves nothing about the deployed identity.

## Verification

- `python3 <skill>/scripts/check_proactivity_contract.py ~/.hermes/SOUL.md` exits `0`, and
  reports `OK — proactivity contract intact` with the file's character count.
- The same command against an unfilled copy of the template fails with `placeholder`,
  `rules-file-missing`, `observations-log-missing`, and the quiet-hours/suppression warnings —
  the template is a scaffold, not a contract.
- The checker also fails on a missing `[SILENT]` marker, a tier defined twice, a tier with no
  body, tiers out of order, or a missing observations-log / rules-file path; the message-quality
  bar is a warning, promoted to a failure with `--strict`.
- `/context` lists `SOUL.md` under Context files with a token estimate and a state of `loaded`
  (not `truncated`, `shadowed`, or `blocked`).
- End to end: one tick with nothing to report produces no delivery; a tick whose rule matched
  produces one message naming the event and the decision; a Tier 3 heal appears as one
  `actioned` line in the observations log and nowhere else.
