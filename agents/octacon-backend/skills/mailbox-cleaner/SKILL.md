---
name: mailbox-cleaner
description: "Use when Sahil wants daily inbox triage across 7 accounts (3 Gmail + 4 Outlook), job hunt digest, or urgent recruiter pings. Two-flow architecture with approval-gated destructive actions."
version: 1.0.0
author: KENSEI
metadata:
  hermes:
    tags: [email, gmail, outlook, triage, digest, cron, job-hunt, approval-gate]
    related_skills: [mailbox-agent, gmail-inbox-audit, google-workspace-mcp, add-gmail-account]
adoption_status: permanent
---

# Mailbox Cleaner — Daily Triage & Digest Engine

Autonomous multi-inbox email cleaner. Runs two daily cron flows (main + job hunt), plus an hourly urgent detector. Categorises every unread email across 7 inboxes, applies labels/folders for safe categories, queues destructive actions behind a Telegram approval gate, and delivers HTML digests.

## When to Use

- Daily scheduled cron execution (08:00 + 08:05 UK).
- Hourly urgent-detection cron (09:00-20:00 UK) for job-hunt pings.
- On-demand force-run: `hermes cron run <job-id>`.
- Never: as an interactive tool. This is a cron-driven system.

## Architecture

Two flows, three cron jobs, one approval gate.

```
Flow 1: Main Cleaner (mailbox-cleaner-main)
  Schedule: 0 8 * * * (08:00 UK)
  6 inboxes → categorise → safe actions → queue destructive → post digest

Flow 2: Job Hunt Cleaner (mailbox-cleaner-jobhunt)
  Schedule: 5 8 * * * (08:05 UK)
  1 inbox (sahil_ss@outlook.com) → sophisticated triage → post digest

Urgent Detector (mailbox-cleaner-urgent-detector)
  Schedule: hourly 09:00-20:00 UK
  Rule-based pattern match → silent or one-line alert
```

### Current deployment (20/05/26)

| Flow | Job ID | Schedule | Delivery | Mode |
|---|---:|---|---|---|
| Main Cleaner | `bbcc783939f6` | `0 8 * * *` | `discord:#job-hunt` | dry-run/read-only |
| Job Hunt Cleaner | `65d1447fbfc1` | `5 8 * * *` | `discord:#job-hunt` | dry-run/read-only |
| Urgent Detector | `847f441061fd` | `0 9-20 * * *` | `discord:#job-hunt` | dry-run/read-only |

Discord is the active delivery surface for this deployment. Do not emit raw Telegram HTML to Discord; convert digests to Discord-readable text and attach any generated HTML report with `MEDIA:/absolute/path`.

MCP bridge: `mcp_servers.mailbox_cleaner` runs `/home/kensei/.hermes/scripts/mailbox_cleaner_mcp.py` and exposes health, job listing, and dry-run trigger tools.

Current healthcheck coverage uses the existing `HEALTHCHECKS_PING_URL` env var (`/start`, success, `/fail`). Separate healthchecks.io slugs were not created because no slug-management credential was available.

Original Telegram Inbox delivery remains the target architecture only if Sahil moves mailbox approvals back to Telegram.

## Connected Inboxes (7 accounts)

| Platform | Account | Role | Aggression |
|---|---|---|---|
| Gmail | saghir.sahil@gmail.com | Primary personal | Aggressive |
| Gmail | sahilsaghir.ss9@gmail.com | Dev/learning | Aggressive on Skool |
| Gmail | fusionfirststudios@gmail.com | Apps/business | Light |
| Outlook | sahil_ss9@hotmail.com | Family operations | **Conservative — never auto-delete** |
| Outlook | sahil_saghir@hotmail.co.uk | Promo dump | Aggressive promo cleanup |
| Outlook | sahil_ss@outlook.com | **Primary job hunt** | Sophisticated triage |
| Outlook | matchdaymaestro@outlook.com | Abandoned, social | Bulk archive |

### Auth Status (as of 19/05/26)

- Gmail: all 3 healthy (indefinite refresh tokens)
- Outlook: all 4 healthy. sahil_ss@outlook.com lacks User.Read scope — mail/calendar only, `/me` returns 403. Non-blocking for this skill.
- MCP: Google Workspace MCP + MS 365 MCP both enabled in config.yaml

## Category Taxonomy

All labels prefixed `kensei/` (Gmail) or `KENSEI/` (Outlook).

### Universal (most inboxes)
- `Receipts` — transactional (Amazon, Stripe, Apple, etc.)
- `Promo` — marketing, discounts
- `Newsletter` — subscribed digest content
- `Service` — operational notifications from tools/platforms
- `Uncertain` — couldn't categorise confidently

### Gmail-specific
- `Infrastructure` — Healthchecks, server alerts
- `AI-Tools` — OpenAI, Ollama, OpenRouter, Anthropic
- `Skool` — Skool community digests (sahilsaghir.ss9)
- `Coaching` — football coaching content (sahilsaghir.ss9)
- `AppStore` — Apple/Google Play receipts (fusionfirststudios)
- `Legal` — privacy notices, ToS (fusionfirststudios)

### Outlook-specific
- `Bills` — utilities (sahil_ss9 family inbox)
- `Childcare` — Baby's Days, Famly, gymnastics
- `Legal` — NRI Legal, property legal
- `Property` — estate agents
- `Social-Notifications` — Facebook, Instagram (matchdaymaestro)
- `Security` — Microsoft account security alerts

### Job hunt only (sahil_ss@outlook.com)
- `Applied` → existing "Job Applications" folder
- `Active` → existing "Job Applications" folder, **flag urgently**
- `Rejected` → KENSEI/Rejected, auto-archive after 30 days
- `JobAlerts` → KENSEI/JobAlerts, auto-archive after 14 days
- `Recruiter` → KENSEI/Recruiter, flag, never auto-delete

**Critical:** sahil_ss@outlook.com has an existing "Job Applications" folder curated since Feb 2025. Use this as canonical for Applied + Active. Do NOT create parallel KENSEI/Applied.

## Confidence Tiers

For every email categorised:

| Tier | Threshold | Action |
|---|---|---|
| High | >90% | Execute per-inbox default rule |
| Medium | 60-90% | Label suspected category, flag in digest |
| Low | <60% | Leave in inbox, flag as Uncertain |

## Per-Inbox Rules Summary

See `main-prompt.md` for the full per-inbox categorisation tables and confidence thresholds. Key invariants:

- **sahil_ss9@hotmail.com:** NO auto-delete from this inbox. All destructive actions queued.
- **matchdaymaestro@outlook.com:** Social-Notifications archived immediately, never proposed for delete.
- **sahil_ss@outlook.com:** Recruiter emails flagged, never auto-delete. Rejected + JobAlerts auto-archived on timers.

## Server-Side Rules (One-Time Setup)

Applied at platform level — execute even if cron fails. KENSEI guides Sahil through these in browser. Documented in `server-rules-guide.md`.

### Outlook web (5 rules)
1. Healthchecks → KENSEI/Infrastructure, skip inbox (sahil_ss9@hotmail.com)
2. Microsoft security → KENSEI/Security, skip inbox (all 4 Outlooks)
3. Job hunt consolidation: Outlook 2 → forward to Outlook 3
4. Facebook/Instagram bulk → KENSEI/Social-Notifications, mark read (matchdaymaestro)
5. The Rundown AI → KENSEI/Newsletter, skip inbox (sahil_ss9@hotmail.com)

### Gmail filters (2 filters)
1. Healthchecks → kensei/Infrastructure, skip inbox (saghir.sahil@gmail.com)
2. Skool high-volume → kensei/Skool, mark read, skip inbox (sahilsaghir.ss9@gmail.com)

## Digest Format

Both digests delivered as HTML attachments + concise Telegram summary. See `main-prompt.md` and `jobhunt-prompt.md` for the full format templates.

### Main Daily Digest (Flow 1)
Sections: Needs your attention, Auto-organised, Proposed for deletion.
Sunday edition includes weekly stats.

### Job Hunt Digest (Flow 2)
Sections: Active opportunities, New applications, Rejections this week, Job alerts overnight.

Empty inbox → silent. No "0 emails today" noise.

## Approval Gate

Destructive actions (delete, archive, unsubscribe) queue behind a two-step Telegram approval gate:

1. Sahil replies to digest with verb (e.g. "delete promos")
2. KENSEI: "About to delete N emails. Confirm?"
3. Sahil: "yes" / "confirm"
4. KENSEI executes

Queued actions auto-release after 7 days of no response.

## Reply Verbs

Parsed from Sahil's Telegram replies to digests. See `reply-parser.md` for full implementation.

- `delete promos` / `delete all promos` → execute queued promo deletions
- `archive alerts` / `archive job alerts` → archive overnight job alerts
- `review` / `show all` / `show list` → send full enumerated list
- `keep N` / `spare N` → mark specific emails to keep
- `unsubscribe [sender]` → generate unsubscribe action
- `mark N read` → bulk mark as read
- `done` / `ignore` / no reply → digest expires after 7 days

## Urgent Detection (Flow 2 mid-day)

Hourly cron (09:00-20:00 UK) checks sahil_ss@outlook.com for:

- Subject contains: "interview", "screening", "phone call", "schedule", "next step"
- Sender pattern: direct human (not `noreply@`, `donotreply@`)
- From folder: KENSEI/Active or sender previously categorised as Recruiter

Rule-based — no LLM needed for detection. Silent when no matches. One-line Telegram alert when triggered.

## Telemetry

Every run logs to `~/.hermes/cron/output/<job-id>/<timestamp>.md`:
- Run start/end timestamps
- Per-inbox emails processed
- Per-category counts
- High/Medium/Low confidence breakdown
- Auto-actions executed
- Items queued for approval
- Errors / API failures

Sunday digest includes weekly stats.

## Edge Cases

1. **Mid-day urgent email** — hourly detector fires Telegram alert
2. **Approval queue accumulation** — auto-release after 7 days with warning
3. **Server-side rule misfire** — sanity check footer in digest if traffic patterns deviate
4. **Cron failure** — healthchecks.io heartbeat for each job, alerts if no fire within 15 min
5. **Personal correspondence as junk** — confidence-based safety (>90% threshold) only safeguard. No personal allowlist in v1.
6. **Empty inbox** — skip digest entirely, stay silent
7. **New sender pattern** — auto-flag as Uncertain, surface in digest

## Rollback

```bash
# Pause all three cron jobs
hermes cron pause <mailbox-cleaner-main-id>
hermes cron pause <mailbox-cleaner-jobhunt-id>
hermes cron pause <mailbox-cleaner-urgent-detector-id>

# Or remove entirely
hermes cron remove <mailbox-cleaner-main-id>
hermes cron remove <mailbox-cleaner-jobhunt-id>
hermes cron remove <mailbox-cleaner-urgent-detector-id>
```

Server-side rules persist at platform level — Sahil removes manually. Labels/folders remain as empty containers. Email actions taken before rollback: Trash items restorable for ~30 days. Archived items remain archived (searchable).

## Skill Files

| File | Purpose | Phase |
|---|---|---|
| `SKILL.md` | This file — metadata, architecture, entry point | 3 |
| `spec.md` | Original design specification (read-only reference) | — |
| `implementation-plan.md` | 7-phase build plan with dependencies | 1 |
| `main-prompt.md` | Flow 1: self-contained cron prompt for 6-inbox main cleaner | 3 |
| `jobhunt-prompt.md` | Flow 2: self-contained cron prompt for job hunt cleaner | 3 |
| `urgent-detection-prompt.md` | Self-contained prompt for hourly urgent detector | 3 |
| `reply-parser.md` | Verb-pattern definitions for Telegram reply handling | 3 |
| `server-rules-guide.md` | Step-by-step guides for Outlook/Gmail server-side rules | 2 |
| `folder-mapping.md` | Existing folder audit results (produced at Phase 1) | 1 |

## Common Pitfalls

1. **Creating KENSEI/Applied parallel to existing "Job Applications" folder.** The spec is explicit: use the existing folder as canonical. Sahil has curated it since Feb 2025.
2. **Auto-deleting from sahil_ss9@hotmail.com.** This is the family inbox. Hard rule: queue, never auto-delete. Two-step approval required.
3. **Silent cron failure without healthchecks.** Each cron job needs a separate healthchecks.io heartbeat slug. If the digest doesn't fire within 15 min of schedule, healthchecks.io alerts Sahil.
4. **Parallel MCP calls.** Gmail batch calls can trigger 429 or port contention. Always serialise across accounts. Load `mailbox-agent` for rate-limit backoff patterns.
5. **Outlook `account` parameter omission.** Without explicit `account=<email>`, MS 365 MCP may return messages from the default account (sahil_ss@outlook.com). Always explicit.
6. **Markdown in Telegram output.** The gateway delivers via HTML parse mode. Use `<b>`, `<code>`, `<blockquote expandable>`. Never `**bold**` or `*italic*`. Load `telegram-house-style` for full spec.
7. **First-run bulk archive temptation.** Sahil explicitly rejected this. Clean from day one — no historical sweep. matchdaymaestro will look messy for weeks.

## Related Skills

- `mailbox-agent` — Read-only daily digest across all accounts. No mutations. The read-only counterpart.
- `gmail-inbox-audit` — One-off deep Gmail audit and filter creation.
- `google-workspace-mcp` — Gmail MCP tool reference and OAuth troubleshooting.
- `add-gmail-account` — Adding new Gmail accounts to the MCP server.
