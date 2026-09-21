# KENSEI Skill Spec: Mailbox Cleaner v1

**Created:** 2026-04-29
**Skill name:** `mailbox-cleaner`
**Purpose:** Daily triage across 7 inboxes (3 Gmail + 4 Outlook) with two digest outputs (main + job hunt) and queued destructive action approval.
**Paste this into:** Code or Ops topic in KENSEI Workspace
**Expected build time:** 2-4 hours of focused KENSEI execution

---

## CONTEXT FOR KENSEI

This is a **skill design specification** written by web-Claude. It is NOT a step-by-step procedure. Your job is to translate this design into the actual implementation: skill files, cron jobs, prompts, and supporting tooling.

When you read this spec, do NOT begin implementation immediately. Confirm with Sahil first that:
1. You've read the entire spec
2. You understand the architecture (two flows, two cron jobs, two digests)
3. You have access to all the infrastructure named (Workspace MCP, Outlook MCP, Hermes cron, Telegram delivery)
4. You're clear on the approval gate constraint (only fires through Telegram messaging layer)

Then ask for go-ahead before building.

---

## EXECUTIVE SUMMARY

Build two cron jobs:

1. **Main mailbox cleaner** (`mailbox-cleaner-main`)
   - Schedule: every day at 08:00 UK time
   - Reads 6 inboxes (3 Gmail + 3 non-job-hunt Outlook)
   - Categorises emails with confidence scoring
   - Auto-executes safe actions (labelling, light archiving)
   - Queues destructive actions for approval
   - Posts main daily digest to Telegram Inbox topic

2. **Job hunt cleaner** (`mailbox-cleaner-jobhunt`)
   - Schedule: every day at 08:05 UK time (5 min after main, prevents Telegram message collision)
   - Reads 1 inbox: `sahil_ss@outlook.com`
   - Sophisticated triage with application status tracking
   - Posts job hunt digest to Telegram Inbox topic
   - **Plus**: real-time urgent ping for job-hunt-urgent emails detected mid-day (interview invites, screening callbacks, direct recruiter outreach)

Plus supporting work:

- One-time server-side rules setup in Outlook (consolidate Outlook 2 job hunt → Outlook 3, Healthchecks → folder, etc.)
- Personal allowlist: SKIPPED for v1 per Sahil's decision

---

## INBOX ROLES (FROM ANALYSIS)

These have been confirmed via inbox analysis run on 2026-04-28/29.

| Inbox | Platform | Role | Cleaning Aggression |
|-------|----------|------|---------------------|
| saghir.sahil@gmail.com | Gmail | Primary personal — mostly junk | Aggressive |
| sahilsaghir.ss9@gmail.com | Gmail | Dev/learning — Skool noise dominates | Aggressive on Skool, light elsewhere |
| fusionfirststudios@gmail.com | Gmail | Apps/business — already clean | Light |
| sahil_ss9@hotmail.com | Outlook | Family operations | **Conservative — never auto-delete** |
| sahil_saghir@hotmail.co.uk | Outlook | Promo dump with hidden signal | Aggressive promo cleanup |
| sahil_ss@outlook.com | Outlook | **PRIMARY JOB HUNT** | Sophisticated triage |
| matchdaymaestro@outlook.com | Outlook | Abandoned, social notif dump | Bulk archive, then minimal |

---

## ARCHITECTURE: TWO FLOWS

### Flow 1: Main Cleaner (`mailbox-cleaner-main`)

**Cron schedule:** `0 8 * * *` (08:00 UK time daily)

**Inboxes covered:**
- saghir.sahil@gmail.com
- sahilsaghir.ss9@gmail.com
- fusionfirststudios@gmail.com
- sahil_ss9@hotmail.com (family — conservative rules)
- sahil_saghir@hotmail.co.uk (promo cleanup)
- matchdaymaestro@outlook.com (abandoned, bulk archive)

**Output:** Main daily digest message posted to Telegram Inbox topic.

**Note:** sahil_ss@outlook.com is NOT in this flow. It's handled by Flow 2.

### Flow 2: Job Hunt Cleaner (`mailbox-cleaner-jobhunt`)

**Cron schedule:** `5 8 * * *` (08:05 UK time daily)

**Inboxes covered:**
- sahil_ss@outlook.com only

**Output:** Job hunt digest message posted to Telegram Inbox topic (separate message from main digest).

**Plus:** Real-time urgent detection. When new emails arrive in this inbox during the day, KENSEI checks rule-based patterns (subject contains "interview", "screening", "phone call", direct recruiter outreach patterns). If matched, KENSEI sends an immediate alert to Telegram Inbox topic — one-line, no LLM call needed for the detection itself.

**Implementation note for real-time:** This requires either:
- A second cron job running every 15 minutes during 8am-8pm UK that checks for new emails matching patterns, OR
- A webhook/long-polling integration with Outlook (more complex, faster)

For v1: cron-based hourly check from 09:00 to 20:00 UK is acceptable. We can upgrade to webhook later.

---

## CATEGORISATION TAXONOMY

KENSEI uses these labels (Gmail) / folders (Outlook). All prefixed with `kensei/` (Gmail) or `KENSEI/` (Outlook).

### Universal categories (used across most inboxes)
- `Receipts` — transactional (Amazon, Tesco, Stripe, Apple, etc.)
- `Promo` — marketing emails, discounts, "we miss you" emails
- `Newsletter` — intentionally subscribed digest content
- `Service` — operational notifications from tools/platforms
- `Uncertain` — KENSEI couldn't categorise confidently

### Gmail-specific
- `Infrastructure` — Healthchecks, server alerts (skip inbox via server-side rule)
- `AI-Tools` — OpenAI, Ollama, OpenRouter, Anthropic dev tools
- `Skool` — Skool community digests (sahilsaghir.ss9 only)
- `Coaching` — football coaching content (sahilsaghir.ss9)
- `AppStore` — Apple/Google Play receipts (fusionfirststudios)
- `Legal` — privacy notices, ToS (fusionfirststudios)

### Outlook-specific
- `Bills` — utilities (sahil_ss9 family inbox)
- `Childcare` — Baby's Days, Famly, gymnastics (sahil_ss9 family inbox)
- `Legal` — NRI Legal, property legal (sahil_ss9 family inbox)
- `Property` — estate agents (multiple inboxes)
- `Social-Notifications` — Facebook, Instagram (matchdaymaestro)
- `Security` — Microsoft account security alerts

### Job hunt categories (sahil_ss@outlook.com only)
- `Applied` — application confirmations, "we received your application"
- `Active` — interviews, screening calls, in-progress conversations
- `Rejected` — rejection emails (auto-archive after 30 days)
- `JobAlerts` — Indeed, CV-Library, Outside IR35 daily digests (auto-archive after 14 days)
- `Recruiter` — direct recruiter outreach (flag, never auto-delete)

---

## PER-INBOX RULES TABLE

### saghir.sahil@gmail.com (Aggressive cleaning)

| Category | Default Action |
|----------|---------------|
| Receipts | Label, keep in inbox |
| Promo | Label, propose-delete after 7 days |
| Newsletter | Label, auto-archive after read |
| Service | Label, flag in digest |
| Property | Label, flag in digest (Martin & Co type) |
| AI-Tools | Label, keep (dev signal) |
| Infrastructure | Label, skip inbox (Healthchecks) |
| Uncertain | Flag in digest |

### sahilsaghir.ss9@gmail.com (Aggressive on Skool)

| Category | Default Action |
|----------|---------------|
| Skool | Label, propose-delete after 7 days |
| Service | Label, flag in digest (Railway, Supabase, GitHub) |
| Coaching | Label, keep (Coaching Manual) |
| Promo | Label, propose-delete after 7 days |
| Uncertain | Flag in digest |

### fusionfirststudios@gmail.com (Light cleaning)

| Category | Default Action |
|----------|---------------|
| AppStore | Label, keep |
| Legal | Label, keep, flag if action-required |
| Service | Label, flag if action-required |
| Receipts | Label, keep |
| Uncertain | Flag in digest |

### sahil_ss9@hotmail.com (Conservative — family ops)

| Category | Default Action |
|----------|---------------|
| Bills | Move to KENSEI/Bills, keep |
| Childcare | Move to KENSEI/Childcare, **keep indefinitely**, flag |
| Legal | Move to KENSEI/Legal, **keep indefinitely**, flag |
| Newsletter | Move to KENSEI/Newsletter, auto-archive after 7 days |
| Promo | Move to KENSEI/Promo, propose-delete after 7 days |
| Security | Move to KENSEI/Security, archive after 30 days |
| Uncertain | Flag in digest |

**Hard rule:** NO auto-delete from this inbox. All destructive actions queued for approval.

### sahil_saghir@hotmail.co.uk (Aggressive promo cleanup)

| Category | Default Action |
|----------|---------------|
| Promo | Move to KENSEI/Promo, propose-delete after 7 days |
| Property | Move to KENSEI/Property, flag (HoldenCopley) |
| Uncertain | Flag in digest |

**Server-side rule (one-time setup):** Forward any job-hunt-pattern emails (Workable, recruitment platform domains) to sahil_ss@outlook.com for consolidation.

### matchdaymaestro@outlook.com (Bulk archive)

| Category | Default Action |
|----------|---------------|
| Social-Notifications | Move to KENSEI/Social-Notifications, archive immediately, never propose delete |
| AI-Tools | Move to KENSEI/AI-Tools (Kling), keep |
| Promo | Move to KENSEI/Promo, propose-delete after 7 days |

**One-time bulk action on first run:** Archive everything older than 30 days.

### sahil_ss@outlook.com (Job hunt — sophisticated triage, Flow 2)

**IMPORTANT:** This inbox has an existing manually-curated folder called **"Job Applications"** that Sahil has been using since Feb 2025. KENSEI MUST use this folder as the canonical place for applied/active job content. Do NOT create a parallel `KENSEI/Applied` folder.

Folder strategy:
- **Existing "Job Applications" folder** = canonical for `Applied` and `Active` (combine — Sahil's existing organisation didn't separate these)
- KENSEI may create sub-folders WITHIN "Job Applications" if useful (e.g., "Job Applications/Active", "Job Applications/Confirmations") — but only if Sahil approves on first run
- Other categories below get fresh KENSEI/ folders since they don't have existing equivalents

| Category | Default Action |
|----------|---------------|
| Applied | Move to existing "Job Applications" folder, keep |
| Active | Move to existing "Job Applications" folder, **flag urgently in digest** |
| Rejected | Move to KENSEI/Rejected (new), auto-archive after 30 days |
| JobAlerts | Move to KENSEI/JobAlerts (new), auto-archive after 14 days |
| Recruiter | Move to KENSEI/Recruiter (new), flag, never auto-delete |
| Uncertain | Flag in digest |

**Server-side rule (one-time setup):** Outlook 2 (sahil_saghir@hotmail.co.uk) → forward job-hunt patterns to here. Configured manually by Sahil in Outlook web settings.

---

## CONFIDENCE TIERS (UNIVERSAL)

For every email KENSEI categorises:

- **High confidence (>90%)** → execute the per-inbox default rule
- **Medium confidence (60-90%)** → label suspected category, flag in digest's "needs your eye" section
- **Low confidence (<60%)** → leave in inbox, flag as `Uncertain`, surface in digest

---

## SERVER-SIDE RULES (ONE-TIME SETUP)

KENSEI guides Sahil through these in browser (Outlook web settings + Gmail filters). Each rule shrinks the cleaner's workload.

### Outlook (web)

1. **Healthchecks alerts** (sahil_ss9@hotmail.com): emails from `noreply@healthchecks.io` → folder `KENSEI/Infrastructure`, skip inbox
2. **Microsoft security** (all 4 Outlooks): from `account-security-noreply@accountprotection.microsoft.com` → folder `KENSEI/Security`, skip inbox
3. **Outlook 2 → Outlook 3 job consolidation**: in sahil_saghir@hotmail.co.uk, emails from `*.workable.com`, `*.lever.co`, `*.greenhouse.io`, `*.jobs.com`, `recruitment` patterns → forward to sahil_ss@outlook.com
4. **Facebook/Instagram bulk** (matchdaymaestro@outlook.com): from `*.facebookmail.com`, `*.instagram.com` → folder `KENSEI/Social-Notifications`, mark read, skip inbox
5. **The Rundown AI newsletter** (sahil_ss9@hotmail.com): from `news@daily.therundown.ai` → folder `KENSEI/Newsletter`, skip inbox

### Gmail (filters)

1. **Healthchecks alerts** (saghir.sahil@gmail.com): from `noreply@healthchecks.io` → label `kensei/Infrastructure`, skip inbox
2. **Skool community high-volume** (sahilsaghir.ss9@gmail.com): from `*.skool.com`, `*.skool.community` → label `kensei/Skool`, mark as read, skip inbox

These rules are applied at platform level — they execute even if the cleaner cron fails. Belt-and-braces.

---

## DIGEST FORMAT

### Main Daily Digest (Flow 1 output)

Posted to Telegram Inbox topic at 08:00 UK.

```
📬 Mailbox digest — [Day] [Date], [Time]

Across 6 inboxes: [N] new overnight, [N] actions taken, [N] needing your eye.

🚨 Needs your attention ([N])
[Inbox-tag] [Sender]: [Subject summary]
... up to 10 entries, prioritised by urgency
... if more than 10, "Plus [N] more — reply 'show all' to see"

✅ Auto-organised ([N])
- [N] receipts → kensei/Receipts
- [N] newsletters → kensei/Newsletter (auto-archived after read)
- [N] Healthchecks → kensei/Infrastructure (skipped inbox)
- [other category counts]

🗑️ Proposed for deletion ([N])
[N] promo emails older than 7 days. Top senders:
- [Sender] ([N])
- [Sender] ([N])
- Other ([N])

Reply 'delete promos' to confirm all [N], or 'review' to see the list, or just ignore.
```

### Job Hunt Digest (Flow 2 output)

Posted to Telegram Inbox topic at 08:05 UK (separate message from main digest).

```
🎯 Job hunt digest — [Day] [Date], [Time]

🚨 Active opportunities ([N])
- [Company]: [Status, e.g. "Screening call Wednesday 10am (recruiter [Name])"]
- [Company]: [Status]
... full list, no truncation since this is the high-priority section

📝 New applications ([N])
- [Company] ([Role]): Submitted [date]
... 

❌ Rejections this week ([N])
- [Company list]

🔔 Job alerts overnight ([N])
Top matches:
- "[Subject]" via [Source] ([key detail like rate/location])
... up to 5 entries
- [N-5] others

Reply 'archive alerts' to clear all [N] alerts, or 'show top 5' for full details.
```

### Real-Time Urgent Ping (Flow 2 mid-day)

Posted to Telegram Inbox topic when triggered.

```
🎯 JOB HUNT URGENT — [Time]

[Sender]: [Subject]
[First 200 chars of body, if helpful]

Open in Outlook: [link]
```

Triggered by rule-based detection (no LLM call):
- Subject contains: "interview", "screening", "phone call", "schedule", "next step"
- Sender pattern: direct human (not `noreply@`, `donotreply@`)
- From folder: KENSEI/Active or sender previously categorised as Recruiter

---

## REPLY VERB PATTERNS

KENSEI parses Sahil's replies to digests. These verbs trigger actions:

- `delete promos` / `delete all promos` → execute queued promo deletions
- `archive alerts` / `archive job alerts` → archive overnight job alerts
- `review` / `show all` / `show list` → KENSEI sends full enumerated list
- `keep [number]` / `spare [number]` → marks specific emails to keep (e.g., "keep 3, 7, 12")
- `unsubscribe [sender]` → KENSEI generates unsubscribe (clicks unsubscribe link, or generates email)
- `mark [N] read` → bulk mark as read
- `done` / `ignore` / no reply → digest expires from queue after 7 days (auto-release)

**Approval gate behaviour:** When Sahil replies with destructive verbs (`delete`, `archive`, `unsubscribe`), KENSEI's tool calls fire through the Telegram messaging layer = the approval gate WILL fire. Sahil sees one final "About to delete N emails. Confirm?" prompt before execution.

This means **two-step approval** is the experience for destructive actions:
1. Sahil reads digest, replies "delete promos"
2. KENSEI: "About to delete 27 emails. Confirm?"
3. Sahil: "yes" / "confirm"
4. KENSEI executes

This is annoying but architecturally correct given current Hermes constraints. Document this in the skill.

---

## EDGE CASES (REQUIRED HANDLING)

### Edge Case 1: Mid-day urgent email
Real-time detection runs hourly 09:00-20:00 UK. Rule-based patterns trigger immediate Telegram alert. No LLM call needed for detection.

### Edge Case 2: Approval queue accumulates
Auto-release pending destructive actions after 7 days. Each digest mentions: "The following are >5 days old and will be released back to inbox if not actioned by [date]."

### Edge Case 3: Server-side rule misfires
KENSEI's daily digest includes a "Sanity check" footer if expected traffic patterns don't match (e.g., "Job alerts down 80% from average — possible rule misfire, check folder").

### Edge Case 4: Cron fails to run
healthchecks.io heartbeat for both cron jobs (mailbox-cleaner-main and mailbox-cleaner-jobhunt). If digest doesn't fire within 15 min of scheduled time, healthchecks.io alerts Sahil.

### Edge Case 5: KENSEI categorises personal correspondence as junk
Per Sahil's v1 decision: skip personal allowlist. Confidence-based safety (>90% threshold for auto-actions) is the only safeguard. Document this trade-off in the skill so it's revisitable.

### Edge Case 6: Inbox is empty / no new emails
Skip the digest entirely. Don't post "0 emails today" — that's noise. Just stay silent.

### Edge Case 7: New sender pattern KENSEI hasn't seen
Auto-flag as Uncertain. Surface in digest. Sahil's response trains future categorisation (manual confirmation creates a soft rule for next time).

---

## TELEMETRY / MONITORING

Skill should log:
- Run start/end timestamps
- Per-inbox emails processed
- Per-category counts assigned
- High/Medium/Low confidence breakdown
- Auto-actions executed
- Items queued for approval
- Errors / API failures

Save to `~/.hermes/cron/output/<job-id>/<timestamp>.md` (existing Hermes pattern).

KENSEI: include a "this week's stats" section in Sunday's digest:
```
📊 Weekly stats
- 412 emails processed across 7 inboxes
- 287 auto-organised
- 89 deleted (with your approval)
- 36 needed your eye
- KENSEI's accuracy: [TBD — based on your overrides]
```

---

## BUILD CHECKLIST FOR KENSEI

When you build this, work through in order:

### Phase 1: Foundation
- [ ] Read this entire spec, confirm understanding with Sahil
- [ ] Verify all 7 inboxes are accessible (3 Gmail via google_workspace MCP, 4 Outlook via ms-365-mcp-server)
- [ ] Verify Hermes cron infrastructure works (you've already used it for heartbeat)
- [ ] Verify Telegram delivery to Inbox topic works

### Phase 2: Existing folder audit (NEW — before any folder creation)
- [ ] List existing folders in each of the 7 inboxes
- [ ] Identify any existing folders that overlap with proposed KENSEI/ folders
- [ ] **Outlook 3:** confirm "Job Applications" folder still exists, use as canonical
- [ ] For each overlap: ask Sahil whether to use existing or create new
- [ ] Document the final folder mapping per inbox before starting build
- [ ] Only create new KENSEI/ folders for categories without existing equivalents

### Phase 3: Server-side rules (one-time setup, KENSEI guides Sahil)
- [ ] Outlook web rules (5 rules listed above)
- [ ] Gmail filters (2 filters listed above)
- [ ] Sahil confirms each rule is in place

### Phase 4: Skill files
- [ ] Create `~/.hermes/skills/mailbox-cleaner/SKILL.md` with the full skill description
- [ ] Create `~/.hermes/skills/mailbox-cleaner/main-prompt.md` with the Flow 1 prompt
- [ ] Create `~/.hermes/skills/mailbox-cleaner/jobhunt-prompt.md` with the Flow 2 prompt
- [ ] Create `~/.hermes/skills/mailbox-cleaner/urgent-detection-prompt.md` for the real-time detector
- [ ] Create `~/.hermes/skills/mailbox-cleaner/reply-parser.md` for the verb-pattern logic

### Phase 5: Cron jobs
- [ ] Create `mailbox-cleaner-main` cron: schedule `0 8 * * *`, attach skill, deliver to Telegram Inbox topic
- [ ] Create `mailbox-cleaner-jobhunt` cron: schedule `5 8 * * *`, attach skill, deliver to Telegram Inbox topic
- [ ] Create `mailbox-cleaner-urgent-detector` cron: schedule `0 9-20 * * *`, attach urgent-detection skill
- [ ] Add healthchecks.io heartbeats for each cron job (similar to existing heartbeat)

### Phase 6: Testing
- [ ] Force-run main cleaner manually, review output
- [ ] Force-run job hunt cleaner manually, review output
- [ ] Verify approval gate fires when Sahil replies "delete promos"
- [ ] Verify destructive actions execute correctly post-approval
- [ ] Run for 7 days, collect feedback, iterate

### Phase 7: Documentation
- [ ] Update `~/.hermes/runbooks/` with mailbox-cleaner runbook
- [ ] Update USER.md to reference the cleaner is operational
- [ ] Save this spec as a reference document in `~/.hermes/skills/mailbox-cleaner/spec.md`

---

## SAHIL'S CONFIRMED ANSWERS (LOCKED)

These were resolved during design session. KENSEI does NOT need to re-ask these.

1. **First-run bulk archive: NO.** No aggressive first-run cleanup. The cleaner runs daily from day one with no historical sweep. This means matchdaymaestro will look messy for the first few weeks while KENSEI processes new arrivals only. Old emails remain in inbox until manually addressed. Trade-off: slower visible progress, but KENSEI's mistakes are caught at low volume.

2. **Existing folders: use if fit-for-purpose, otherwise migrate.** Before creating any KENSEI/-prefixed folder, KENSEI must audit existing folders in each inbox. Specifically:
   - **Outlook 3** has an existing "Job Applications" folder (manually curated since Feb 2025). USE THIS as the canonical applied/active folder. Do NOT create `KENSEI/Applied` parallel to it.
   - For any other existing folders, KENSEI inspects naming and content. If a folder serves the same purpose as a proposed KENSEI/ folder, use the existing one. If it's a partial overlap, migrate items to a new KENSEI/ folder that better fits and propose archiving the old one.
   - When in doubt, ask Sahil before creating a new folder that might overlap with existing organisation.

3. **Weekly stats: YES.** Sunday digest includes a weekly stats section (see Telemetry section). Format: "this week's stats" appended to Sunday's main digest.

4. **Confidence thresholds: 90/60/40 (defaults).** No tightening for v1. Sahil may revisit after 2 weeks of usage data.

5. **Cron timezone: BST tracking (Europe/London).** Schedule `0 8 * * *` fires at 08:00 local UK time. Automatically follows BST↔GMT transitions. VPS is already configured for Europe/London — no additional cron config needed.

---

## ROLLBACK PROCEDURE

If the cleaner misbehaves badly:

```bash
# Pause both cron jobs
hermes cron pause <mailbox-cleaner-main-id>
hermes cron pause <mailbox-cleaner-jobhunt-id>
hermes cron pause <mailbox-cleaner-urgent-detector-id>

# Or remove entirely
hermes cron remove <mailbox-cleaner-main-id>
hermes cron remove <mailbox-cleaner-jobhunt-id>
hermes cron remove <mailbox-cleaner-urgent-detector-id>
```

Server-side rules (Gmail filters / Outlook rules) are platform-level, not deleted automatically. If Sahil wants to fully reverse, he removes them manually in each platform's web settings.

Existing labels/folders (kensei/* in Gmail, KENSEI/* in Outlook) can stay — they're empty containers without the cron jobs running. Or KENSEI can delete them on rollback request.

Email actions taken before rollback are NOT auto-reversible. Trash items can be restored from Trash folder for ~30 days. Archived items remain archived (recoverable via search).

---

## END OF SPEC

KENSEI: this is a v1 design. Once you've built and Sahil has used it for 1-2 weeks, expect refinements. The categorisation rules, confidence thresholds, and digest format will need tuning based on real data. Treat the first 2 weeks as calibration period.