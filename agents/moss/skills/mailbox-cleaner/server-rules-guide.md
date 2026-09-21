# Server-Side Rules Guide — One-Time Setup

**Phase 2 of mailbox-cleaner implementation.**
**Created:** 19/05/2026
**Purpose:** Step-by-step guides for applying the 7 platform-level rules (5 Outlook, 2 Gmail) that reduce the cleaner's workload.
**Action:** Sahil applies these manually in browser — KENSEI cannot automate Outlook web rules or Gmail filters.

---

## Why Server-Side Rules?

These rules execute at platform level — they run even if a mailbox-cleaner cron job fails. They handle high-volume patterns that would otherwise overwhelm the daily digest with noise:

- Healthchecks alerts that fire hourly
- Facebook/Instagram notifications that arrive multiple times daily
- Skool community posts that arrive 10-20 times per day
- Automated job alert emails that have no actionable news

With these rules in place, the daily cleaner only processes meaningful signal: receipts, newsletters, actual recruiter outreach, and uncategorised mail.

---

## Rule 1: Healthchecks Alerts → KENSEI/Infrastructure (Outlook)

**Applies to:** sahil_ss9@hotmail.com (family inbox)

### Steps

1. Open https://outlook.live.com/mail/0/ in browser (logged in as sahil_ss9@hotmail.com).
2. Click **Settings** (gear icon, top right) → **Mail** → **Rules** → **Add new rule**.
3. Configure:
   - **Name:** `Healthchecks -> KENSEI/Infrastructure`
   - **When a new message arrives:** Apply rule when: `From` → `noreply@healthchecks.io`
   - **Do all of:** Move to folder `KENSEI/Infrastructure`, mark as read, stop processing more rules
4. Click **Save**.

### Verification

Send a test ping from healthchecks.io dashboard (or wait for next scheduled ping). It should land in KENSEI/Infrastructure, not the inbox.

---

## Rule 2: Microsoft Security Alerts → KENSEI/Security (All 4 Outlooks)

**Applies to:** sahil_ss9@hotmail.com, sahil_saghir@hotmail.co.uk, sahil_ss@outlook.com, matchdaymaestro@outlook.com

### Steps

Apply this rule on ALL 4 Outlook accounts.

1. Open Outlook web for each account.
2. **Settings** → **Mail** → **Rules** → **Add new rule**.
3. Configure:
   - **Name:** `Microsoft Security -> KENSEI/Security`
   - **When a new message arrives:** `From` → `account-security-noreply@accountprotection.microsoft.com`
   - **Do all of:** Move to folder `KENSEI/Security`, mark as read, stop processing more rules
4. Click **Save**.

### KENSEI Folder Setup

Before this rule works, the `KENSEI/Security` folder must exist in each Outlook account. The cleaner creates it automatically on first run, or you can create it manually now:

- In Outlook web, right-click your inbox → **New folder** → name `KENSEI/Security`

---

## Rule 3: Job Hunt Consolidation: sahil_saghir@hotmail.co.uk → Forward to sahil_ss@outlook.com

**Applies to:** sahil_saghir@hotmail.co.uk (promo dump inbox)

### Steps

1. Open Outlook web for sahil_saghir@hotmail.co.uk.
2. **Settings** → **Mail** → **Rules** → **Add new rule**.
3. Configure:
   - **Name:** `Job hunt -> forward to sahil_ss@outlook.com`
   - **When a new message arrives:** From contains any of:
     - `*.workable.com`
     - `*.lever.co`
     - `*.greenhouse.io`
     - `recruitment@`
     - `jobs@`
     - `careers@`
   - **Do all of:** Forward the message to `sahil_ss@outlook.com`, stop processing more rules
4. Click **Save**.

### Why This Exists

Some job application emails land in the promo inbox (sahil_saghir@hotmail.co.uk) instead of the main job hunt inbox. This rule catches them and consolidates everything into sahil_ss@outlook.com so the job hunt cleaner (Flow 2) has complete coverage.

### After Setup

The main cleaner (Flow 1) will still check sahil_saghir@hotmail.co.uk and flag any job-hunt senders that slipped through. This is a belt-and-braces measure.

---

## Rule 4: Facebook/Instagram Bulk → KENSEI/Social-Notifications (Outlook)

**Applies to:** matchdaymaestro@outlook.com

### Steps

1. Open Outlook web for matchdaymaestro@outlook.com.
2. **Settings** → **Mail** → **Rules** → **Add new rule**.
3. Configure:
   - **Name:** `Social notifs -> KENSEI/Social-Notifications`
   - **When a new message arrives:** From contains any of:
     - `*.facebookmail.com`
     - `*.instagram.com`
   - **Do all of:** Move to folder `KENSEI/Social-Notifications`, mark as read, stop processing more rules
4. Click **Save**.

### KENSEI Folder Setup

Create folder `KENSEI/Social-Notifications` in this inbox before the rule takes effect.

---

## Rule 5: The Rundown AI Newsletter → KENSEI/Newsletter (Outlook)

**Applies to:** sahil_ss9@hotmail.com (family inbox)

### Steps

1. Open Outlook web for sahil_ss9@hotmail.com.
2. **Settings** → **Mail** → **Rules** → **Add new rule**.
3. Configure:
   - **Name:** `Rundown AI -> KENSEI/Newsletter`
   - **When a new message arrives:** `From` → `news@daily.therundown.ai`
   - **Do all of:** Move to folder `KENSEI/Newsletter`, mark as read, stop processing more rules
4. Click **Save**.

---

## Rule 6: Healthchecks → kensei/Infrastructure (Gmail Filter)

**Applies to:** saghir.sahil@gmail.com (primary personal)

### Steps

1. Open Gmail web for saghir.sahil@gmail.com.
2. Click the search bar → **Show search options** (down arrow on right).
3. Configure:
   - **From:** `noreply@healthchecks.io`
   - Click **Create filter**.
4. Check:
   - [x] Skip the Inbox (Archive it)
   - [x] Apply the label: `kensei/Infrastructure` (create if not exists)
5. Click **Create filter**.

### Label Setup

If `kensei/Infrastructure` doesn't exist in your Gmail labels, create it first:
- In Gmail, scroll left sidebar to bottom → **More** → **Create new label**
- Name: `kensei/Infrastructure`
- Nest under: (none — it's a top-level label)

---

## Rule 7: Skool High-Volume → kensei/Skool (Gmail Filter)

**Applies to:** sahilsaghir.ss9@gmail.com (dev/learning inbox)

### Steps

1. Open Gmail web for sahilsaghir.ss9@gmail.com.
2. Click search bar → **Show search options**.
3. Configure:
   - **From:** `*.skool.com` OR `*.skool.community` (you may need separate filters for each)
   - Click **Create filter**.
4. Check:
   - [x] Skip the Inbox (Archive it)
   - [x] Mark as read
   - [x] Apply the label: `kensei/Skool` (create if not exists)
5. Click **Create filter**.

### Why Separate Filters

Gmail filters don't support wildcard domains in a single entry the same way Outlook does. You may need 2 separate filters:
- Filter 1: From `*.skool.com` → label `kensei/Skool`, skip inbox, mark read
- Filter 2: From `*.skool.community` → same actions

---

## Post-Setup Verification Checklist

After all 7 rules are in place:

| # | Rule | Account | Expected effect | Verified? |
|---|------|---------|----------------|-----------|
| 1 | Healthchecks → Infrastructure | sahil_ss9@hotmail.com | Healthchecks land in KENSEI/Infrastructure | ☐ |
| 2 | MS Security → Security | All 4 Outlooks | Security alerts skip inbox | ☐ |
| 3 | Job hunt forward | sahil_saghir@hotmail.co.uk | Job emails auto-forward to sahil_ss | ☐ |
| 4 | Social → Social-Notifications | matchdaymaestro@outlook.com | FB/IG bulk to KENSEI/Social-Notifications | ☐ |
| 5 | Rundown AI → Newsletter | sahil_ss9@hotmail.com | Rundown AI to KENSEI/Newsletter | ☐ |
| 6 | Healthchecks → Infrastructure | saghir.sahil@gmail.com | Healthchecks archived under label | ☐ |
| 7 | Skool → Skool | sahilsaghir.ss9@gmail.com | Skool posts archived/marked read | ☐ |

---

## Rollback

To undo a rule:

- **Outlook:** Settings → Mail → Rules → find rule by name → Delete
- **Gmail:** Settings → Filters and Blocked Addresses → find filter → Delete

Deleting rules is immediate. No lingering side effects.
