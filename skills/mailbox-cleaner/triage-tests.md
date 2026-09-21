# Triage Tests — Main Cleaner Categorisation Scenarios

**Purpose:** Validate the main-cleaner prompt produces correct categorisation for each inbox and edge case.
**How to run:** Force-run the main cleaner cron (`hermes cron run <main-job-id>`) against a test inbox with sample emails, or review these scenarios manually against the prompt logic.
**Created:** 19/05/2026

---

## Scenario 1: Amazon Receipt (saghir.sahil@gmail.com)

| Field | Value |
|-------|-------|
| From | `order-update@amazon.co.uk` |
| Subject | "Your Amazon.co.uk order #204-1234567-8901234 has been dispatched" |
| Body | "Your order has been dispatched. Estimated delivery: Friday." |
| **Expected category** | Receipts (HIGH) |
| **Expected action** | Label `kensei/Receipts`, keep in inbox |

**Rule match:** From `amazon*` + receipt language. HIGH confidence (>90%).

---

## Scenario 2: Skool Community Post (sahilsaghir.ss9@gmail.com)

| Field | Value |
|-------|-------|
| From | `notifications@skool.com` |
| Subject | "New post in Sahil's Community: 'How to build AI agents'" |
| Body | "Check out this new discussion in the community..." |
| **Expected category** | Skool (HIGH) |
| **Expected action** | Label `kensei/Skool`, propose-delete after 7 days |

**Rule match:** From `*.skool.com`. HIGH confidence. Server-side filter should have caught this — if it lands in inbox, flag as possible misfire.

---

## Scenario 3: Tesco Clubcard Promotion (sahil_ss9@hotmail.com)

| Field | Value |
|-------|-------|
| From | `promotions@tesco.com` |
| Subject | "Your Clubcard points are expiring — spend them now!" |
| Body | "You have 500 Clubcard points that expire next month. Shop now..." |
| **Expected category** | Promo (HIGH) |
| **Expected action** | Move to `KENSEI/Promo`, propose-delete after 7 days — **QUEUED, not auto-deleted** |

**HARD RULE:** sahil_ss9@hotmail.com is conservative. Never auto-delete. All destructive actions queued for approval.

---

## Scenario 4: LinkedIn Job Alert (sahil_saghir@hotmail.co.uk)

| Field | Value |
|-------|-------|
| From | `jobs-listings@linkedin.com` |
| Subject | "5 new Senior Product Manager jobs in Nottingham" |
| Body | "Here are the latest PM roles matching your profile..." |
| **Expected category** | Promo (or Uncertain if server rule was meant to forward this) |
| **Expected action** | Move to `KENSEI/Promo`, propose-delete after 7 days OR flag as possible misfire |

**Note:** If the server-side rule 3 (job hunt forward) is in place, this should have been forwarded to sahil_ss@outlook.com. Flag in digest if this lands here.

---

## Scenario 5: Baby's Days Update (sahil_ss9@hotmail.com)

| Field | Value |
|-------|-------|
| From | `notifications@babysdays.com` |
| Subject | "Settling-in report for this morning" |
| Body | "Your child attended nursery today. Here's the daily report..." |
| **Expected category** | Childcare (HIGH) |
| **Expected action** | Move to `KENSEI/Childcare`, keep indefinitely, flag in digest |

**HARD RULE:** Keep indefinitely — these are important family records.

---

## Scenario 6: New Sender — Unknown Pattern (saghir.sahil@gmail.com)

| Field | Value |
|-------|-------|
| From | `newsletter@quirky-startup.io` |
| Subject | "Our new product just launched 🚀" |
| Body | "We've been building in stealth for 12 months and we're excited to share..." |
| **Expected category** | Uncertain (LOW) |
| **Expected action** | Leave in inbox, flag as Uncertain in digest, no label applied |

**Edge case:** New sender pattern KENSEI hasn't seen. Auto-flag as Uncertain.

---

## Scenario 7: Facebook Notification (matchdaymaestro@outlook.com)

| Field | Value |
|-------|-------|
| From | `notification@facebookmail.com` |
| Subject | "Your Page Insights for this week" |
| Body | "Here's how your MatchdayMaestro page performed this week..." |
| **Expected category** | Social-Notifications (HIGH) |
| **Expected action** | Move to `KENSEI/Social-Notifications`, archive immediately |

**Note:** Server-side rule 4 should skip inbox entirely for this sender.

---

## Scenario 8: Stripe Invoice (fusionfirststudios@gmail.com)

| Field | Value |
|-------|-------|
| From | `receipts@stripe.com` |
| Subject | "Your receipt from Stripe — £29.00" |
| Body | "Thanks for your payment. Here's the receipt..." |
| **Expected category** | Receipts (HIGH) |
| **Expected action** | Label `kensei/Receipts`, keep |

---

## Scenario 9: GitHub Dependabot Alert (saghir.sahil@gmail.com)

| Field | Value |
|-------|-------|
| From | `dependabot@github.com` |
| Subject | "[dependabot] 3 dependency updates in kensei-agent" |
| Body | "We've detected vulnerabilities in the following packages..." |
| **Expected category** | Service (HIGH) |
| **Expected action** | Label `kensei/Service`, flag in digest |

---

## Scenario 10: All Inboxes Empty

| Condition | Expected behaviour |
|-----------|-------------------|
| 0 new emails across all 6 inboxes | Post short: "📬 No new mail overnight across 6 inboxes." No expandable block. |

**HARD RULE:** Empty inbox → skip silently. Don't name-check individual zero-traffic inboxes.

---

## Scenario 11: MCP Transport Failure on One Inbox

| Condition | Expected behaviour |
|-----------|-------------------|
| saghir.sahil@gmail.com returns ClosedResourceError. Other 5 inboxes OK. | Log the error. Process the other 5. Include error note in digest. Don't crash entire run. |

---

## Scenario 12: Approval Queue >5 Days Old

| Condition | Expected behaviour |
|-----------|-------------------|
| 15 promo items queued for 6 days | Add warning to digest: "⚠️ 15 items in approval queue are >5 days old — they'll auto-release on DD/MM/YY if not actioned." |

---

## Test Log

| Date | Scenario | Result | Notes |
|------|----------|--------|-------|
| | 1: Amazon receipt | ☐ | |
| | 2: Skool post | ☐ | |
| | 3: Tesco promo (family) | ☐ | |
| | 4: LinkedIn alert (promo inbox) | ☐ | |
| | 5: Baby's Days | ☐ | |
| | 6: New sender | ☐ | |
| | 7: Facebook notif | ☐ | |
| | 8: Stripe receipt | ☐ | |
| | 9: GitHub alert | ☐ | |
| | 10: Empty all | ☐ | |
| | 11: MCP failure | ☐ | |
| | 12: Queue warning | ☐ | |
