# Reply Parser — Telegram Verb Patterns

# Handles Sahil's replies to mailbox-cleaner digests.
# Parses verb patterns, enforces two-step approval for destructive actions, and manages auto-release.

This is the canonical reference for all reply verbs parsed from Sahil's Telegram replies to mailbox-cleaner digests. Both the cron jobs and the Telegram message handler should reference this document for consistent behaviour.

## ARCHITECTURE

Two integration modes:

1. **Cron-based polling (v1):** A separate cron job checks for replies to recent digest messages every 5 minutes during working hours. Simplest to implement — works within current Hermes architecture.

2. **Telegram message handler (v2):** Direct integration into the existing Telegram message handler. Lower latency, but requires modifying the gateway message router.

v1 ships first. v2 when gateway architecture supports it.

## VERB PATTERNS

All patterns are case-insensitive. Sahil's replies are parsed for the first matching verb pattern.

### Destructive Verbs (require two-step approval)

| Pattern | Action | Example |
|---------|--------|---------|
| `delete promos` / `delete all promos` | Execute all queued promo deletions across all inboxes | "delete promos" |
| `delete [N] promos` | Delete N most recent queued promos | "delete 12 promos" |
| `delete [sender]` | Delete all queued emails from a specific sender | "delete amazon" |
| `archive alerts` / `archive job alerts` | Archive all overnight job alerts from Flow 2 digest | "archive alerts" |
| `unsubscribe [sender]` | Generate unsubscribe action for sender (click link or craft email) | "unsubscribe asos" |

### Non-Destructive Verbs (execute immediately)

| Pattern | Action | Example |
|---------|--------|---------|
| `review` / `show all` / `show list` | Send full enumerated list of queued items | "review" |
| `show top [N]` / `show alerts` | Show top N job alerts from Flow 2 digest | "show top 5" |
| `keep [numbers]` / `spare [numbers]` | Mark specific items to keep (remove from queue) | "keep 3, 7, 12" |
| `mark [N] read` | Bulk mark N most recent emails as read | "mark 50 read" |
| `done` / `ignore` | Acknowledge digest, release queue items from pending state | "done" |

### Unknown / No Match

If no pattern matches, KENSEI responds with:
```
❓ Didn't recognise that. Try:
• delete promos — clear queued promos
• review — see full list
• archive alerts — clear job alerts
• done — acknowledge and release queue
```

## TWO-STEP APPROVAL FLOW

For destructive verbs (`delete`, `archive`, `unsubscribe`):

```
Step 1: Sahil replies "delete promos"
Step 2: KENSEI responds with confirmation prompt
        → "About to delete N emails across M inboxes. Confirm?"
Step 3: Sahil replies "yes" / "confirm"
Step 4: KENSEI executes and responds with result
        → "✅ Done: N emails deleted, N kept."
```

### Confirmation Prompt Format

```html
⚠️ <b>Confirm deletion</b>

• <b>27 promo emails</b> across 4 inboxes
• Top senders: Amazon (8), ASOS (5), Tesco (4), other (10)
• All are older than 7 days

Reply <b>yes</b> to confirm, or <b>keep 3, 7</b> to spare specific items.
```

### Execution Confirmation Format

```html
✅ <b>Deletion complete</b>

• 27 promos deleted
• 0 kept (all confirmed)
• Inboxes cleaned: saghir.sahil@gmail.com, sahilsaghir.ss9@gmail.com, sahil_saghir@hotmail.co.uk, matchdaymaestro@outlook.com
```

### Confirmation Verbs Accepted

- `yes`
- `confirm`
- `y`
- `do it`
- `go ahead`
- `proceed`

### Cancellation

If Sahil replies with anything else during the confirmation window, treat as cancellation:
```
❌ Deletion cancelled. Queue preserved.
```

## AUTO-RELEASE (7-DAY TIMEOUT)

Queued items auto-release after 7 days of no response. The daily digest includes a warning for items >5 days old:

```
⚠️ N items in approval queue are >5 days old — they'll auto-release on DD/MM/YY if not actioned.
```

When items are about to auto-release (day 7):
```
⏰ N items auto-released from queue today (expired after 7 days). They remain in their labelled folders.
```

## DIGEST EXPIRY

- Each digest message has a 7-day reply window.
- After 7 days, the digest and its queued actions are expired.
- Emails remain categorised/labelled. Only the approval queue is released.
- Sahil can still manually manage emails via platform (Gmail web, Outlook web).

## STATE TRACKING

The reply parser needs to track state across runs:

| State field | Where stored | Purpose |
|-------------|-------------|---------|
| Which digest is "current" | Telemetry log (`~/.hermes/cron/output/`) | Know which digest Sahil is replying to |
| What's in the queue | Telemetry log (queued items section) | Know what can be actioned |
| Pending confirmations | Memory or a state file | Know if a destructive action is awaiting step 2 |
| Expired items count | Telemetry log (auto-release section) | Track weekly stats |

### State File Approach (Recommended for v1)

Write a simple JSON state file at `~/.hermes/skills/mailbox-cleaner/state.json`:

```json
{
  "last_main_digest": {
    "message_id": null,
    "timestamp": "DD/MM/YY HH:MM:SS",
    "queued": {
      "promos": 27,
      "items": []
    },
    "expires": "DD/MM/YY HH:MM:SS"
  },
  "last_jobhunt_digest": {
    "message_id": null,
    "timestamp": "DD/MM/YY HH:MM:SS",
    "queued": {
      "alerts": 15,
      "items": []
    },
    "expires": "DD/MM/YY HH:MM:SS"
  },
  "pending_confirmation": null
}
```

## REPLY HANDLER CRON (v1)

```
Schedule: */5 9-20 * * * (every 5 min, 09:00-20:00 UK, Mon-Fri)
```

The cron job:
1. Reads the state file
2. Checks for Telegram replies to the last digest (via Telegram API or message history)
3. If a reply is found, parses it against verb patterns
4. If destructive verb: sends confirmation prompt, updates `pending_confirmation` state
5. If confirmation verb: executes, reports result, clears `pending_confirmation`
6. If non-destructive verb: executes immediately, reports result
7. Checks for expired items, processes auto-release

## INTEGRATION TESTING

Before shipping, verify the full flow:

| Test | Expected result |
|------|----------------|
| Sahil replies "delete promos" to main digest | KENSEI sends confirmation prompt with N count |
| Sahil replies "yes" to confirmation | KENSEI deletes, sends "Done: N deleted" |
| Sahil replies "no" to confirmation | KENSEI cancels, sends "Deletion cancelled" |
| Sahil replies "review" to main digest | KENSEI sends enumerated list |
| Sahil replies "archive alerts" to job hunt digest | KENSEI sends confirmation for N alerts |
| Sahil ignores digest for 7 days | Auto-release fires, items remain in folders |
| Sahil replies gibberish | KENSEI sends "Didn't recognise that" help message |
| Sahil replies "delete promos" after queue expired | KENSEI: "Queue expired on DD/MM/YY. Items remain in folders." |

## IMPORTANT CONSTRAINTS

- Never execute destructive actions without two-step confirmation. The approval gate is non-negotiable.
- Never auto-release before 7 days. Even if Sahil replies "done" — that acknowledges the digest but doesn't trigger deletions.
- Never delete from sahil_ss9@hotmail.com. The queue for this inbox is display-only — all destructive actions on this inbox are blocked at the per-inbox rules level.
- Never match verb patterns across digest boundaries. A "delete promos" reply to the job hunt digest should only apply to job hunt queued items.
- The state file must be atomic. Use write-to-temp + rename pattern to prevent corruption.
