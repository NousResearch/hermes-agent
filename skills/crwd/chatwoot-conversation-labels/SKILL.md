---
name: chatwoot-conversation-labels
description: "Use every Chatwoot turn to classify the conversation with one or more support labels via chatwoot_labels."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [crwd, chatwoot, labels, classification, triage]
    related_skills: [crwd-handoff, crwd-payment-status, crwd-gig-execution, crwd-troubleshooting]
    requires_toolsets: [chatwoot]
---

# Chatwoot Conversation Labels

Internal triage only — classify each Chatwoot conversation with one or more
labels so human agents can filter the inbox. **Never mention labels to the
member.**

## When to Use

- **Every agent turn** on Chatwoot (after you understand the latest member
  message and thread context).
- Re-run classification when the topic shifts mid-conversation.

Don't use for: CLI, Telegram, or other non-Chatwoot platforms (`chatwoot_labels`
no-ops gracefully there).

## Quick Reference

| Member intent | Label(s) |
|---------------|----------|
| Finding gigs, Explore, availability | `gig-discovery` |
| Doing/submitting a gig, proof | `gig-execution` |
| Paid? when? payout history | `payment-payout` |
| Where in app, Home vs Explore | `app-navigation` |
| Broken page/link/button | `troubleshooting` |
| Frustration, dispute, rejection, human needed | `handoff-escalation` (+ topic label) |
| Ban, account, membership | `account-membership` |
| None of the above | `general-inquiry` |

More examples: `skill_view("chatwoot-conversation-labels", "references/label-taxonomy.md")`.

## Procedure (every turn)

Labels are **applied automatically** after each turn via a Chatwoot plugin hook.
You do not need to call `chatwoot_labels` for normal triage — the hook classifies
from the member's message and assigns 1–3 labels. Optionally call the tool if you
want to **override** the auto-classification (e.g. you know the topic better).

1. **Bootstrap** (first conversation turn, optional): `chatwoot_labels` with
   `action=create_labels_if_not_exists` — the auto-hook also bootstraps on assign.
2. **Classify** mentally from the full thread — the auto-hook uses the same
   predefined set (see Quick Reference). Override via `assign_labels` when needed.
3. **Do not mention labels to the member** — internal triage only.
4. **Done when:** your member-facing reply is sent (labels are applied in the background).

## Multi-label examples

- Member asks why payout is late **and** the payout page won't load →
  `["payment-payout", "troubleshooting"]`
- Member is frustrated about a rejected submission →
  `["gig-execution", "handoff-escalation"]`
- Simple "where is Explore?" → `["app-navigation"]`

## Common Pitfalls

1. **Mentioning labels to the member** — internal only; never say "I've tagged
   this as payment-payout."
2. **Too many labels** — stick to 1–3 that match the *current* turn's intent.
3. **Skipping every turn** — re-classify each turn so labels stay accurate when
   topics shift.
4. **Forgetting bootstrap** — `assign_labels` auto-creates missing predefined
   labels, but calling `create_labels_if_not_exists` on turn 1 avoids mid-thread
   422 errors on strict Chatwoot installs.

## Verification Checklist

- [ ] `create_labels_if_not_exists` called once at conversation start (when on Chatwoot)
- [ ] 1–3 labels chosen from the predefined set for this turn's intent
- [ ] `assign_labels` called with the label array before or after the member reply
- [ ] Member was not told about labels
- [ ] On handoff, `handoff-escalation` is included with the topic label
