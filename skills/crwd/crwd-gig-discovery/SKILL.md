---
name: crwd-gig-discovery
description: "Help a CRWD member find and understand gigs — what's available now, gig details (payout, deadline, store), applying, and approval status. Use when a member asks what gigs are open, about a specific gig, how to apply, or whether they're approved."
version: 1.0.0
metadata:
  hermes:
    tags: [crwd, gigs, campaigns, browse, apply, approval, payout, deadline]
    related_skills: [crwd-gig-execution, crwd-application-expert, crwd-reference]
    requires_toolsets: [crwd]
---

# CRWD Gig Discovery

Find gigs and explain them against the member's **real** data — not in the abstract.

## When to Use

- "What gigs are available?" / "Any new gigs?"
- "Tell me about the [X] gig" — payout, deadline, store, what's involved
- "How do I apply?" / "Am I approved yet?"
- "What gigs do I have?"

## Procedure

1. **Open gigs:** `crwd_db` action `list_active_gigs` — returns open gigs sorted by soonest
   end date, with payout, dates, stores, and proof type.
2. **A specific gig by name/text:** `get_gig_details` (fuzzy-matches, returns ranked
   candidates with an `_id`). **Confirm the right `_id`** before you quote details or use it
   elsewhere — if two candidates are close, ask which one they mean.
3. **Their gigs / approval state:** `get_user_gigs`. The current member's CRWD `user_id` is
   provided to you in context (a `[CRWD member]` line) — pass it straight through as `user_id`.
   Only fall back to `get_user` (by email/phone) if that line isn't present or you're looking
   up a **different** person. This shows the campaigns they're an active member of and their
   membership status.
4. Explain the flow against their **actual** state, not generically: browse → apply →
   **get approved** → perform → submit proof → get paid. If they're waiting on approval, say
   that; if approved, point them at what to do next (`crwd-gig-execution`).
5. Be precise on **payout, deadline, and estimated time** — quote the real numbers; never guess.

For the deeper lifecycle detail, load
`skill_view("crwd-reference", "references/gig-lifecycle.md")`.

## Pitfalls

- Don't quote a gig's payout/deadline from memory — look it up.
- `get_gig_details` returns *candidates*; picking the wrong `_id` sends the member to the
  wrong gig. Confirm first.
- Approval is gated by CRWD/the brand — you can report the state, but don't promise approval.

## Verification

- Details you gave (payout, deadline, store) came from `crwd_db`, not assumption.
- You confirmed the specific gig `_id` when there was any ambiguity.
- The member knows their current step in the flow and what to do next.
