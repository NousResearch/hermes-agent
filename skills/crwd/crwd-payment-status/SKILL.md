---
name: crwd-payment-status
description: "Tell a CRWD member their payment status and history — whether a gig's payout has been sent via Dot, when to expect it, and what they've earned. Figures out which gig they mean, checks approval state, and reads live Dot payout status. Use when a member asks 'did I get paid?', 'where's my money?', 'when will I be paid?', or 'show my payment history'."
version: 1.0.0
metadata:
  hermes:
    tags: [crwd, payment, payout, dot, paid, money, history, earnings, status]
    related_skills: [crwd-gig-discovery, crwd-gig-execution, crwd-reminders-followups, crwd-handoff, crwd-reference]
    requires_toolsets: [crwd]
---

# CRWD Payment Status

Answer "did I get paid?" against the member's **real** data: which gig, is it
approved yet, and what Dot says about the payout. Payments go out through **Dot**
(CRWD's payments partner) — the `dot` tool reads live payout status/history; the
`crwd_db` tool supplies the gig and approval context. Combine them.

## When to Use

- "Did I get paid for [gig]?" / "Where's my money?"
- "When will I be paid?" / "How long does payment take?"
- "Show my payment history." / "How much have I earned?"

## Procedure

1. **Member `user_id`** comes from the `[CRWD member]` context line — pass it
   straight through to both `crwd_db` and `dot`. Only use `crwd_db` `get_user`
   for a **different** person.
2. **Which gig?** If they're asking about a specific gig, resolve it first with
   `crwd_db` `get_gig_details` (confirm the `_id` when candidates are close) or
   `get_user_gigs`. For "all"/"history", skip this.
3. **Approval context (`crwd_db`)** — payment only flows **after** the work is
   approved. Check `get_user_gigs` (membership `hasPaid` / `isCompleted` /
   `status`) and, if useful, `get_user_receipts` (proof validation state). If a
   submission isn't approved yet, say that — there's nothing for Dot to send.
4. **Live payout (`dot`)** — once approved (or for a general history question):
   - one gig → `dot` `get_payment_status` with `user_id` (and `gig_id`).
   - all payouts → `dot` `get_payment_history` with `user_id`.
5. **Answer plainly, in a line or two:** approved yet? → has Dot sent it? (method
   + date if shown). Quote the **real payout amount** from the gig data, not a
   guess.
6. **Framing** (`skill_view("crwd-reference", "references/payments-dot.md")`):
   payout ≠ reimbursement (they keep the product); once approved, Dot typically
   lands in **1–2 business days** — say *typical, not guaranteed*, never promise a
   date.
7. **If the `dot` tool is unavailable or errors, don't hand off — fall back:** give
   the approval state from `crwd_db` plus the honest "1–2 business days after
   approval" framing. Only **escalate to `crwd-handoff`** for a genuine dispute you
   can't resolve from the data: Dot shows the payout **sent but the member never
   received it**, a wrong/missing amount, a refund request, or a **rejected**
   submission. Don't guess about money that's supposedly already gone out.

## Pitfalls

- **Don't claim the money landed** unless Dot actually reports it sent/paid.
  "Approved" and "paid" are different states — read them separately.
- Approval gates payment. If it's not approved/completed, there's no payout yet —
  don't send them to check their bank.
- Don't invent timing. "Typically 1–2 business days after approval" is the only
  promise, and even that is *typical*.
- `get_gig_details` returns *candidates* — confirm the right `_id` before quoting
  a gig's payout.
- Money disputes and rejections are a human's job — hand off, don't improvise.
- Keep it short: this is a phone chat widget.

## Verification

- Used the `[CRWD member]` `user_id` for both `crwd_db` and `dot`.
- Confirmed the right gig `_id` when the question was about a specific gig.
- Separated approval state (`crwd_db`) from Dot's payout state (`dot`) — didn't
  conflate "approved" with "paid".
- Quoted the real payout amount and framed timing as *typical*, not guaranteed.
- Handed off on Dot errors, "sent but not received" disputes, or rejections.
