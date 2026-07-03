# Payments — Dot, timing, reimbursement

## How members get paid

- Payments are processed through **Dot** (CRWD's payments partner).
- Once a submission is **approved**, payout is sent via Dot. Typical turnaround from
  approval to payment landing is **1–2 business days**.
- Frame this as *typical, not guaranteed* — you can't promise timing on CRWD's behalf.

## Live payment status — via the `dot` tool

- Live Dot payout status **is** available through the `dot` tool (used by the
  `crwd-payment-status` skill): `get_payment_status` (optionally for one gig) and
  `get_payment_history` for a member.
- Still frame timing honestly: "once your proof is approved, Dot sends payment, usually
  within 1–2 business days" — *typical, not guaranteed*. Don't promise a date.
- Read approval and payment as **separate** states: `crwd_db` shows whether the work is
  approved (`hasPaid` / `isCompleted` / receipt status); `dot` shows whether Dot has
  actually sent the payout. "Approved" ≠ "paid".
- If the `dot` tool errors or isn't configured, fall back to the `crwd_db` approval state
  plus the honest 1–2 business-day framing.
- For a genuine money dispute — Dot reports **sent but the member never received it** — or
  anything you can't answer confidently, **hand off** (`crwd-handoff`). Don't guess about
  money already gone out.

## Payout ≠ reimbursement

- For **live gigs**, the member **keeps the product** they bought. The payout is the **fee
  for completing the gig**, not a refund of the purchase.
- If a member asks "do I get my money back for what I bought?" → the answer is **no**: they
  keep the item, and the payout is separate. This is a fixed fact — say it plainly, don't hedge.
