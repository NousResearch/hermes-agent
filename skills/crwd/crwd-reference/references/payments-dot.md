# Payments — Dot, timing, reimbursement

## How members get paid

- Payments are processed through **Dot** (CRWD's payments partner).
- Once a submission is **approved**, payout is sent via Dot. Typical turnaround from
  approval to payment landing is **1–2 business days**.
- Frame this as *typical, not guaranteed* — you can't promise timing on CRWD's behalf.

## Live payment status is NOT wired up yet

- The Dot live-status integration is **coming soon** — it is not available right now.
- Do **not** claim to check real-time "has the money landed" status. Instead:
  - Explain the normal process honestly ("once your proof is approved, Dot sends payment,
    usually within 1–2 business days").
  - Check what you *can* see with `crwd_db` (submission/approval state on the membership /
    receipt records).
  - If the member needs a definitive answer on money already sent, **hand off** (see the
    `crwd-handoff` skill) — don't guess or paper over the gap.

## Payout ≠ reimbursement

- For **live gigs**, the member **keeps the product** they bought. The payout is the **fee
  for completing the gig**, not a refund of the purchase.
- If a member asks "do I get my money back for what I bought?" → the answer is **no**: they
  keep the item, and the payout is separate. This is a fixed fact — say it plainly, don't hedge.
