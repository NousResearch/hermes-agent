# Gig lifecycle — browse → paid

Every member goes through the same flow:

1. **Browse** — gigs are listed with a payout, a deadline ("Due in X days"), an estimated
   time ("Takes X hours"), and a source (Walmart, Amazon, JoinCRWD, etc.).
2. **Apply** — the member applies for a gig in the CRWD app/website.
3. **Get approved** — CRWD or the brand approves the application before the member can start.
4. **Perform the gig** — see gig types below.
5. **Submit proof** — receipts, screenshots, review links, UGC video/photo links, uploaded
   to the CRWD dashboard.
6. **Get paid** — payment is triggered once proof is reviewed and verified.

## Gig types

**Live gigs (in-store):**
- Member goes to a physical store (often Walmart or Target), finds the product, buys it —
  sometimes with a specific requirement like "two purchases with two different payment
  methods."
- Member creates content per the gig's "approved concepts" (a natural, relatable,
  non-scripted UGC video or photo showing the product clearly).
- Proof: photo of the receipt, the store location, and a link to the UGC/content.

**Online gigs:**
- Member orders the product (commonly via Amazon), leaves a review, and submits order +
  review screenshots.
- Payment process starts once that proof is submitted and verified.

Every gig has a **payout**, a **deadline**, and an **estimated time to complete**. Be precise
about these when asked, and never guess if you don't have the real numbers in front of you —
look them up with `crwd_db`.

**Live (`gig_type: "irl"`) and online are the two types in the data.** If a gig doesn't fit
either cleanly — a sampling run, a foot-traffic activation, or anything with unusual
requirements — don't force it into the wrong bucket. Describe it from its **actual** data
(payout, `type_of_work_proof`, source, requirements) and, if what's being asked for is
genuinely unclear, hand off rather than guess (`crwd-handoff`).
