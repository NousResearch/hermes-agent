---
name: crwd-gig-execution
description: "Walk a CRWD member through doing a gig end to end — buying the product (with buy links), meeting special requirements, creating the UGC content, and submitting the right proof so it isn't rejected. Use when a member asks how to do a gig, where to buy, what content to make, or what proof to submit."
version: 1.0.0
metadata:
  hermes:
    tags: [crwd, gig, execution, buy, product, ugc, content, proof, receipt, submission]
    related_skills: [crwd-store-locator, crwd-reference, crwd-handoff]
    requires_toolsets: [crwd]
---

# CRWD Gig Execution

Get the member from "approved" to "proof submitted" — buying, content, and proof, in one
skill (proof is just the tail of doing the gig).

## When to Use

- "How do I do this gig?" / "What are the steps?"
- "Where do I buy the product?" / "What's the buy link?"
- "What kind of video/photo do I need to make?"
- "What proof do I submit?" / "What do I upload?"

## Procedure

1. **Confirm the gig and its type** (live vs online) with `crwd_db` `get_gig_details`.
2. **Surface the exact product + buy link.** Use `get_user_products` (the member's approved
   products for their gigs — product name + `product_url`). The current member's CRWD
   `user_id` is provided in context (a `[CRWD member]` line) — pass it straight through;
   `get_user_products` and `get_user_receipts` both take that `user_id`. Only use `get_user`
   for a different person. Give them the real link, don't describe it vaguely.
3. **Live gig steps:** go to the store (see `crwd-store-locator` if they need to find it),
   buy the product, and **call out any special requirement precisely** — e.g. *two purchases
   with two different payment methods* means two separate transactions and two receipts.
   Then create the content: a natural, non-scripted UGC video/photo showing the product
   clearly, matching the gig's "approved concepts."
4. **Online gig steps:** order the product (commonly Amazon) via the buy link, then leave a
   review per the gig's instructions.
5. **Proof — tell them the exact format** so it isn't rejected:
   - Live: receipt photo (readable, showing the product), store location, and the UGC content
     link. Both receipts if there's a two-purchase requirement.
   - Online: order screenshot + review screenshot.
   - Full detail: `skill_view("crwd-reference", "references/proof-requirements.md")`.
6. **Check submission status** if they ask "did it go through?" — `get_user_receipts` shows
   receipt/proof validation state (pass/fail + reason).
7. **If a submission is rejected → hand off** (`crwd-handoff`). Do not guess the rejection
   reason or coach a resubmission yourself — that's a human's job.

## Pitfalls

- Payout is **not** reimbursement — the member keeps what they bought (see
  `references/payments-dot.md`). Don't imply they'll be refunded for the purchase.
- Don't paraphrase a buy link or requirement — quote the real product URL and the exact
  requirement from the gig data.
- Rejected submissions always go to a human. Never coach a resubmission.

## Verification

- Product name + real buy link came from `get_user_products`.
- Any special requirement (e.g. two payment methods) was stated precisely.
- The member knows the exact proof to submit for their gig type.
- Rejections were handed off, not self-diagnosed.
