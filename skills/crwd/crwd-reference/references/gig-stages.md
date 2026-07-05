# Gig stages (maintainer reference)

Machine-readable `stage` values returned by `crwd_db` action `get_user_gig_status`
and injected as `[CRWD gig context]` on Chatwoot gig-related turns.

| Stage | Meaning |
|-------|---------|
| `waitlisted` | Applied but `isAccepted` is false |
| `rejected` | Membership has `rejectionReason` — hand off |
| `pending_approval` | Accepted path but not `isApproved` |
| `need_purchase` | Approved, no `user_product_purchases` row |
| `need_receipt` | Purchased, no receipt uploaded |
| `receipt_review` | Receipt submitted, awaiting approval |
| `receipt_rejected` | Receipt rejected — hand off |
| `need_review` | Receipt approved, review/UGC not submitted |
| `review_review` | Review submitted, awaiting approval |
| `review_rejected` | Review rejected — hand off |
| `awaiting_payout` | All proof approved, payout not yet marked paid |
| `paid` | `hasPaid` is true on membership |

Progress sources: `added_crwd_members`, `user_product_purchases`,
`gig_store_orders`, `gig_product_reviews`, `order_receipt_reviews`.
