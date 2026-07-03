# Case state machine

One case = one (subject x broker). `pdd.py record` validates every transition against this table and
appends it to `audit.jsonl`. Authoritative definition lives in `scripts/ledger.py`.

## States

| State | Meaning |
|---|---|
| `new` | Case created, nothing done |
| `searching` | Scan in progress |
| `not_found` | Subject not listed (will be re-checked next cycle) |
| `found` | Listing confirmed; action needed |
| `indirect_exposure` | Subject's PII (email/phone/name) appears on a **third party's** record (e.g. named in a relative's "Family" field). Not removable via self-service opt-out; needs a targeted CCPA/GDPR delete-my-PII request |
| `action_selected` | Tier/method chosen |
| `submitted` | Opt-out submitted |
| `verification_pending` | Awaiting email/callback verification |
| `awaiting_processing` | Submitted, no verification needed; broker processing |
| `confirmed_removed` | Verified gone |
| `reappeared` | Was removed, now listed again |
| `human_task_queued` | Needs an operator step (captcha/ID/phone/fax/mail) |
| `blocked` | Broker dead / mechanics broken -> flag for DB re-verification |

## Allowed transitions

```
new                  -> searching | found | not_found | indirect_exposure | blocked
searching            -> not_found | found | indirect_exposure | blocked
not_found            -> searching | found | indirect_exposure | blocked
found                -> action_selected | submitted | human_task_queued | indirect_exposure | blocked
indirect_exposure    -> submitted | human_task_queued | not_found | found | blocked
action_selected      -> submitted | human_task_queued | blocked
submitted            -> verification_pending | awaiting_processing | human_task_queued | blocked
verification_pending -> confirmed_removed | human_task_queued | blocked
awaiting_processing  -> confirmed_removed | human_task_queued | blocked
confirmed_removed    -> reappeared | confirmed_removed   (recheck refreshes the date)
reappeared           -> found | indirect_exposure
human_task_queued    -> found | indirect_exposure | action_selected | submitted | verification_pending
                        | awaiting_processing | confirmed_removed | blocked
blocked              -> searching | found | not_found | indirect_exposure | action_selected
```

A transition to the same state is always allowed (idempotent field updates).
