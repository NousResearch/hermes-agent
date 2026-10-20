# AgentMail mailbox operations (beyond fetch)

Trigger: purge / clean / delete-all requests against the alert inbox
(`PETASOS_INBOX`). REST base is `PETASOS_API_BASE` (default
`https://api.agentmail.to/v0`), `Authorization: Bearer $AGENTMAIL_API_KEY`,
and the inbox ADDRESS is the inbox id in the path. Source env the same way
the cron wrapper does.

## Purge all emails (threads) — snapshot-first

1. NEVER delete the inbox object itself — banks send alerts to that
   address; "delete all emails" means THREADS. Deleting the mailbox kills
   the pipeline's evidence source.
2. Snapshot BEFORE deleting: list
   `GET /inboxes/{INBOX}/threads?limit=50` (follow the cursor), then
   `GET /inboxes/{INBOX}/threads/{id}` per thread for full bodies; write
   everything to ONE JSON file, re-read it, and assert the on-disk count
   equals the listed count — that snapshot is the only recoverable evidence
   once threads are gone.
3. Delete per thread: `DELETE /inboxes/{INBOX}/threads/{thread_id}`; then
   re-list and require 0 remaining.
4. Health check afterwards: `pipeline.py --dry` must still fetch ("0 new
   transactions") — proves credential + inbox intact. A purge never
   refetches ingested mail: dedupe lives in `state.json` seen_threads plus
   ledger `old_ref`s, both of which survive.
5. In-flight items (an `available > balance` credit) are unaffected — the
   posting alert arrives later as a NEW thread.

Run as a scratch script with the asserts above (owner preference: scratch
script files, never inline). Scratch prunes after ~24h idle — move the
snapshot somewhere durable if the owner wants the evidence kept.
