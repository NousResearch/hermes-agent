# Scheduling Workflow

The mechanics of getting finished posts safely onto the calendar — the order of operations,
timing, batching, and how not to make a mess.

## Queue vs schedule vs publish-now

Three modes, three different user intents — confirm which one you're in:

- **Schedule** — post goes out at a specific future datetime. The default and safest. Always
  pin the exact date, time, and **timezone**.
- **Queue** — post is added to a recurring posting slot/cadence rather than a fixed time. Use
  when the user thinks in "add to the queue" terms. Confirm the queue/cadence it lands in.
- **Publish now** — goes out immediately. **Irreversible** — treat it like a one-way door and
  confirm explicitly and separately (see `safety-and-confirmation.md`).

When the user is vague ("post this"), ask which they mean; don't assume publish-now.

## The order of operations

Always in this order — skipping validation or confirmation is how accounts get spammed:

1. **Connect / discover** — Health check; discover projects + accounts (Step 0).
2. **Prepare media** — upload raw bytes; let WoopSocial detect MIME.
3. **Validate** — run every post through WoopSocial's validate before committing. Fix/flag
   failures.
4. **Preview + confirm** — show what/where/when/how-many; get explicit yes.
5. **Create** — schedule/queue/publish.
6. **Verify** — read back IDs; report results; optionally register a webhook.

## Timezones (a common, costly mistake)

A time with no timezone is a bug. Always:

- Confirm the timezone explicitly the first time you schedule in a session.
- Echo scheduled times back **with the timezone** ("Tue Jul 8, 9:00am CEST").
- Resolve relative times ("tomorrow morning," "this evening") to an exact datetime and confirm
  it before scheduling.

## Batching & bulk plans

For multi-post or 30-day plans (often from `batch-content-plan`):

- **Validate the whole set first**, then confirm **once** with a clear summary table (count,
  platforms, date range), not one prompt per post.
- Show the table so the user can spot a wrong date or account before anything is committed.
- If the batch exceeds a plan limit (e.g., X monthly cap), say so up front and propose how to
  handle it (trim, reschedule, or skip that platform) — don't silently drop posts.

## Deduplication — never double-post

- Before scheduling, check whether an equivalent post is already scheduled for the same
  account/time; if so, flag it rather than creating a duplicate.
- On **partial failure** in a batch, track exactly which posts succeeded (by returned ID) and
  re-attempt **only** the failures. Re-running the whole batch is the classic way to double-post.
- Never "retry to be safe" a create call whose result you didn't read.

## Error handling

- Surface the **actual** error message; don't paper over it.
- Distinguish *validation* failures (fix the post) from *delivery/connection* failures (retry
  the specific call or check the connection).
- If an account is disconnected or a token expired, say so and point the user to reconnect in
  WoopSocial — don't keep retrying.
- For confirmation that a scheduled post actually published, prefer a **webhook** over
  re-checking manually.

## Editing & deleting

- The core Posts tools support delete, not in-place update. To change a scheduled post, confirm
  a **delete + recreate**.
- Deletion is irreversible — confirm the specific post(s) explicitly before deleting.
