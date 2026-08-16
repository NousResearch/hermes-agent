# Cron delivery boundary runbook

## Rule

A user-facing destination must receive **only the successful content that the
job was created to publish**. Scheduler headers, delivery/status text,
preflight warnings, provider/configuration/script failures, and watchdog
output belong in local execution records or a dedicated operational target —
never in a coaching or customer-facing topic.

`[SILENT]` is always no delivery. It is not a success message, and it must not
be transformed into a scheduler notification.

## Per-job policy

`deliver` remains the successful-output destination. Configure the independent
`failure_deliver` policy for operational output:

| Value | Failure/status behavior |
| --- | --- |
| `deliver` (default) | Backward-compatible: use the normal `deliver` target. |
| `local` | Do not send chat content; retain execution status, output, and logs locally. |
| `suppress` | Do not send a failure/status notification. |
| `<platform>:<chat_id>[:<topic>]` | Send only failures/status to that explicit dedicated system target. |

Examples:

```bash
# Keep successful Masa content in Valmennus but route operational failures to Järjestelmä.
hermes cron edit <job-id> \
  --deliver telegram:<group-id>:435 \
  --failure-deliver telegram:<group-id>:1

# A private coaching job with no chat failure notification.
hermes cron edit <job-id> --failure-deliver suppress
```

Never use `local` as a replacement for a user-facing job's successful
`deliver`; it only belongs on the operational path.

## Audit checklist

1. List active jobs: `hermes cron list`.
2. Identify every job whose successful `deliver` target is a protected
   user-facing topic.
3. For each protected job, set `failure_deliver` to `local`, `suppress`, or a
   dedicated system target. Do not use `deliver` for protected destinations.
4. Confirm successful output still has its original `deliver` target.
5. Test or inspect a normal success, a `[SILENT]` response, and a representative
   provider/config/script failure. Only the normal success may reach the
   protected topic.
6. Record target IDs only as necessary; redact group IDs and credentials in
   tickets and reports.

## Rollback

To restore legacy behavior for one job after an operational routing incident,
set `--failure-deliver deliver`. This changes only failure/status routing;
successful delivery and the job schedule remain untouched.
