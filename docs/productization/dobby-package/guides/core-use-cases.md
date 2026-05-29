# Core Use Cases

Reader: buyer, operator, or implementer validating the V1 package promise.
Next action: verify each use case in staging with synthetic data and confirm
the safe boundary before live use.

## 1. Discord Command Center

Purpose: operate Dobby from Discord with help, status, quota, memory, reminder,
attachment, repo, and webhook entry points.

Safe boundary:

- Only allowed users and allowed channels can command the bot.
- Server channels require mention unless explicitly allowlisted.
- Status and diagnostics redact secrets.

Safe prompt:

```text
/dobby status
```

Expected outcome: a concise status report showing gateway, model, memory,
webhook, and scheduler health without token values.

## 2. Research Scout

Purpose: produce sourced research briefs from approved sources or fixtures.

Safe boundary:

- Verification uses fixtures or mockable data paths.
- Live browsing is optional and should disclose source freshness.
- The scout must distinguish evidence from inference.

Safe prompt:

```text
Scout this synthetic vendor note and list risks, open questions, and next steps.
```

Expected outcome: a short brief with source notes, uncertainty, and no claim of
access to private systems.

## 3. Reminders

Purpose: schedule one-time or recurring reminders that deliver to Discord.

Safe boundary:

- Jobs are stored under package-owned `HERMES_HOME`.
- Cron prompts cannot recursively create more cron jobs.
- Delivery target is the configured Discord home channel unless overridden by
  an allowed route.

Safe prompt:

```text
Remind me in 10 minutes to review the staging checklist.
```

Expected outcome: the reminder is listed, can be canceled, and delivers a
redacted message to the staging channel.

## 4. Attachment Review

Purpose: let operators inspect attachment metadata before content is read or
summarized.

Safe boundary:

- Metadata first: filename, type, size, source, and risk notes.
- Content access requires explicit approval.
- Oversized, unsupported, or expired attachments fail closed.

Safe prompt:

```text
Review the metadata for this synthetic incident log before reading it.
```

Expected outcome: Dobby asks for approval before reading content and records
whether the operator approved or denied access.

## 5. Repo Helper

Purpose: inspect repositories and propose changes without mutating them by
default.

Safe boundary:

- Read-only inspection is allowed.
- Patch proposals are allowed.
- Write, commit, push, deploy, destructive git, and network mutation are
  blocked unless a separate approved mode exists.

Safe prompt:

```text
Inspect this sample diff and propose a patch summary. Do not write files.
```

Expected outcome: a proposed patch or review summary, with no local write or
remote action.

## 6. Signed Webhook Inbox

Purpose: receive events from approved systems and route summarized outcomes to
Discord or logs.

Safe boundary:

- HMAC signature required per route.
- Route allowlist, body-size limit, replay protection, and idempotency are on.
- Unsigned and stale payloads are rejected.

Safe prompt:

```text
Use the synthetic build-failed webhook fixture and summarize operator action.
```

Expected outcome: accepted signed fixture, rejected unsigned fixture, and a
redacted Discord summary for the accepted event.

## Sellable V1 Boundary

V1 is complete when these six use cases work in staging, have mocked or
fixture-backed verification, and fail closed when authorization, consent, or
signature checks are missing.
