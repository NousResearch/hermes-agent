# Three-Minute Demo Script

Reader: seller or solutions engineer demoing V1 safely. Next action: run the
script in a staging Discord channel with synthetic data only.

## Demo Setup

Use:

- A staging Discord server.
- One allowed operator user.
- One allowed channel.
- A low-risk staging model key.
- Synthetic webhook and attachment fixtures.

Do not show real env files, tokens, customer logs, private repositories, or live
incident data.

## 0:00 - Positioning

Say:

```text
This is Dobby/Hermes: a self-hosted Discord operator. You bring your Discord
app and model endpoint; the package runs in your environment with safe defaults.
```

Show: the staging Discord channel and the bot online.

Expected outcome: audience understands this is Discord-first and self-hosted.

## 0:30 - Command Center

Prompt:

```text
/dobby status
```

Expected outcome:

- Gateway, model, memory, scheduler, and webhook status are visible.
- Secrets are redacted.
- Any degraded state has a clear next action.

Say:

```text
The operator starts from status. Nothing here exposes a token or assumes access
to systems we have not configured.
```

## 1:00 - Research Scout

Prompt:

```text
Scout this synthetic vendor note: "ACME Staging API latency rose after build
42. The vendor says a cache warmup is likely." Give risks and next steps.
```

Expected outcome:

- Brief risk summary.
- Evidence vs inference separated.
- No claim of private source access.

## 1:30 - Reminder

Prompt:

```text
Remind me in 10 minutes to review the ACME staging rollback checklist.
```

Expected outcome:

- Reminder is created.
- Delivery target is the staging Discord home channel.
- The reminder can be listed or canceled.

## 2:00 - Attachment Review

Prompt:

```text
I uploaded synthetic-build-log.txt. Show metadata first and wait for approval
before reading content.
```

Expected outcome:

- Dobby shows filename, type, size, and source.
- Dobby asks for explicit approval.
- Demo operator denies or approves a synthetic file only.

## 2:25 - Repo Helper And Webhook Inbox

Prompt:

```text
Inspect this sample diff summary and propose review comments. Do not write
files, commit, push, or deploy.
```

Expected outcome: Dobby proposes comments only.

Then trigger the signed synthetic webhook fixture.

Expected outcome:

- Signed fixture is accepted and summarized.
- Unsigned fixture is rejected.
- Summary posts to the staging Discord channel.

## 2:50 - Close

Say:

```text
The sellable V1 boundary is deliberate: Discord command center, research,
reminders, attachment review, read-only repo help, signed webhooks, and native
memory with consent controls. Live promotion waits for verification and has a
rollback path that preserves data by default.
```

Expected outcome: buyer sees the core operator loop and the safety boundary.
