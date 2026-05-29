# Incident Response Runbook

Reader: on-call operator handling a Dobby package incident. Next action:
contain the issue, preserve redacted evidence, and recover through rollback or
verified remediation.

## Severity

- Critical: secret exposure, unauthorized Discord response, unsigned webhook
  accepted, repo write/push/deploy, or deletion outside package-owned data.
- High: attachment content read without approval, memory write without consent,
  model-cost runaway, or repeated spam in an allowed channel.
- Medium: degraded health, quota unavailable, reminder delivery failure, or
  stale webhook replay blocked as designed.

## First 15 Minutes

1. Stop new intake: pause gateway and webhook routes.
2. Preserve redacted logs and timestamps.
3. Identify affected surface: Discord, model, memory, attachment, repo helper,
   reminder, or webhook.
4. Rotate any credential that may have been exposed.
5. Run rollback if the bot cannot be proven safe quickly.

## Evidence To Collect

Collect only what is needed:

- Incident start and detection time.
- Redacted status output.
- Discord channel ID and message IDs, not message contents unless approved.
- Webhook route name, event type, and signature validation result.
- Package `HERMES_HOME` path.
- Version or commit identifier.
- Operator actions taken.

Do not copy raw tokens, private attachments, customer records, or personal
runtime directories into the incident folder.

## Common Incidents

Secret leak:

- Stop gateway.
- Rotate Discord token, model key, or webhook secret at the source.
- Replace host config.
- Verify scanners and redacted status before restart.

Discord over-response:

- Set mention-required mode on.
- Remove free-response channels.
- Narrow user and channel allowlists.
- Test denied channel and denied user behavior.

Webhook auth failure:

- Disable the route.
- Rotate route secret.
- Verify unsigned, bad-signature, replayed, and oversized payload rejection.

Attachment read without approval:

- Stop intake.
- Quarantine attachment cache.
- Preserve metadata-only evidence.
- Delete content copies only after privacy owner approval.

Memory consent failure:

- Turn durable consent off.
- Export redacted memory state.
- Forget or delete affected package-owned entries after approval.
- Verify `session_search` no longer returns deleted content.

Repo helper attempted mutation:

- Stop the session.
- Inspect working tree without reverting unrelated user changes.
- Confirm no commit, push, deploy, or destructive git ran.
- Keep repo helper disabled until deny tests pass.

Model-cost runaway:

- Stop gateway.
- Revoke or lower the staging key quota.
- Inspect active sessions and cron jobs.
- Restart only after loop source is identified and bounded.

## Recovery

Recovery requires:

- Root cause or bounded uncertainty.
- Rollback or fix applied in staging first.
- Verification runbook pass.
- Operator signoff for live restart.

## Post-Incident

Record:

- What failed.
- What contained it.
- What verification now proves.
- What data was exported, forgotten, deleted, or intentionally retained.
- Any product docs or tests that need tightening.
