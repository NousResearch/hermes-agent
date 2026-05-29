# Customer Onboarding Checklist

Reader: solutions engineer, customer operator, support owner.
Next action: onboard a customer without copying live secrets, personal runtime
state, or unsupported claims.

## Prerequisites

Before onboarding starts, confirm:

- Customer has an approved operator owner.
- Customer accepts self-hosted V1 boundaries.
- Customer understands Dobby weights are not bundled.
- Customer understands Honcho is not bundled or required.
- Customer has a BYO model endpoint and provider account.
- Customer has a private Discord server or channel for staging.
- Customer has reviewed privacy, retention, export, forget, and delete
  boundaries.
- Delivery artifact has a matching SBOM and attribution pack.

## 1. Customer Discord App

Customer action:

- Create a customer-owned Discord application and bot.
- Restrict bot permissions to the minimum needed command and message surfaces.
- Add the bot only to approved guilds and channels.
- Record allowed operator users or roles.

Package action:

- Use placeholders in setup docs until the customer enters values on their
  host.
- Preflight must reject missing token, missing channel, and unsafe permissions.
- Diagnostics must redact token values and sensitive IDs where appropriate.

Evidence:

- Approved guild, channel, and operator scope recorded.
- Preflight result captured with secrets redacted.

## 2. BYO Model Endpoint

Customer action:

- Provide model endpoint URL and API key through the customer's secret handling
  process.
- Confirm provider terms, retention, billing owner, and allowed data classes.
- Define quota or budget expectations.

Package action:

- Configure Hermes provider settings with placeholders replaced only on the
  customer host.
- Run a redacted health check that does not print the key.
- Handle quota unavailable state without claiming unlimited access.

Evidence:

- Redacted model health result.
- Customer acknowledgment of provider data and billing responsibility.

## 3. Fresh Package Home

Customer action:

- Choose a fresh package `HERMES_HOME`.
- Do not point setup at personal `~/.hermes` or another production runtime
  unless a separate migration plan is approved.

Package action:

- Preflight must fail on unsafe home paths and accidental live paths.
- Create package-owned memory, logs, replay caches, and support directories
  only under the selected home.

Evidence:

- Selected package home recorded.
- Preflight confirms no personal runtime import.

## 4. Allowed Users and Channels

Customer action:

- Name staging and production Discord channels.
- Name allowed operators, roles, and escalation contacts.
- Decide whether webhook summaries may post to Discord.

Package action:

- Configure allowlists before enabling the bot.
- Keep allowed mentions restrictive.
- Reject commands outside approved scope.

Evidence:

- Allowlist reviewed by customer owner.
- Out-of-scope command rejection tested in staging.

## 5. Privacy Policy and Data Boundary

Customer action:

- Publish or approve customer-facing privacy language for Discord, model
  endpoint, webhooks, attachments, memory, logs, support bundles, and optional
  tools.
- Define retention windows for local logs, approved attachments, webhook
  payload summaries, support bundles, and backups.

Package action:

- Provide the local export, forget, and delete boundary.
- Explain that package deletion does not delete Discord history, model provider
  records, customer backups, or optional third-party systems.
- Keep durable memory disabled until operator consent is enabled.

Evidence:

- Privacy and retention approval recorded.
- Memory consent state captured before production use.

## 6. Webhook Setup

Customer action:

- Identify each webhook sender and route.
- Create route-specific HMAC secrets through customer secret handling.
- Decide payload retention and Discord delivery rules.

Package action:

- Enable only signed routes.
- Reject unsigned, stale, replayed, oversized, and wrong-signature payloads.
- Redact payload summaries.

Evidence:

- Negative webhook tests pass in staging.
- Route allowlist reviewed.

## 7. Attachment Review

Customer action:

- Define allowed file types, size limits, and retention expectations.
- Choose who may approve attachment content reads.

Package action:

- Show metadata before content access.
- Require explicit approval before download, extraction, summarization, or model
  submission.
- Redact attachment-derived logs and support output.

Evidence:

- Denied, expired, oversized, and approved attachment paths tested in staging.

## 8. Repo Helper Boundary

Customer action:

- Name allowed repositories and read paths.
- Confirm V1 repo helper is read-only/propose-only.

Package action:

- Block write, commit, push, deploy, branch deletion, destructive git, and live
  remote host mutation by default.
- Produce patch proposals only.

Evidence:

- Read-only inspection works.
- Write and destructive command tests are blocked.

## 9. Staging Test

Run staging before production:

- Discord command help, health, quota/status, and redaction.
- BYO model health with redacted diagnostics.
- Signed webhook negative and positive paths.
- Reminder create, list, cancel, and Discord delivery.
- Attachment metadata-first review.
- Memory status, consent, export, forget, and delete on a staging package home.
- Repo helper read-only/propose-only flow.
- Support bundle redaction.
- SBOM and attribution pack match delivered artifact.

Staging must use customer-approved test data or synthetic fixtures. Do not use
personal runtime state or live production secrets for package verification.

## 10. Production Enablement

Before production:

- Rotate any credentials used during staging if customer policy requires it.
- Confirm production channel/user allowlists.
- Confirm support escalation path.
- Confirm rollback owner and decision threshold.
- Confirm data retention and deletion contacts.
- Capture redacted final preflight and health results.

Go-live is blocked if any critical security, privacy, license, SBOM, or sales
claim boundary is unresolved.

## 11. Rollback

Rollback plan:

- Stop package services.
- Disable the Discord bot or remove it from production channels.
- Disable webhook routes or rotate webhook secrets.
- Revoke or rotate model endpoint credentials if exposure is suspected.
- Restore previous package configuration if applicable.
- Preserve package home for export and investigation unless the customer
  explicitly requests deletion.
- Run support bundle redaction before sharing diagnostics.

Rollback is not deletion. If the customer wants deletion, follow the explicit
delete model for package-owned data.

## Onboarding Blockers

- Customer has no Discord app or approved private channel.
- Customer has no model endpoint or provider terms approval.
- Allowed users/channels are not defined.
- Privacy and retention boundary is not approved.
- Staging tests are skipped or use real production secrets.
- Rollback path is unknown.
- Artifact lacks SBOM, attribution pack, or license review.
