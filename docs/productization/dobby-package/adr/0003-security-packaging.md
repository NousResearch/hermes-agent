# ADR 0003: Security and Packaging Defaults

Status: Accepted for Stage 3 foundation.

## Context

This package will be operated by customers in their own Discord servers and on
their own hosts. The risky failure modes are secret leakage, accidental reuse
of personal runtime state, unauthenticated webhooks, attachment exfiltration,
repo write access, cost runaway, and accidental production mutation.

## Decision

The package defaults must be fail-closed:

- Use a fresh `HERMES_HOME`; never copy `~/.hermes`, logs, sessions, or secrets.
- Use placeholders in examples, such as `<DISCORD_BOT_TOKEN>`, never fake
  secrets that look usable.
- Preflight fails on placeholders, weak webhook secrets, live-path targets, and
  missing Discord/model prerequisites.
- Diagnostics and logs pass through redaction before display.
- Signed webhooks require HMAC per route; insecure bypass is not part of the
  package quickstart.
- Attachments are metadata-only until explicit review approval.
- Repo helper defaults to read-only inspection and patch proposals only.
- Demo kit uses synthetic fixtures and mock endpoints.
- Rollback stops package services and restores previous package config without
  deleting user data unless explicitly requested.

## Packaging Boundaries

Allowed by default:

- Discord gateway.
- Signed webhook inbox.
- Cron/reminders for Discord delivery.
- Native memory inside package-owned `HERMES_HOME`.
- BYO model endpoint.

Disabled by default:

- Telegram, Slack, WhatsApp, Signal, SMS, email, and other broad integrations.
- Honcho server or Honcho cloud setup.
- Write-capable repo automation.
- Live deploy, push, or remote host mutation.

## Evidence Gates

- Static scan of examples finds only angle-bracket placeholders for secrets.
- Preflight tests cover placeholder, weak secret, and unsafe home failures.
- Webhook tests cover unsigned, bad signature, replay, and oversized payloads.
- Attachment tests prove content is unavailable until approved.
- Repo-helper tests prove write, commit, push, deploy, and destructive commands
  are blocked in default mode.
