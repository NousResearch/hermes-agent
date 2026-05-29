# ADR 0001: Product Architecture

Status: Accepted for Stage 3 foundation.

## Context

The sellable package should productize Hermes without turning it into a new
agent runtime. The current repo already has a messaging gateway, Discord
adapter, webhook adapter, cron scheduler, status helpers, redaction utilities,
memory files, and `session_search`. The safe product move is to package and
gate those capabilities, not fork them.

## Decision

Dobby/Hermes will be a thin product profile around Hermes Agent:

- Discord is the primary and default command center.
- The model is bring-your-own via existing Hermes provider configuration.
- Signed webhooks are optional inbound triggers, disabled unless configured.
- Research, reminders, attachment review, repo helper, and memory controls are
  exposed through a Dobby command layer with policy gates.
- Native memory uses `SOUL.md`, built-in memory files, SessionDB, and
  `session_search`.
- Setup, preflight, health, redaction, runbook, rollback, privacy, and demo
  artifacts live in source control and must be runnable without live secrets.

## Data Flow

```text
Discord
  -> Discord adapter
  -> Dobby command router
  -> policy gate
  -> Hermes agent and approved tools
  -> BYO model endpoint
  -> response/redaction
  -> Discord

Webhook sender
  -> HMAC and size validation
  -> route template
  -> Hermes agent
  -> Discord or local log delivery
```

Memory writes are separate from command transport. Durable memory must pass the
memory consent policy before touching `SOUL.md`, memory files, or SessionDB
maintenance flows.

## Consequences

- Later workers should extend existing Hermes surfaces before adding new
  services.
- Package code must not read from an existing personal `~/.hermes` unless the
  operator explicitly points `HERMES_HOME` there.
- Documentation and tests must prove the default path works with Discord,
  webhook, native memory, and BYO model only.
- Optional integrations can remain in Hermes but must stay out of the package
  default config and demo.
