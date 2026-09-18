---
name: profile-readiness
description: "Use before dispatching work to verify worker profiles."
version: 1.0.0
author: Ian Koratsky (@iankoratskydcn), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [profiles, orchestration, credentials, dispatch, readiness]
    category: autonomous-ai-agents
capability_class: EXTERNAL_WRITE
mixed_capability: true
---

# Profile Readiness Skill

Use this before dispatching a durable worker or assigning a Kanban card to a named
profile. It prevents workers from starting with invalid configuration or missing
provider credentials. It does not copy secrets between profiles and cannot complete
an interactive OAuth/device login for the owner.

## When to Use

Run for every non-default worker profile before a new wave, after an update, and
whenever a worker exits immediately with a config or authentication error.

## Prerequisites

The orchestrator needs `terminal` and the target profile name. Use the target
profile's own commands; do not inspect or print token files.

## How to Run

For profile `<profile>` and provider `<provider>`:

```text
hermes -p <profile> config check
hermes -p <profile> auth status <provider>
hermes -p <profile> config get timezone
```

Run these through `terminal` and inspect exit status plus redacted output. `auth
status` is sufficient; never read `auth.json`, `.env`, or token values.

## Quick Reference

| Finding | Action |
|---|---|
| Invalid timezone, especially `US/Eastern` | Set `timezone` to the canonical IANA name, normally `America/New_York`, with `hermes -p <profile> config set timezone America/New_York`. |
| Provider logged out / no credentials | Do not dispatch. Ask the owner to run `hermes -p <profile> auth add <provider>` interactively and finish the device/browser flow. |
| Device flow timed out | Start a fresh flow; never reuse an expired code. |
| Profile has valid config and provider is logged in | Dispatch. |

## Procedure

1. Identify the exact target profile from the card's assignee. Do not substitute
the root profile.
2. Run `config check`. Repair only non-secret configuration with `config set`.
3. Validate timezone with `config get timezone`. Prefer `America/New_York` over the
legacy alias `US/Eastern`.
4. Run `auth status <provider>`. If logged out, block the card with the exact
profile-scoped command needed for owner authentication. A device-code prompt is
interactive and may require the owner; an orchestrator must not guess, paste, or
request a one-time code in chat.
5. Re-run both checks after the owner reports completion. Read back the exact
profile-scoped status before dispatching.
6. If the profile remains unready, preserve the blocker and route the card to
human input instead of retrying the worker until the failure limit is reached.

## Pitfalls

- Named profiles are isolated by design. Never copy the root `auth.json`, refresh
tokens, `.env`, or credentials into a worker profile.
- `hermes auth status openai-codex` without `-p <profile>` checks the wrong profile.
- A successful `auth add` process is not proof of authentication; verify with
`auth status` after the device flow completes.
- Do not dispatch first and rely on the worker to discover setup problems.

## Verification

A profile is ready only when `config check` succeeds, the timezone is canonical,
and the target provider's `auth status` reports logged in. Record the commands and
status in the Kanban task thread; never record secrets or one-time codes.
