---
name: contact-routing
description: Resolve verified contact routes before authorized messaging.
version: 1.0.0
author: Julian Albou (@julianships), with Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [contacts, messaging, routing, safety]
---

# Contact Routing Skill

Resolve a named person to a profile-scoped, purpose-specific outbound route without sending. This skill keeps identity, reachability, route preference, and messaging authorization separate so one fact is never inferred from another.

## When to Use

- The user asks to contact, notify, DM, email, or hand off to a person by name.
- Several platform identities or destinations could refer to the same person.
- The communication purpose determines which route should be used.
- A discovered or allowlisted destination must not be mistaken for route preference.

## Prerequisites

- Initialize `$HERMES_HOME/contacts.yaml` with `hermes contacts init`.
- Record only source-backed identity and route information.
- Configure the relevant messaging or mail adapter separately; the registry stores no credentials.

## How to Run

Resolve without sending:

```bash
hermes contacts resolve "Person" --purpose purpose_name
```

Use `--route route-key` only when the user or a durable workflow selected that exact route. Add `--show-destination` only when preparing an authorized delivery.

## Quick Reference

```bash
hermes contacts init
hermes contacts validate
hermes contacts list
hermes contacts show "Person"
hermes contacts resolve "Person" --purpose internal
```

The registry lives at `$HERMES_HOME/contacts.yaml`. Keep it owner-readable, source identities and route preferences, mark stale or unverified routes honestly, and avoid storing credentials or message history in it.

## Procedure

1. Identify the communication purpose explicitly, such as `internal`, `external_work`, or `urgent`.
2. Run `hermes contacts resolve` with the person's exact ID, display name, or alias. Matching is case-insensitive and Unicode-normalized but preserves punctuation and word boundaries.
3. Stop on any status other than `ok`, including unknown or ambiguous contacts, missing routes, stale endpoints, stale directory caches, or directory mismatches.
4. Treat a successful resolution as reachability evidence only. It still reports `authorization_check: required`.
5. Confirm the user authorized the communication and that no platform-specific identity check remains.
6. Re-run with `--show-destination` only when preparing the authorized delivery.
7. Send through the existing messaging or mailbox tool; the resolver never sends.
8. Read back the destination, sender identity, message ID, and content when the platform supports it.

## Pitfalls

- Do not automatically merge contacts based on similar names or endpoints.
- Do not remove punctuation or whitespace to force a match.
- Do not promote every discovered destination into the registry.
- Do not treat a directory match or route preference as authorization.
- Ask for clarification when identity, purpose, preferred route, or authority is materially ambiguous.

## Verification

- `hermes contacts validate` reports `status: ok`.
- Resolution reports the expected contact ID, route key, and live-check state.
- A directory match is accepted only when the generated cache is at most ten minutes old.
- Resolution output always reports `send_performed: false` and `authorization_check: required`.
- After an authorized send, the platform readback matches the resolved destination and expected sender identity.
