# ADR-002: Use Nettu Scheduler Fork as Calendar Core v1

## Status

Accepted

## Context

The Calendar Capability needs a self-hosted backend that supports programmatic calendars, events, services, booking slots, metadata, multi-tenancy, and future external calendar sync. The backend should be usable by agents through APIs without forcing a user-facing scheduler UI.

Nettu Scheduler provides a compact Rust/API-first scheduler with users, calendars, events, services, booking slots, free/busy behavior, reminders, metadata, webhooks, and Google/Outlook integration primitives.

## Decision

Use a fork of Nettu Scheduler as the Calendar Core v1.

Fork:

- `SiteOneTech/nettu-scheduler`

Local service path during prototype:

- `/home/jean/services/nettu-scheduler/scheduler`

Deployment baseline:

- Docker Compose.
- Postgres 16 Alpine.
- Scheduler bound locally by default.
- External host port 5055 for scheduler API.
- External host port 55432 for Postgres during local build/development.

## Why not Cal.diy as core v1

Cal.diy is stronger as a booking-page/product UI platform. The SitioUno requirement is agent-first scheduling where the user talks to the agent and the backend is invisible. Cal.diy remains a valid future adapter when a public booking page or Calendly-like workflow is needed.

## Operational Findings

Nettu uses SQLx compile-time query checking. During Docker build, Rust compilation requires `DATABASE_URL` to point at a reachable Postgres database with the expected schema applied. A blank database fails compilation with many `relation does not exist` errors.

For the prototype, the canonical workaround is not to bypass SQLx. The build should use a reachable local Postgres instance with Nettu migrations applied. Runtime migration can be disabled with `MIGRATE_ON_START=false` when the DB has already been prepared.

## Consequences

Positive:

- API-first, agent-friendly backend.
- Fork gives SiteOneTech control over build/deployment patches.
- Small service footprint compared with UI-heavy alternatives.

Risks:

- Upstream is older and uses old Rust dependencies.
- SQLx build-time DB coupling complicates Docker reproducibility.
- Migrations are not fully idempotent; repeated startup migrations can fail unless guarded or disabled.

## Follow-up Work

- Improve fork build reproducibility with an explicit migration/build script or SQLx offline metadata.
- Make migrations idempotent or track migration state safely.
- Add CI build for the fork.
- Add future adapter interface if a second backend is introduced.
