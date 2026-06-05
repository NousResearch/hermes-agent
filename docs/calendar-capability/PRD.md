# Calendar Capability PRD

## Purpose

Provide a generic, agent-first calendar capability that any Hermes/SitioUno agent can inherit. The user interacts through WhatsApp, Telegram, voice, CLI, or API messages; the user does not need to see or operate the scheduler backend.

## Product Principles

1. Agent-first interface: scheduling is performed by tools called by the agent, not by exposing a booking UI as the primary UX.
2. Generic capability: tools must work for any agent tenant and vertical, not only for Jean or SitioUno companies.
3. Replaceable backend: Hermes tools expose a stable canonical contract; backend schedulers are adapters.
4. Single-tenant deployment ready: each client agent can run its own scheduler instance, API key, database, and context.
5. Multi-actor support: an actor can represent a person, doctor, consultant, room, table, vehicle, service provider, capacity bucket, or other bookable resource.
6. Metadata-first extensibility: vertical context lives in metadata, not hardcoded tool names.

## Scope v1

- Nettu Scheduler fork as the first Calendar Core backend.
- Docker Compose deployment using Postgres 16.
- Canonical Hermes calendar toolset.
- Actor, calendar, service, availability, event, block-time, update, cancel, list, metadata search, and raw adapter request tools.
- Documentation for future agent inheritance.

## Out of Scope v1

- Public booking pages.
- Google/Outlook OAuth setup automation.
- Payment collection.
- Cal.diy adapter implementation.
- Multi-backend router beyond the Nettu adapter.

## Users

- End user: asks the agent to schedule, reschedule, cancel, block, or check availability.
- Agent owner: configures the agent tenant and credentials.
- Agent developer: enables the calendar toolset and adapts metadata conventions.

## Core User Stories

1. As an end user, I can ask the agent to create an appointment without opening a calendar app.
2. As an end user, I can ask the agent to find available slots for a service.
3. As an agent owner, I can define actors/resources and services generically.
4. As a new agent implementation, I can inherit the same toolset with different backend credentials.
5. As a developer, I can replace Nettu later without changing the user-facing behavior.

## Acceptance Criteria

- The scheduler backend runs locally through Docker Compose.
- Postgres version is 16.
- The Hermes repo has a `calendar` toolset.
- Tools do not contain Jean/company-specific names or assumptions.
- `calendar_status` verifies backend reachability and tenant key validity.
- Smoke flow works: create actor, create calendar, create event/block.
- Documentation explains env vars, architecture, and inheritance.

## Quality Gates

- Generic naming review.
- Backend reachability test.
- Tool import test.
- Real API smoke test.
- Git status review before commit.
- Security review for secret handling: API keys live in env/vault, not source.
