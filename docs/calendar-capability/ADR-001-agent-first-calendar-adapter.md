# ADR-001: Agent-first Calendar Capability with Replaceable Backend Adapter

## Status

Accepted

## Context

SitioUno agents need to schedule appointments, service reservations, internal blocks, and availability checks from conversational channels. The primary UX is the agent conversation, not a scheduler UI.

A backend scheduler is still necessary for persistence, conflict checks, services, availability, reminders, and future external calendar sync. If agents call backend-specific APIs directly, every future backend replacement would leak into agent behavior and prompts.

## Decision

Implement a canonical Calendar Tool Layer inside Hermes. The agent calls generic calendar tools. Those tools call a backend adapter. The first adapter targets Nettu Scheduler.

Functional v1 creates a personal calendar for each owner/user first. That personal calendar is the default scheduling surface. After that, each owner/user configures additional calendars, services, availability rules, business contexts, locations, resources, and reservation workflows according to their own use cases and companies. The platform should not force a fixed SitioUno/Jean-specific calendar model into reusable tools.

Layering:

1. Conversation channel: WhatsApp, Telegram, voice, CLI, API.
2. Agent reasoning/runtime: Hermes.
3. Canonical Calendar Tool Layer: `tools/calendar_tool.py` and toolset `calendar`.
4. Backend adapter: Nettu REST API via `HERMES_CALENDAR_BASE_URL` and `HERMES_CALENDAR_API_KEY`.
5. Calendar Core: self-hosted Nettu Scheduler + Postgres 16.

## Consequences

Positive:

- New agents inherit one stable toolset.
- Vertical-specific behavior lives in agent context and metadata, not tool names.
- Personal scheduling works immediately, while business-specific scheduling is layered through user-requested configuration.
- Nettu can be replaced by Cal.diy or another backend later.
- Each tenant can have isolated scheduler credentials and storage.

Tradeoffs:

- The tool layer must maintain adapter mapping logic.
- Some backend-specific fields are normalized only partially in v1.
- Backend migrations and compile-time SQLx checks are operational concerns.

## Non-goals

- Exposing Nettu UI as the product UX.
- Hardcoding Jean/SitioUno company calendars.
- Building a full Cal.diy adapter in v1.

## Verification

A smoke test must call the tool layer, not only curl the backend:

- `calendar_status`
- `calendar_create_actor`
- `calendar_create_calendar`
- `calendar_block_time` or `calendar_create_event`
