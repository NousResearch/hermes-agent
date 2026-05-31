# Calendar Capability Inheritance Guide

## Summary

The Calendar Capability is a generic Hermes toolset for conversational scheduling. New agents inherit it by enabling the `calendar` toolset and providing tenant-specific backend credentials.

The user should not need to know whether the backend is Nettu, Cal.diy, Google Calendar, Outlook, or another scheduler. The agent uses canonical tools and maps the owner's intent to metadata, actors, services, calendars, events, and availability checks.

## Canonical Functional Model

Every owner/user starts with one personal calendar. That calendar is created as the default schedule for the user and is enough for immediate personal appointments, reminders, and busy-time blocking.

Additional configuration is owner-driven. The user can later ask the agent to add calendars, services, availability windows, buffers, company-specific agendas, staff/resources, restaurant/medical/service reservations, or external calendar sync according to their own needs. Do not pre-bake company-specific calendars into the reusable capability unless the owner explicitly requests that configuration.

Recommended bootstrap flow:

1. Create or find the user's personal actor/resource.
2. Create or find that actor's personal calendar.
3. Use the personal calendar as the default target for scheduling.
4. Add business/service-specific structures only when requested by the owner/user.

## Runtime Environment

Required environment variables for the Nettu adapter:

```bash
HERMES_CALENDAR_BASE_URL=http://127.0.0.1:5055/api/v1
HERMES_CALENDAR_API_KEY=<tenant-account-api-key>
```

Alternative accepted key for compatibility:

```bash
NETTU_ACCOUNT_API_KEY=<tenant-account-api-key>
```

Secrets must be injected from the proper runtime/vault mechanism. Do not commit real API keys.

## Enable Toolset

Enable the `calendar` toolset for the agent profile/platform. Tool changes require a fresh session or gateway restart depending on how the agent is running.

The toolset exposes:

- `calendar_status`
- `calendar_create_actor`
- `calendar_find_actor_by_metadata`
- `calendar_create_calendar`
- `calendar_find_calendar_by_metadata`
- `calendar_create_service`
- `calendar_find_service_by_metadata`
- `calendar_add_actor_to_service`
- `calendar_add_busy_calendar`
- `calendar_find_availability`
- `calendar_create_event`
- `calendar_block_time`
- `calendar_list_events`
- `calendar_update_event`
- `calendar_cancel_event`
- `calendar_raw_request`

## Generic Vocabulary

Use generic terms in reusable tools and docs:

- Actor: bookable owner/resource. Examples: person, doctor, consultant, room, table, vehicle, capacity bucket.
- Calendar: schedule container for one actor.
- Service: bookable offering or reservation type.
- Event: appointment, reservation, meeting, or scheduled block.
- Block: busy time that prevents booking.
- Metadata: tenant/vertical-specific facts.

Do not hardcode company names, Jean-specific workflows, or vertical-specific terms into tools.

## Suggested Metadata Keys

Use metadata to adapt the capability per tenant:

```json
{
  "tenant_id": "example-tenant",
  "external_ref": "crm-or-erp-id",
  "source_channel": "whatsapp",
  "service_type": "consultation",
  "location_id": "main-office",
  "labels": ["vip", "first-visit"]
}
```

## Smoke Test

After backend deployment and env injection, verify:

1. `calendar_status` returns HTTP 200.
2. `calendar_create_actor` creates an actor with test metadata.
3. `calendar_create_calendar` creates a calendar for that actor.
4. `calendar_block_time` creates a busy event.

## Backend Deployment Notes

Prototype fork:

- `SiteOneTech/nettu-scheduler`

Prototype compose file in fork:

- `scheduler/docker-compose.sitiouno.yml`

Important local defaults:

- Agent Core Postgres server: `agent-postgres` on Docker network `agent-core`
- Agent Core host port: `127.0.0.1:55430`
- Primary agent database: `zeus_agent`
- Calendar backend database: `nettu_calendar`
- Scheduler API: `127.0.0.1:5055`
- Postgres image: `postgres:16-alpine`

Nettu build note: SQLx compile-time checks require a database with migrations applied. The Hermes Agent Core DB migrator prepares `nettu_calendar` before building/running Nettu. Runtime uses `MIGRATE_ON_START=false` because the DB was already prepared and upstream Nettu migrations are not fully idempotent.

## Future Adapters

Future adapters should keep the canonical tool names unchanged. Add adapter-specific mapping behind the tool layer rather than exposing backend details to prompts.

Candidate future adapter:

- Cal.diy for public booking pages and Calendly-like workflows.
