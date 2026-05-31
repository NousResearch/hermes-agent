# Calendar Capability Implementation Plan and QA Report

## Methodology

This project uses the Hybrid Factory lane:

- BMAD-style discipline for PRD, ADRs, acceptance criteria, and QA gates.
- SitioUno Factory discipline for implementation evidence, deterministic verification, and generic reusable capabilities.

## Task Graph

1. Select backend core.
2. Fork backend under SiteOneTech.
3. Implement canonical Hermes tool layer.
4. Register `calendar` toolset.
5. Create local Docker Compose deployment.
6. Resolve backend build/runtime issues canonically.
7. Smoke test backend and tools.
8. Document inheritance for new agents.
9. Commit and push.

## Implemented Artifacts

Hermes repo:

- `tools/calendar_tool.py`
- `toolsets.py`
- `docs/calendar-capability/PRD.md`
- `docs/calendar-capability/ADR-001-agent-first-calendar-adapter.md`
- `docs/calendar-capability/ADR-002-nettu-scheduler-core.md`
- `docs/calendar-capability/INHERITANCE_GUIDE.md`
- `docs/calendar-capability/IMPLEMENTATION_QA.md`

Nettu fork:

- `scheduler/docker-compose.sitiouno.yml`
- `scheduler/.env.sitiouno.example`
- `scheduler/Dockerfile`

## Build Findings

Nettu Scheduler uses SQLx compile-time query validation. `cargo build --release` fails if `DATABASE_URL` points to a blank or unreachable Postgres database.

Observed failures:

- First failure: build could not reach/validate DB schema.
- Second failure: DB reachable but missing migrated tables/domains.
- Runtime failure: startup migration was not idempotent after manual schema preparation.

Applied prototype fixes:

- Pass `DATABASE_URL` as Docker build arg instead of hardcoding upstream local address.
- Use Postgres 16 via Docker Compose.
- Apply Nettu migrations before build for SQLx validation.
- Add `MIGRATE_ON_START` guard to the Dockerfile entrypoint.
- Set `MIGRATE_ON_START=false` in the prototype compose because the DB was already prepared for build/runtime.

## Verification Evidence

Backend:

- `docker compose -f docker-compose.sitiouno.yml build scheduler` completed successfully.
- `docker compose -f docker-compose.sitiouno.yml up -d scheduler` started scheduler successfully.
- Scheduler reachable at `127.0.0.1:5055`.
- Authenticated `GET /api/v1/account` returned HTTP 200.

Tool-layer smoke test:

- `calendar_status` returned HTTP 200.
- `calendar_create_actor` returned HTTP 201.
- `calendar_create_calendar` returned HTTP 201.
- `calendar_block_time` returned HTTP 201.

## Current Limitations

- The Nettu fork still needs a more polished build script or SQLx offline metadata to make Docker builds reproducible from zero without manual migration ordering.
- Startup migrations are not fully idempotent upstream.
- Google/Outlook OAuth flows are not configured in this v1.
- Availability/service flow still needs deeper end-to-end tests.

## QA Status

Status: PASS for v1 prototype capability.

Passed gates:

- Generic naming gate.
- Backend runtime gate.
- Tool import gate.
- Authenticated status gate.
- Actor/calendar/event smoke gate.
- Documentation gate.

Remaining before production hardening:

- Add automated tests for `tools/calendar_tool.py` with mocked HTTP responses.
- Add fork CI build.
- Add an idempotent migration/build runner.
- Add service availability smoke tests.
