# CRM Core Hybrid Sprint Plan

## Methodology

Hybrid Factory lane: BMAD-style documentation discipline + SitioUno Factory execution/gates.

## Sprint 0 — Foundation

1. Verify repo/remotes/tooling.
2. Create/verify SiteOneTech Twenty fork.
3. Write PRD and ADRs.
4. Define task graph and QA gates.

## Sprint 1 — CRM Core DB

1. Extend `crm` schema with relationships, products, quotes, quote items, invoices, follow-ups, external links.
2. Keep all module data inside shared Agent Core Postgres.
3. Apply migrations locally.
4. Smoke-test synthetic customer graph.

## Sprint 2 — Hermes Tools

1. Expand `tools/crm_tool.py`.
2. Register tools in `toolsets.py`.
3. Keep adapter optional.
4. Add tests for core helpers and adapter request behavior.

## Sprint 3 — Twenty Adapter

1. Add env-based Twenty client.
2. Add sync for company/person/opportunity.
3. Store `crm.external_links`.
4. Document fork/install/config pattern.

## Sprint 4 — Skills

1. Create `agent-crm-core` skill for using the tools.
2. Create `twenty-crm-adapter` skill for Twenty-specific operations.
3. Include CRM relationship/pipeline habits so Zeus uses the system proactively.

## Sprint 5 — QA/Delivery

1. Run SQL migration.
2. Run unit tests.
3. Run local smoke test with synthetic data.
4. Report evidence and limitations.

## Task Graph

- T0 repo-state -> T1 docs
- T1 docs -> T2 migration
- T2 migration -> T3 tools
- T3 tools -> T4 tests
- T3 tools -> T5 skills
- T4 tests + T5 skills -> T6 delivery

## QA Gates

- Import gate: `python -m py_compile tools/crm_tool.py`
- Unit gate: targeted pytest for CRM tool.
- DB gate: migration applies to `agent-postgres/zeus_agent`.
- Smoke gate: create org/contact/opportunity/product/quote/invoice/interaction/follow-up and query timeline.
- Adapter gate: Twenty raw request returns configured=false when env missing; no hardcoded secrets.
