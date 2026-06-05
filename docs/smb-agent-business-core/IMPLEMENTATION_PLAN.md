# SMB Agent Business Core Implementation Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task after the planning gates are accepted.

## Goal

Build local Agent Core SQL modules and Hermes tools for the 80% SMB operating layer: commercial/sales, marketing, and accounting-lite.

## Architecture

Module-owned SQL migrations in Agent Core DB, thin JSON-returning tool handlers, deterministic tests, and optional adapters for external systems. No sandbox deploy is required for the core modules.

## Tech Stack

- Python tool handlers under `tools/`
- Agent Core SQL migrations under `db/modules/`
- Hermes tool registry + toolsets
- pytest tests under `tests/tools/`
- Infisical/runtime env for adapter secrets

## Increment 0 — Planning artifacts

Files:
- `docs/smb-agent-business-core/*`

Verification:
- All docs present.
- Factory gates recorded.

## Increment 1 — Sales schema

Tasks:
1. Create `db/modules/sales/000001_sales_schema.sql`.
2. Add products/catalog extensions if not reusing CRM products.
3. Add inventory tables or reference inventory module if split.
4. Add orders/order_items.
5. Add invoices/invoice_items/payment_requests.
6. Run migrations in test DB.

Tests:
- SQL migration applies cleanly.
- Foreign keys and indexes exist.

## Increment 2 — Sales tools

Tasks:
1. Create `tools/sales_tool.py`.
2. Register sales_product_upsert, sales_quote_create, sales_order_create, sales_invoice_create, sales_payment_request_create.
3. Add to `toolsets.py`.
4. Write tests for totals, state transitions, missing adapter behavior.

## Increment 3 — Accounting Lite schema/tools

Tasks:
1. Create `db/modules/accounting_lite/000001_accounting_lite_schema.sql`.
2. Create `tools/accounting_lite_tool.py`.
3. Add expense/income/account/report tools.
4. Tests for monthly totals and export shape.

## Increment 4 — Marketing schema/tools

Tasks:
1. Create `db/modules/marketing/000001_marketing_schema.sql`.
2. Create `tools/marketing_tool.py`.
3. Add brand profile, campaign, content draft, publishing queue tools.
4. Integrate metadata contracts for image/video generation skills.
5. Tests for approval gating.

## Increment 5 — Adapter contracts

Tasks:
1. Define payment adapter contract.
2. Define publishing adapter contract.
3. Define ERP/fiscal adapter contract.
4. Define email marketing adapter contract.
5. Document env vars and smoke tests.

## Increment 6 — Inheritance package

Tasks:
1. Create/patch skills for sales, marketing, accounting-lite.
2. Add profile template guidance for SMB agents.
3. Add demo conversation scripts.
4. Run end-to-end smoke flows.

## Definition of done

- Migrations apply cleanly.
- Toolsets export new tools.
- Unit tests pass.
- Smoke flows pass without external adapters.
- Adapter-missing behavior is graceful.
- Docs and skills explain boundaries.
- PR open with CI green.
