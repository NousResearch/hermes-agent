# Sprint 1.2 — Commerce Workspace Surface Report

## Scope

This increment implements the first real customer-facing surface for `https://zeus.kidu.app/w/<public-token>` on top of the Sales Core workspace tables introduced in ADR-005.

## Delivered

- Public-token workspace routes mounted before the dashboard SPA catch-all:
  - `GET /w/{public_token}` renders quote/catalog/invoice review page.
  - `POST /w/{public_token}/comment` records customer comments.
  - `POST /w/{public_token}/approve` records approval/signature and, for quotes, converts quote → order → invoice.
  - `POST /w/{public_token}/reject` records rejection.
- New backend module: `hermes_cli/commerce_workspace_surface.py`.
- Quote/invoice/catalog rendering from local Sales Core SQL.
- Lifecycle events persisted to `sales.customer_workspace_events`.
- Workspace status transitions persisted to `sales.customer_workspaces`.
- Quote approval conversion uses existing Sales Core handlers, keeping Agent Core SQL as canonical source of truth.

## Non-goals still preserved

- No customer account portal.
- No owner back-office UI.
- No Stripe implementation yet; payment adapter remains the next increment after this surface contract.
- No PDF-only flow as primary UX.

## QA

- Added `tests/test_commerce_workspace_surface.py`.
- Verified route behavior, rendered document content, comment/reject forms, and quote approval conversion.
