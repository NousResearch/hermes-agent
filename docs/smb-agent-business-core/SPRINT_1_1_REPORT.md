# Sprint 1.1 Report — Commerce Workspace + Notification Adapter

## Date

2026-05-31

## Scope

Follow-up to Sprint 1 after approval. This increment defines the customer-facing commerce surface and adds the first provider-neutral notification adapter.

## Delivered

- Infisical/runtime
  - Restarted `zeus-secrets-sync` after `SALES_DB_RUNTIME_USER`, `SALES_DB_RUNTIME_PASSWORD`, and `SENDGRID_API_KEY` were added in Infisical.
  - Added `SALES_DATABASE_URL` in Infisical following the same Agent Core local Postgres URL pattern as the other modules.
  - Verified presence of `SALES_DATABASE_URL`, `SALES_DATABASE_URL_DOCKER`, and SendGrid variables without printing secret values.
  - Ran `scripts/agent_core_roles.py`; verified `sales_runtime` can connect to the local Agent Core Postgres DB.
  - Restarted `hermes-gateway`; gateway is active.

- Notification Core
  - `tools/notification_tool.py`
  - Generic tools:
    - `notification_status`
    - `notification_email_send`
  - SendGrid is the first email adapter, hidden behind the generic notification contract.

- Sales Core customer workspace
  - `db/modules/sales/000002_customer_workspaces.sql`
  - Tables:
    - `sales.customer_workspaces`
    - `sales.customer_workspace_events`
  - `sales_customer_workspace_create` creates URLs such as `https://zeus.kidu.app/w/<token>` for quote/catalog/invoice review.
  - Optional email sending uses Notification Core, not direct SendGrid coupling.

- Architecture decision
  - `ADR-005-customer-commerce-workspace.md`

## Surface decision

A quote/catalog/invoice should be sent as a link to a customer workspace, not only as a PDF.

Target flow:

```text
quote/catalog/invoice created locally
  → workspace URL generated at zeus.kidu.app
  → link sent via Notification Core email adapter
  → client opens, comments, approves/rejects/signs
  → approval can convert quote → order → invoice
  → invoice workspace gets payment CTA through Stripe adapter later
```

## Verification

Commands run:

```bash
systemctl --user restart zeus-secrets-sync
python3 scripts/agent_core_roles.py
python -m pytest tests/tools/test_notification_tool.py tests/tools/test_sales_tool.py -q -o addopts=
python3 -m compileall -q tools hermes_cli scripts
ruff check tools/notification_tool.py tools/sales_tool.py tests/tools/test_notification_tool.py tests/tools/test_sales_tool.py
```

Live DB smoke:

- Applied `db/modules/sales/000002_customer_workspaces.sql`.
- Created a customer workspace row for a smoke quote.
- Verified URL starts with `https://zeus.kidu.app/w/`.
- Verified Notification Core status handler returns OK.

## Deferred

- Actual `zeus.kidu.app` frontend/API implementation.
- Stripe payment adapter and hosted payment button.
- Signature capture implementation.
- Webhook/event receiver for customer actions.
