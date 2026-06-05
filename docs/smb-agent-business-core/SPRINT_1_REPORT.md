# Sprint 1 Report — Commercial/Sales Core

## Date

2026-05-31

## Scope

Implemented the repo-native Sales Core foundation for SMB agents. This is an Agent Core SQL module plus Hermes tools; no sandbox deploy and no external vendor backend.

## Delivered

- `db/modules/sales/000001_sales_schema.sql`
  - `sales.products`
  - `sales.inventory_balances`
  - `sales.inventory_movements`
  - `sales.quotes` / `sales.quote_items`
  - `sales.orders` / `sales.order_items`
  - `sales.invoices` / `sales.invoice_items`
  - `sales.payment_requests`

- `tools/sales_tool.py`
  - `sales_status`
  - `sales_product_upsert`
  - `sales_inventory_adjust`
  - `sales_quote_create`
  - `sales_order_create`
  - `sales_invoice_create`
  - `sales_payment_request_create`

- `toolsets.py`
  - Added `sales` toolset.

- Runtime support
  - Added `SALES_DB_RUNTIME_USER` default in Agent Core SQL runtime helper.
  - Added `sales_runtime` role to Agent Core role migration and role bootstrap script.
  - Added Sales DB URL derivation to `zeus-sync-secrets.sh`.

## Behavior

- Quote totals are deterministic and support line discounts and tax.
- Orders can be created from quotes and preserve item totals.
- Invoices can be created from orders and preserve item totals.
- Inventory adjustments record movements and update balances.
- Payment requests are stored locally; if no payment adapter is configured, the tool returns a graceful `unavailable` adapter status instead of failing.

## Verification

TDD red checks observed:

- `tests/tools/test_sales_tool.py` initially failed because `tools.sales_tool` did not exist.
- `test_invoice_from_order_preserves_items` initially failed because invoice creation did not return copied invoice items.

Commands run:

```bash
python -m pytest tests/tools/test_sales_tool.py -q -o addopts=
python -m pytest tests/tools/test_sales_tool.py tests/tools/test_crm_tool.py -q -o addopts=
python -m compileall -q tools hermes_cli scripts
ruff check tools/sales_tool.py tests/tools/test_sales_tool.py hermes_cli/agent_core_sql.py scripts/agent_core_roles.py
```

Results:

- Sales tests: 7 passed.
- Sales + CRM tests: 12 passed.
- Compile/ruff: passed.

Live Agent Core DB smoke:

- Created product.
- Adjusted inventory.
- Created quote.
- Converted quote to order.
- Created invoice from order.
- Created payment request with adapter status `unavailable`.
- Verified quote/order/invoice item totals: `208.8`.

## Deferred

- Payment adapter implementation: IzyPagos/Flexipos/Stripe/etc.
- Fiscal/e-invoicing adapter.
- Document rendering/PDF templates.
- Inventory reservation/fulfillment workflows.
- Public UI or sandbox preview.
