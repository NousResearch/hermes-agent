# Repo Organization Proposal — SMB Agent Business Core

## Goal

Organize internal modules so future client agents can inherit capabilities from the central repo without pulling in unnecessary vendor backends.

## Proposed structure

```text
db/modules/
  crm/                 # existing
  sales/               # products, catalog, quotes, orders, invoice/payment request
  inventory/           # stock items and movements if split from sales
  marketing/           # brand, campaigns, content calendar, publishing queue
  accounting_lite/     # income, expenses, monthly reports

tools/
  crm_tool.py          # existing
  sales_tool.py
  inventory_tool.py
  marketing_tool.py
  accounting_lite_tool.py
  payment_adapter_tool.py
  publishing_adapter_tool.py

docs/smb-agent-business-core/
  PRD, ADRs, sprint plan, gates

skills/productivity/
  agent-sales-core/
  agent-marketing-core/
  agent-accounting-lite/
```

## Toolset strategy

Expose modules as focused toolsets:

- `crm`
- `sales`
- `inventory`
- `marketing`
- `accounting`
- `payments`
- `publishing`

Default SMB agent profile can enable crm, sales, marketing, accounting, calendar, messaging, file, vision, tts as appropriate.

## Migration strategy

Each module owns its migration ledger under `db/modules/<module>/`. The shared Agent Core DB remains per-agent-instance and single-tenant.

## Adapter strategy

Adapters should be generic and replaceable:

- Payment: IzyPagos, Flexipos, Stripe, PayPal, MercadoPago.
- CRM UI: Twenty.
- ERP/fiscal: Odoo, ERPNext, Alegra, Siigo, QuickBooks/Xero.
- Email marketing: Brevo, Mailchimp, SendGrid, Resend.
- Social: Meta Graph, X, LinkedIn, TikTok, YouTube.
- Video: existing video generation pipeline, own renderer, or third-party platform after evaluation.

## Inheritance target

A new client agent should be bootstrapped with a SOUL/profile using agent-prompt-architect, then configured with the base business-core toolsets and per-client Infisical secrets.
