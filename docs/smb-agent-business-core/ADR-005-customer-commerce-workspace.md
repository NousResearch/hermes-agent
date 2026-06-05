# ADR-005 — Customer Commerce Workspace Surface

## Status

Accepted for the next commerce increment.

## Context

A quote or invoice sent as a static PDF is not enough for the SitioUno agent product. The client needs a simple place to review the commercial object, ask questions, approve, sign, and pay. The SMB owner should still operate agent-first through WhatsApp/Telegram/voice, while the customer receives a link to a lightweight web surface.

## Decision

Create a customer-facing commerce workspace hosted under the agent domain, starting with:

```text
https://zeus.kidu.app/w/<public-token>
```

The workspace is a thin interaction surface over Agent Core SQL, not a new CRM/ERP source of truth.

### Canonical object flow

```text
Agent creates quote/catalog/invoice locally
  → Sales Core stores document and line items
  → Sales Core creates customer workspace row + public token
  → Notification Core sends the link through an adapter (SendGrid first for email)
  → Customer opens workspace
  → Customer comments / approves / rejects / signs
  → Approval converts quote → order → invoice
  → Payment button is adapter-backed (Stripe first, later IzyPagos/Flexipos/etc.)
  → Agent receives event and continues the conversation with owner/customer
```

### Surface responsibilities

The workspace should support:

- Quote/catalog/invoice rendering from local Sales Core data.
- Customer comments/questions tied to `sales.customer_workspace_events`.
- Explicit approval/rejection actions.
- Signature/acceptance capture when needed.
- Payment CTA after invoice or approved quote.
- Adapter-backed email notifications; do not bind product logic directly to SendGrid.
- Adapter-backed payment links/buttons; do not bind product logic directly to Stripe.

### Non-goals for this increment

- Full customer portal account system.
- Back-office UI for the SMB owner.
- Fiscal/e-invoicing compliance.
- Stripe implementation itself. Stripe comes after the surface contract is stable.

## Data model seed

Sprint 1.1 adds:

- `sales.customer_workspaces`
- `sales.customer_workspace_events`

These rows let Sales Core create stable public URLs and capture lifecycle events without committing to any frontend framework yet.

## Adapter rules

### Notifications

Use `notification_email_send` / Notification Core. SendGrid is the first email adapter because it is now configured in Infisical, but business logic must stay provider-neutral.

Required runtime variables:

- `SENDGRID_API_KEY`
- `SENDGRID_FROM_EMAIL`
- optional: `SENDGRID_FROM_NAME`

### Payments

Use Sales/Payment adapter contracts. Stripe is the first target for payment buttons, but the workspace should only depend on a generic payment-request shape:

- amount
- currency
- invoice/order/quote reference
- customer reference
- success/cancel URLs
- provider metadata

## Consequences

- We can send links instead of PDFs while preserving interaction history.
- Client activity becomes structured Agent Core data, not email-only history.
- Stripe can be added cleanly as a payment adapter after the workspace contract is defined.
- Future customer surfaces can move from sandbox/static to production without changing the Sales Core source of truth.
