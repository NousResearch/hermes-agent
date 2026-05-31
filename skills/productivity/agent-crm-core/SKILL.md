---
name: agent-crm-core
description: "Operate Zeus/SitioUno Agent CRM Core: companies, contacts, relationships, opportunities, products, quotes, invoices, interactions, follow-ups, and customer timeline."
version: 1.0.0
category: productivity
metadata:
  hermes:
    tags: [crm, agent-core, customer-management, sales, quotes, invoices]
    created_by: agent
---

# Agent CRM Core

Use this skill when the user asks Zeus to remember/manage clients, contacts, business relationships, opportunities, sales follow-up, products, quotes, or invoices.

## Canonical Principle

CRM Core is the source of truth for Zeus-style agents. External systems such as Twenty, Odoo, Lago, or Medusa are adapters. Prefer the generic `crm_*` tools first; use adapter sync tools only when the user needs external UI/integration.

## Standard Workflow

1. Identify the entity:
   - Company/account -> `crm_organization_upsert`
   - Person/contact -> `crm_contact_upsert`
   - Sales/pipeline -> `crm_opportunity_upsert`
   - Product/service -> `crm_product_upsert`
   - Quote -> `crm_quote_create`
   - Invoice -> `crm_invoice_create`

2. Preserve relationship context:
   - Link contacts to organizations via `organization_id`.
   - Use `crm_relationship_upsert` for non-hierarchical relationships: partner, decision_maker, influencer, referral, competitor, vendor, owner, investor, stakeholder.

3. Record every meaningful touchpoint:
   - Use `crm_interaction_record` for calls, WhatsApp, email, meetings, notes, and decisions.
   - Include channel, direction, actor, and concise summary.
   - Add `follow_up_at` + `follow_up_summary` when a next action is implied.

4. For sales:
   - Create/update the opportunity before creating quotes.
   - Use stages consistently: lead, qualified, proposal, negotiation, won, lost.
   - Keep `value_amount`, `currency`, and `expected_close_date` when known.

5. For quotes/invoices:
   - Upsert products first when reusable.
   - Create quote with line items.
   - Create invoice from quote only when user says it should become billable/formal.
   - Treat CRM Core invoices as operational records, not legal tax/e-invoicing unless an ERP/e-invoicing adapter is configured.

6. Before answering “what is going on with customer X?”:
   - Run `crm_search` if the ID is unknown.
   - Run `crm_customer_timeline` with the selected organization/contact/opportunity.
   - Summarize open opportunities, last interactions, pending follow-ups, quotes, invoices, and risks.

## Metadata Conventions

Keep metadata generic and adapter-neutral:

- `business_id`: sitiouno, izypagos, flexipos, qrovia, personal, etc.
- `owner_id`: owner/user when needed.
- `source_channel`: whatsapp, telegram, email, voice, api.
- `external_ref`: stable external key from another system.
- `labels`: list or comma-separated tags.
- `notes`: short supporting context.

## When to Sync to Twenty

Use `sync_twenty=true` or `crm_twenty_sync` only when:

- The user wants a CRM UI/workspace record.
- A human team will review the data in Twenty.
- The external app needs the data for workflow/reporting.

If Twenty is not configured, continue using CRM Core and report only if the user asked about sync.

## Pitfalls

- Do not store CRM data only in memory or chat history.
- Do not create contacts without enough identifying info if the user gave company/email/phone.
- Do not confuse a relationship note with an interaction; use relationships for durable graph edges and interactions for dated events.
- Do not treat local CRM invoices as compliant fiscal invoices until an accounting/e-invoicing adapter is connected.
- Do not write directly into external app databases; use APIs/adapters.
