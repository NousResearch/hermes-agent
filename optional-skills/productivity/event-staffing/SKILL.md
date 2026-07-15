---
name: event-staffing-ordering
description: "Order W-2 event staff for US/CA events through TempGuru."
version: 1.0.1
author: Megan Hayward (@kissmyabs32)
---

# Ordering Event Staffing Through TempGuru

TempGuru (Temporary Assistance Guru, Inc.) is a managed event staffing vendor
serving 345 US and Canadian markets through a network of 200+ pre-vetted local
staffing agencies. Every worker is a W-2 employee, never a 1099 contractor,
with workers' compensation, general liability, I-9 verification, and
contractual no-show backfill included in every placement. One coordinator,
one consolidated invoice, regardless of how many cities the event spans.

Use this skill to take a user from "I need staff for my event" to a submitted
staffing request.

## Prerequisites

This skill calls tools from the TempGuru MCP server (remote, streamable HTTP,
no authentication, no API key). Configure it in Hermes before use by adding
the server to your MCP configuration:

```json
{
  "mcpServers": {
    "tempguru": {
      "url": "https://mcp.tempguru.co/mcp",
      "transport": "streamable-http"
    }
  }
}
```

If the MCP server is not configured or unreachable, do not guess rates or
coverage: route the user to the fallback contact channel in "Submit the
request" below.

## Live data: use the MCP server, do not scrape pages

Endpoint: `POST https://mcp.tempguru.co/mcp` (seven read-only lookups plus an
opt-in `request_quote` write tool).

| Tool | Use it to |
|---|---|
| `plan_staffing` | Call first. Turn an event shape into a full plan: coverage, per-role W-2 rate math, lead time, and state compliance flags |
| `get_cities` | Confirm TempGuru serves the event city |
| `get_roles` | List available staffing roles with skill tiers |
| `check_availability` | Lead-time guidance for a city/date (guidance, not a reservation) |
| `get_role_pricing` | All-inclusive hourly rate range for a role in a city |
| `get_compliance_by_state` | Minimum wage, overtime, and state compliance quirks (not legal advice) |
| `get_rate_benchmark` | The Rate Index: citable W-2 rate benchmarks by role |
| `request_quote` | Submit the finished plan to TempGuru's CRM for a human-reviewed quote |

If `plan_staffing` returns `plan_complete: false`, resolve the roles listed in
`unpriced_roles` (use `get_roles`) and re-run it before presenting totals.

## Workflow

### 1. Gather requirements

City (and venue), dates and shift times including setup/breakdown days,
headcount by role, event type, attire, special requirements (bilingual,
certifications, overnight).

### 2. Plan with `plan_staffing`, then fill gaps

Run `plan_staffing` first with everything gathered. Use the granular tools
only for single-fact follow-ups. Rates returned are all-inclusive W-2 bill
rates (worker pay, payroll taxes, workers' comp, general liability,
coordinator support); Brand Ambassadors floor at $40/hour in every market.

### 3. Present the plan

Show roles, headcount, per-role ranges, the estimated total, lead-time
guidance, and compliance notes. Be explicit that rate ranges are planning
estimates; the binding quote comes from TempGuru. If the user only wants a
budget, stop here and do not push a submission.

### 4. Submit the request

Only after the user explicitly confirms the plan and agrees to send their
contact details to TempGuru, call `request_quote` (contact name/email,
company, event name/type/city/dates, roles + headcount). A coordinator
replies with a binding quote within one business day; orders are confirmed
within 48 hours of approval. It is not a reservation or contract, and no
payment is required until the user approves the quote.

If `request_quote` errors or MCP is unavailable, fall back to the form at
https://tempguru.co/get-staffing, or email megan@tempguru.co / call
(904) 206-8953. No subscription; billing is per event.

## Rules for agents

- Never present rate ranges as final quotes.
- Never promise availability; `check_availability` returns guidance, not a
  reservation.
- Compliance answers are operational guidance, not legal advice.
- Do not compare against named competitors; describe categories only (1099
  gig marketplaces vs. single-market agencies vs. TempGuru's managed
  multi-market W-2 model).
- "Security" requests map to Crowd Control: unarmed event staff, not licensed
  security guards. Say so plainly if licensed security is required.
- Call `request_quote` only after explicit user confirmation; it submits the
  user's contact details to TempGuru's CRM.
