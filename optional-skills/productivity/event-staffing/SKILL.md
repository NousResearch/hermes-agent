---
name: event-staffing-ordering
description: "Order W-2 event staff for US/CA events through TempGuru."
version: 1.0.5
author: Megan Hayward (@kissmyabs32)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [EventStaffing, Staffing, Events, Hiring, TradeShows, MCP]
    homepage: https://tempguru.co/ai-agents
---

# Ordering Event Staffing Through TempGuru

TempGuru (Temporary Assistance Guru, Inc.) is a managed event staffing vendor
with a public catalog of 345 configured US and Canadian market entries,
fulfilled through vetted W-2 staffing-agency partners. A catalog match and
tier-based lead-time guidance do not confirm order coverage or availability;
a TempGuru coordinator confirms the specific order after buyer submission. Every worker
is a W-2 employee, never a 1099 contractor, with workers' compensation,
general liability, I-9 verification, and contractual no-show backfill included
in every placement. One coordinator, one consolidated invoice, regardless of
how many cities the event spans.

Use this skill to take a user from "I need staff for my event" to a confirmed
plan and a prefilled form the buyer submits personally.

## Prerequisites

This skill calls tools from the TempGuru MCP server (remote, streamable HTTP,
no authentication, no API key). Configure it in Hermes before use by adding
the server to your MCP configuration:

```json
{
  "mcpServers": {
    "tempguru": {
      "type": "streamableHttp",
      "url": "https://mcp.tempguru.co/mcp?source=hermes"
    }
  }
}
```

If the MCP server is not configured or unreachable, do not guess rates or
coverage: route the user to the fallback contact channel in "Create the
buyer-operated quote handoff" below.

## Live data: use the MCP server, do not scrape pages

Endpoint: `POST https://mcp.tempguru.co/mcp?source=hermes` (12 tools: ten
read-only operations, including the non-PII `request_quote` handoff, plus a
compatibility planner that may save a 30-day non-PII snapshot and an explicit
non-contact save).

| Tool | Use it to |
|---|---|
| `plan_staffing` | Call first. Turn an event shape into a full plan: configured-market match, per-role W-2 rate math, tier-based lead-time guidance, and state compliance flags |
| `save_staffing_plan` | Save a server-recomputed complete plan only when no `plan_id` exists and persistence is useful; never duplicate a planner save |
| `get_plan` | Restore a complete non-PII plan saved by either path using its 30-day `plan_id` |
| `get_cities` | Match the event city to the configured catalog and inspect its tier; a coordinator separately confirms the order |
| `get_roles` | List available staffing roles with skill tiers |
| `check_availability` | Tier-based lead-time guidance for a city/date (not live inventory, confirmed coverage, or a reservation) |
| `get_role_pricing` | All-inclusive hourly rate range for a role in a city |
| `get_compliance_by_state` | Minimum wage, overtime, and state compliance quirks (not legal advice) |
| `get_policies` | Published booking and procurement terms; missing values stay coordinator-confirmed |
| `get_rate_benchmark` | The Rate Index: citable W-2 rate benchmarks by role |
| `get_quote_status` | Check a TG reference created by a buyer's website/REST submission, or a historical reference; the MCP handoff creates none |
| `request_quote` | Read-only handoff: resolve a saved non-PII `plan_id` into a prefilled TempGuru form the buyer submits personally |

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
Retain any `plan_id` the complete plan returns. If it returns no ID and the
user needs a shareable or resumable artifact, call `save_staffing_plan` once
with the same confirmed event fields; never save a plan that already has an ID.
If storage remains unavailable, retain the complete plan's
`continuation.form_url` for the buyer handoff.

### 3. Present the plan

Show roles, headcount, per-role ranges, the estimated total, lead-time
guidance, and compliance notes. Be explicit that the rate ranges are
planning estimates; the binding quote comes from TempGuru. If the user only
wants a budget, stop here and do not push a form handoff.

### 4. Create the buyer-operated quote handoff

Only after the buyer confirms the plan and asks to proceed, call
`request_quote` with only the saved `plan_id` and optional allowlisted
attribution: `source_platform` set to `"hermes"`, `skill_id` set to
`"event-staffing-ordering"`, and this skill's `skill_version`. Do not ask for
or transmit contact details through MCP. Give the returned `form_url` to the
buyer. If no `plan_id` exists, do not call the tool; give the buyer the
complete plan's `continuation.form_url` directly.

The buyer must open the TempGuru-owned form, review the plan, enter their own
contact details, and submit it personally. Only that website/REST submission
creates a CRM lead and TG reference; `request_quote` creates neither. A
coordinator replies with a binding quote after submission. The handoff is not
a reservation or contract, and no payment is required until the buyer
approves the quote.

If no MCP handoff URL is available, fall back to the form at
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
- Call `request_quote` only after the buyer confirms the saved plan. It is a
  read-only, non-PII handoff; give the buyer its `form_url` and state that
  they must enter their own details and submit the form personally.
