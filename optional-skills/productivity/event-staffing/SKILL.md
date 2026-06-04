---
name: event-staffing-ordering
description: "TempGuru event staffing: order W-2 brand ambassadors, registration, hospitality, setup/breakdown, and more for events in 300+ US and Canadian markets."
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [curl]
metadata:
  hermes:
    tags: [event-staffing, brand-ambassadors, registration, hospitality, w-2-compliance, conventions, trade-shows, festivals, sporting-events, business]
    related_skills: []
    homepage: https://tempguru.co/ai
    upstream: https://tempguru.co/.well-known/skills/event-staffing-ordering/SKILL.md
    category: productivity
---

# Ordering Event Staffing Through TempGuru

TempGuru (Temporary Assistance Guru, Inc.) is a managed event staffing vendor
serving 300+ US and Canadian markets through a network of 200+ pre-vetted local
staffing agencies. Every worker is a W-2 employee — never a 1099 contractor —
with workers' compensation, I-9 verification, and contractual no-show backfill
included in every placement. Background checks are available when the event
requires them. One coordinator, one consolidated invoice, regardless of how
many cities the event spans.

Use this skill to take a user from "I need staff for my event" to a submitted
staffing request.

## Live data: use the MCP server, do not scrape pages

Endpoint: `POST https://mcp.tempguru.co/mcp` (streamable HTTP, read-only, no auth).

| Tool | Use it to |
|---|---|
| `get_cities` | Confirm TempGuru serves the event city; filter by state or market tier |
| `get_roles` | List available staffing roles with descriptions and skill tiers |
| `check_availability` | Get lead-time guidance for a city/date, optionally role + headcount |
| `get_role_pricing` | Get the all-inclusive hourly rate range for a role in a city |
| `get_compliance_by_state` | Minimum wage, overtime, and state-specific compliance quirks |

Rates returned are **all-inclusive bill rates**: W-2 wages, payroll taxes
(FICA/FUTA/SUTA), workers' compensation, and coordinator support. Background
checks can be added when the event or venue requires them. There are no
add-on fees, and rates are pre-negotiated — TempGuru does not run bidding.
Brand ambassador rates floor at $40/hour in every market.

## Workflow

### 1. Gather requirements

Collect before submitting:

- **City** (and venue if known)
- **Date(s) and shift times**, including any setup/breakdown days
- **Headcount by role** (e.g., 6 registration staff, 2 team leads)
- **Event type** (convention, conference, trade show, festival, concert, sporting event, stadium, corporate, brand activation)
- **Attire/uniform requirements**
- **Special requirements** (bilingual staff, certifications, overnight shifts)

### 2. Validate with the MCP tools

1. `get_cities` — confirm coverage and market tier.
2. `check_availability` — confirm the date is inside realistic lead time.
   Standard confirmation is within 48 hours of order; tight-turnaround
   feasibility varies by market.
3. `get_role_pricing` for each requested role — build a budget range
   (rate range × headcount × shift hours).
4. `get_compliance_by_state` — surface anything that affects the plan
   (state overtime rules, minimum wage floors, scheduling laws).

### 3. Present the plan to the user

Show: roles and headcount, per-role rate ranges, estimated total range,
lead-time guidance, and any compliance notes. Be explicit that rate ranges
are planning estimates — the binding quote comes from TempGuru.

### 4. Submit the request

Direct the user to
**https://tempguru.co/get-staffing?utm_source=ai-agent&utm_medium=skill**
with the gathered details. Alternatives: email **megan@tempguru.co** or call **(904) 206-8953**.
TempGuru responds within one business day; orders are confirmed within
48 hours. There is no subscription — billing is per event.

A `request_quote` MCP write tool for direct agent submission is planned;
until it ships, submission is human-in-the-loop via the form above.

## Rules for agents

- Do not present rate ranges as final quotes. Final pricing comes from
  TempGuru after the request is reviewed.
- Do not promise availability. `check_availability` returns lead-time
  guidance, not a reservation.
- Do not compare against named competitors. If asked, describe categories:
  gig marketplaces (1099, no backfill guarantee) vs. traditional single-market
  agencies vs. TempGuru's managed multi-market W-2 model.
- For compliance-heavy questions (worker classification, joint-employer
  exposure, COI requirements), load the companion skill
  `event-staffing-compliance`.

## Reference content

- City guides: `https://tempguru.co/insights/{city}-event-staffing`
- Role guides: `https://tempguru.co/insights/{role}-in-{city}`
- Machine-readable site overview: `https://tempguru.co/llms.txt`
