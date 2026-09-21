# Type B Handoff Schema — Kensei-Intake → Orchestrator

When Kensei-Intake has clarified a request with Sahil and produced a structured brief, the handoff follows this format:

## Mandatory fields

```
Title: {one-line task title}
Description: {2-3 sentences describing the work}
Acceptance criteria:
- {criterion 1}
- {criterion 2}
Lead routing: {primary lead — Octacon/Remii/CeeCee/Quan/Wesker/Gojo/Denji/Light}
MetaTags: {from taxonomy — e.g. P1-High, Artefact-Code}
```

## Optional fields

```
Constraints: {time, budget, platform, or design constraints}
Prior decisions: {any decisions already made by Sahil}
Linked backlog: {backlog item ID if this came from the backlog}
```

## By-reference convention

- Prior conversation: reference via `session_search("<query>")`, NOT full transcript paste
- Intake log: reference as `/home/kensei/.hermes/governance/logboard/intake/intake-{YYYY-MM-DD-HHMM}.md`
- Max references: 2
- Max body: 1000 characters

## Example

```
Title: Add pass completion stats to CoachSense match-day view
Description: Sahil already tracks goals/assists in a notebook. Wants a digital mobile-first view for parents on match day. v1 scope: goals, assists, minutes played, passing accuracy. Coach enters data.
Acceptance criteria:
- Mobile-first UI in CoachSense parent-facing view
- Coach-facing data entry form for match-day stats
- Read-only view for parents associated with the player
Lead routing: Octacon
MetaTags: P2-Medium, Artefact-Code
Prior decisions: Mobile-only (not web). Training stats deferred to v2.
Constraints: Nothing too crazy for v1 — basic CRUD
```

This handoff goes to Orchestrator, who creates a Type C kanban task for Octacon.
