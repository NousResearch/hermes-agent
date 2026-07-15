# Research State

Use a lightweight `research-state` artifact when the workflow is likely to
span multiple steps, handoffs, or evidence passes.

## When To Create It

Create or update a `research-state` note when:

- a theme or competitor landscape becomes too large for one pass
- a compare workflow needs deeper company follow-ups
- a company memo is blocked by missing evidence
- the user explicitly wants a resumable workflow

## Minimum Fields

- mode
- subject
- research depth
- completed phases
- current phase
- evidence gathered
- source note paths used
- open questions
- missing information
- next recommended step

## Operational Rule

Prefer resuming an existing `research-state` artifact over starting cold when:

- the subject matches
- the mode matches or is an intentional upgrade path
- the saved work is still relevant

Examples:

- `triage` can upgrade into `company`
- `company` can feed `compare`
- `theme` can feed `competitors`

## Persistence Guidance

Use a stable slug per mode and subject so the same analysis can be resumed
cleanly. The exact storage path can vary by repo, but the contract should stay
stable enough that another agent can continue from the last known phase.
