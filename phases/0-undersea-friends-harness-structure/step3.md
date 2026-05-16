# Step 3: standardize-handoff-packets

## Objective
Create a reusable handoff packet standard for Undersea Friends so profiles can transfer work without making the user act as a messenger.

## Scope
- Prefer updating existing docs first:
  - `docs/HARNESS.md`
  - `~/.hermes/shared-memory/team-rules.md`
  - `~/.hermes/shared-memory/undersea-friends/README.md`
- If a template file is necessary, add it under an existing documentation/templates location only after checking existing structure.
- Do not include live Slack IDs, tokens, OAuth values, or secrets.

## Required handoff fields
- Goal
- Why it matters
- Current state
- Target paths/systems
- Role owner and recipient profile
- Risks and safety boundaries
- Execution steps
- Verification criteria
- Report format

## Acceptance Criteria
- The handoff packet standard is documented in one canonical place and referenced from the Undersea Friends operating model.
- The standard explicitly says casual conversation does not need ticketization.
- The standard explicitly says risky live operations require read-only diagnosis first and user approval before change.
- Run JSON validation for phase files and a grep/search check confirming the canonical field names are present.
