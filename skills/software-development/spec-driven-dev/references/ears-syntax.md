# EARS Syntax Cheat Sheet

Easy Approach to Requirements Syntax. Use these patterns for spec.md's
Functional Requirements lines (`FR-001: ...`) instead of loose prose --
this is the part of a requirement that is genuinely machine-checkable and
seedable for property-based test generation (see `ears-schema.json`).

## The Five Patterns

1. **Ubiquitous** -- always true, no trigger.
   `The <system> SHALL <response>.`
   Example: `FR-001: The system SHALL persist user preferences.`

2. **Event-Driven** -- triggered by a discrete event.
   `WHEN <trigger>, the <system> SHALL <response>.`
   Example: `FR-002: WHEN a user submits an empty form, the system SHALL
   display a validation error.`

3. **State-Driven** -- true only while a state holds.
   `WHILE <state>, the <system> SHALL <response>.`
   Example: `FR-003: WHILE a session is authenticated, the system SHALL
   include the user's ID in every audit log entry.`

4. **Optional Feature** -- conditional on a feature being present/enabled.
   `WHERE <feature is included>, the <system> SHALL <response>.`
   Example: `FR-004: WHERE two-factor auth is enabled, the system SHALL
   require a second factor before granting a session token.`

5. **Unwanted Behavior** -- error/exception handling.
   `IF <trigger>, THEN the <system> SHALL <response>.`
   Example: `FR-005: IF a request exceeds the rate limit, THEN the system
   SHALL return HTTP 429 with a Retry-After header.`

Patterns compose: `WHILE <state>, WHEN <trigger>, the <system> SHALL
<response>` is valid EARS.

## Why This Matters Here

Each EARS line is already phrased as trigger + expected response -- a
natural seed for property-based test generation (see Amazon Kiro's
approach: extract properties from EARS lines, generate randomized test
cases from them). This is a *complement* to hand-written tests, not a
replacement: hand-written tests catch what you know to check, generated
ones catch what you didn't think to write down.

## Sensibility Review

Before an EARS line becomes implementation scope, run it through the
`ears-sensibility-gate` skill (if present) or an equivalent check:
proportionality (trigger frequency/criticality vs. response cost),
actor/responsibility match, complete preconditions, testable trigger and
outcome, no conflict with already-approved requirements, and that it
states a goal rather than an implementation detail. A syntactically valid
EARS line can still be semantically nonsensical (e.g. "email the CEO on
every failed login") -- syntax validity is necessary, not sufficient.

## From Prose to Structured JSON

Once an EARS line is agreed, it becomes a schema-validated JSON record
(see `ears-schema.json`) for reliable cross-referencing by ID and for
generating property-based tests -- not re-read as prose each time. The
markdown line stays as the human-readable narrative; the JSON record is
the machine-checkable source of truth.
