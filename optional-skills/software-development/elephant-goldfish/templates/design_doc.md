# [Feature/Project Name]

*   **Author:** [Author Name or GitHub Handle]
*   **Status:** [Draft | Proposed | In Review | Approved]
*   **Date:** [YYYY-MM-DD]

--------------------------------------------------------------------------------

## Overview

<!-- Provide a high-level summary of the feature, the business or technical
problem it solves, and its context within the system. Keep it concise so both
humans and agents can quickly parse the main objective.
-->

## The Problem

[Summarize the problem and the proposed solution here.]

## Goals

<!--
What will this feature achieve? Be specific, measurable, and clear.
These goals represent the absolute criteria for success.
-->

*   Goal 1
*   Goal 2

## Non-Goals

<!--
Explicitly state what is out of scope. This is crucial to prevent feature creep
and to define strict boundaries for the implementation agent.
-->

*   Non-Goal 1
*   Non-Goal 2

## Open Questions

<!--
List unresolved design questions, technical unknowns, or dependencies requiring
investigation. Unanswered questions are retained here across drafting sessions
to preserve conversational memory and prevent future sessions from re-asking
questions that were already discussed. Remove answered questions as they are
resolved.
-->

*   **Question 1**: [Description of technical unknown or open question]
*   **Question 2**: [Description of technical unknown or open question]

## Critical User Journeys (CUJs)

<!--
Define the end-to-end user journeys this design enables. Each CUJ must contain:

1. Given: Preconditions, starting state, and environment configurations.
2. When: The trigger, user action, or system event.
3. Then: The expected outcome, verification criteria, and postconditions.
-->

### CUJ 1: [Journey Title]

*   **Given**: [Preconditions]
*   **When**: [Actions]
*   **Then**: [Expected Outcomes]

### CUJ 2: [Journey Title]

*   **Given**: [Preconditions]
*   **When**: [Actions]
*   **Then**: [Expected Outcomes]

## Technical Plan (Architecture)

<!-- Detail the high-level design, architectural choices, system data flows, and
component interactions. Include ASCII or Mermaid diagrams to clearly model the
relationships.
-->

[Explain your architectural choices, data structures, and system component
mapping here.]

```text
[Optional ASCII Architecture Diagram]
```

## Alternatives Considered

<!-- Document other architectural approaches or designs that were evaluated and
rejected. Explain the pros, cons, and the technical reasoning for why the chosen
plan is superior.
-->

### Alternative A: [Alternative Title]

*   **Pros**: [List benefits]
*   **Cons**: [List drawbacks]
*   **Reason for Rejection**: [Why this option was not selected]

--------------------------------------------------------------------------------

## Non-Functional Requirements

<!-- Analyze non-functional concerns only if they represent critical risks for
this design:

- Security & Privacy: (Data sensitivity, auth/auth, secrets management).
- Performance & Latency: (Execution runtime, storage footprint, LLM token
  overhead).
- Rollout & Rollback: (Feature flags, backward compatibility, rollback
  commands).
-->

[Detail any critical non-functional requirements here. If none are critical,
this section can be omitted or stated as "None".]

--------------------------------------------------------------------------------

### Additional Resources

<!-- Specify the external resources and context that were used during the design
discussion. Optional. If none were provided omit this section.-->

*   Important project docs referenced
*   Important docs/ referenced
*   Related work

### Future Improvements

<!-- Specify areas that have room for future improvement here. If none were
provided omit this section.-->

*   **The idea**: What could be improved?

--------------------------------------------------------------------------------

## Detailed Implementation Plan

<!-- Specify exactly which files must be created or modified. Do not use vague
language or TBDs.
-->

### Target Files to Create/Modify

#### 1. `[File Path 1]` ([Create | Modify])

*   **Type / Scope**: [e.g., Python Service, C++ Library, TypeScript Module, Build
    Configuration]
*   **Proposed Changes**: [Describe the new APIs, class/function signatures, or
    behavioral logic here. Focus on the minimal required details for
    implementation.]

#### 2. `[File Path 2]` ([Create | Modify])

*   **Type / Scope**: [e.g., Unit Test, E2E Test Driver]
*   **Proposed Changes**: [Describe the test coverage, expected test cases, or
    mock assertions.]

### Verification Plan

<!-- Specify what unit, integration, or manual tests must pass to validate the
implementation. What metrics should be measured to verify success?
-->

*   **Compilation/Build**: [e.g., `npm run build`, `cargo build`, `make`]
*   **Unit Tests**: [e.g., `pytest`, `npm test`, `cargo test`]
*   **Static Analysis & Linting**: [e.g., `eslint`, `ruff check`, `golangci-lint`]
*   **Test Coverage Requirement**: [e.g., 85%+ branch coverage]
*   **E2E/Integration**: [e.g., Integration test command or automated eval runner]

--------------------------------------------------------------------------------

<!-- Provenance: Indicates initial structural drafting via EGM. -->

*Drafted with EGM*
