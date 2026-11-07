# Implementation Readiness Standard

This standard defines the quality bar for implementation plans and task lists.

## Core Principle

A plan passes if a competent engineer (or fresh AI session) who is unfamiliar
with the codebase can execute the plan and produce the correct changes on the
first attempt, without guessing, asking clarifying questions, or making
independent design choices.

## The Zero-Decision Rule

The plan must leave **zero design or architectural choices to the implementer**.
If the implementer must make design choices (including, but not limited to,
choosing a database schema, designing an API path, selecting a dependency, or
deciding how to handle a new edge case), the plan **fails** this standard. All
such decisions must be pre-determined and documented in the plan.

## Phasing & Checkpoints

The plan must partition the work into logical, incremental phases. Each phase
must leave the codebase in a stable, buildable, and testable state.

## Requirements (Non-Exhaustive Examples)

The plan must detail these areas if applicable to the proposed changes:

*   **Target Scope**: Explicit list of files to be created, modified, or
    deleted.
*   **Actionable Steps**: Precise logic changes, data transformations, and API
    updates for each file.
*   **Test Plan**: Specific test cases (happy path and edge cases) that must be
    implemented.
