# Re-Implementation Standard

This standard defines the quality bar for documenting codebase state.

## Core Principle

A document passes if a competent engineer (or fresh AI session) who is
unfamiliar with the codebase can write functionally identical code on the first
attempt using *only* the document, without guessing or making independent design
choices.

## The Zero-Assumption Rule

The document fails if the reader must guess parameter defaults, assume error
handling, choose design patterns, or infer output schemas. Every behaviorally
significant contract must be explicit.

## Scope of Detail

*   **Include**: External contracts (APIs, formats, side-effects), behavioral
    rules (logic, state, errors), and performance constraints.
*   **Exclude**: Syntax (loops, variable names), internal code organization
    (private helpers), and non-observable structures (internal collections).

## Requirements (Non-Exhaustive Examples)

Document these areas if applicable to the codebase scope:

*   **APIs**: Signatures, types, and expected behavior.
*   **Logic**: Control flow, algorithms, and validation.
*   **Data**: Schemas, serialization, and state transitions.
*   **Side Effects**: Database writes, network calls, logging, and metrics.
*   **Dependencies**: System integrations and libraries.
*   **Configuration**: Flags, parameters, and hooks.
*   **Constraints**: Hardcoded limits and assumptions.
*   **Errors**: Edge cases, empty states, and retries.
*   **Tests**: Test helpers and mocks (if documenting tests).
