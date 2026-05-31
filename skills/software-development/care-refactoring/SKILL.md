---
name: care-refactoring
description: Pragmatic maintenance refactoring workflow for codebases. Use when asked to make a codebase easier to understand, safer to change, or more consistent with existing architecture without changing behavior; includes repository discovery, scoped refactor selection, challenge review, minimal implementation, validation, and a "no changes needed" outcome.
---

# Care Refactoring

Perform a pragmatic maintenance refactor. Optimize for future feature work by humans and AI agents. Do not refactor for aesthetics alone. A valid result is: No code changes needed.

## Operating Rules

- Follow local agent instructions and repository conventions first.
- Preserve unrelated or user-owned changes. Check git status --short --branch before editing.
- Preserve public API, database/schema contracts, response formats, error shapes, permissions, security behavior, and business behavior unless the user explicitly asks otherwise.
- Prefer one small high-value improvement over a broad rewrite.
- Do not impose a new architecture unless the current structure is clearly failing.
- Do not make code changes until the challenge checkpoint ends with PROCEED_WITH_SCOPE or NARROW_SCOPE.
- Before changing production code, lock important existing behavior with a focused characterization test when the behavior is not already covered and the repository has a proportional test layer.

## Discovery

Understand the project before proposing changes:

1. Read useful project context: README.md, architecture docs, local agent instructions, package scripts, CI config, and relevant tests.
2. Infer the stack, validation workflow, module boundaries, naming conventions, and existing architecture.
3. Inspect current diffs and avoid touching unrelated modified files.
4. Identify the owning layer for the behavior under review.

When broad code discovery is needed and subagents are available, delegate repository search to a subagent. Ask for a compact evidence map only: path:line, symbol or route name, relevant snippet or signature, and why it matters. Verify critical findings before editing.

## Refactor Candidates

Look for high-impact issues, especially:

- Business logic embedded in transport/UI handlers, controllers, jobs, middleware, ORM/query code, API clients, serializers, validators, or framework-specific code.
- God files, services, or components mixing orchestration, persistence, formatting, permissions, external calls, and state changes.
- Duplicated business rules across entrypoints.
- Hidden dependencies on time, randomness, env, globals, singletons, network, filesystem, or framework context.
- shared, common, or utils areas becoming dumping grounds.
- Important behavior only testable through slow end-to-end paths.
- Code that makes the next feature likely to copy a bad pattern.

## Responsibility Boundaries

Use the existing architecture where possible.

- Entry/transport/UI: accept input, read context, call application logic, map result or error to output.
- Application/use-case: own the user or business scenario, permissions, transactions, orchestration, and coordination.
- Domain/rules: hold pure rules, invariants, decisions, state transitions, calculations, and domain errors.
- Infrastructure/adapters: talk to databases, ORM, external APIs, queues, storage, filesystem, SDKs, and platform services.
- Wiring/composition: assemble concrete dependencies.

Heuristics:

- If code answers "what should happen?", keep it in application/domain.
- If code answers "how do we talk to an external system?", keep it in infrastructure.
- If code answers "how do we receive input and return output?", keep it in entry/transport/UI.

## Scope Proposal

Before editing, propose at most 1-3 small high-value refactoring scopes. One scope may be a file, endpoint, component, use case, job, module boundary, or business flow.

For each proposed scope, state:

- Concrete smell.
- Preserved behavior and public contracts.
- Smallest useful change.
- Main risk.
- Focused validation signal.
- Whether a pre-refactor characterization test is needed, already exists, or would be disproportionate.

## Challenge Checkpoint

If subagents are available, ask a fresh challenge agent to validate the proposed scope before implementation. Use the most capable available model with high or extra-high reasoning. The challenge agent must not write code.

Ask it to answer:

- Is this refactor actually worth doing?
- Is the scope small enough?
- What behavior or public contract could be accidentally broken?
- Is there a simpler improvement?
- Is this overengineering?
- What existing or new characterization test should protect the behavior before refactoring?
- Should we proceed, narrow the scope, do nothing, or turn this into a separate product task?

The main agent and challenge agent may do at most two short rounds of disagreement.

If subagents are not available, perform the same challenge as a separate critical pass yourself and summarize the conclusion.

End with exactly one decision:

- PROCEED_WITH_SCOPE
- NARROW_SCOPE
- NO_CHANGES_NEEDED
- SEPARATE_PRODUCT_TASK

Do not change code unless the final decision is PROCEED_WITH_SCOPE or NARROW_SCOPE.

## Characterization Tests

For refactors that preserve behavior, prefer a green characterization test before editing production code:

- Use the highest-value existing test layer that is proportional to the scope.
- Test observable behavior, contracts, boundaries, and invariants, not incidental implementation details.
- Keep the test narrow enough to fail for an accidental behavior change in the accepted scope.
- Run the characterization test before production edits and confirm it passes against current behavior.
- If no suitable test layer exists or adding a test would be disproportionate, state that explicitly and use the fastest reliable validation path instead.

Do not invent a heavy test harness just to satisfy this step. Do not encode known bugs as desired behavior unless the user explicitly wants the bug preserved.

## Implementation Loop

For each accepted scope:

1. Inspect surrounding code and tests.
2. Identify the concrete smell, preserved behavior, public contracts, risks, and smallest useful change.
3. Add or identify the focused characterization test if important behavior is not already covered, then run it before production edits.
4. Make the minimal connected diff.
5. Move decisions to the owner layer; do not patch symptoms in children, leaf helpers, or low-level adapters when the wrong decision is made higher up.
6. Add or update focused tests only when they protect behavior being moved or clarified.
7. Run the most relevant project checks discovered from scripts, docs, Makefile, or CI.

Use TDD for non-trivial behavior, contract, auth, persistence, routing, query, validation, or state-transition changes when the repository has a proportional test layer. For pure maintenance movement where behavior is preserved and tests already cover the path, keep test changes focused.

## Avoid

- Rewriting from scratch.
- Adding new frameworks or dependencies unless clearly necessary.
- Adding an interface, port, repository, factory, mediator, event bus, or CQRS pattern without current value.
- Splitting simple code into many files just to look clean.
- Creating unrelated cleanup or doc churn.
- Leaving TODOs instead of completing the current refactor.
- Adding architecture that makes the code harder to read.
- Moving business scenarios into infrastructure/query/client code.
- Letting infrastructure/framework details leak into pure business logic.

## Pre-Final Review

Review the diff as a fresh reviewer:

- Confirm behavior is preserved.
- Confirm scope stayed small.
- Confirm public contracts are unchanged.
- Confirm characterization coverage was added, already existed, or was intentionally skipped with a reason.
- Confirm tests/checks pass, or clearly explain what could not be run and why.
- Confirm documentation changes are only made when durable architecture, setup, operations, contracts, user flows, or engineering decisions changed.
If the refactor no longer looks worth it, revert only your own changes and report No useful maintenance refactor found.

## Final Report

Report concisely:

- Overall result: no changes, changes made, or partial validation.
- Scopes inspected.
- Challenge checkpoint decision.
- Scopes changed.
- What improved and why it matters.
- Public contracts preserved.
- Characterization coverage added, reused, or intentionally skipped.
- Primary signal status.
- Secondary signal status: tests/checks run.
- Docs status.
- Remaining risks or suggested future cleanup.
- Suggested commit message when changes are ready.
