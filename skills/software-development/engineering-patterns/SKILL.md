---
name: engineering-patterns
description: "Use when designing systems, reviewing code, refactoring legacy code, choosing GoF patterns, defining boundaries, writing tests around risky changes, or evaluating engineering trade-offs. Applies canonical engineering patterns without cargo-culting them."
version: 1.3.0
author: Tyler Mayfield + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [architecture, design-patterns, gof, code-quality, engineering-culture, solid, refactoring, ddd]
    related_skills: [zen-of-python, architecture-doc-generator, explain-code-callout, requesting-code-review, test-driven-development, systematic-debugging, project-understanding, verification-before-completion]
---

# Engineering Patterns

## Overview

Use this skill to apply proven software engineering patterns from canonical engineering literature without turning them into cargo cults. The goal is not to name a pattern first; the goal is to diagnose the engineering pressure, pick the smallest useful abstraction, protect behavior with tests where needed, and explain trade-offs clearly.

This skill synthesizes ideas from Clean Architecture, Domain-Driven Design, GoF Design Patterns, Refactoring, The Pragmatic Programmer, Code Complete, Working Effectively with Legacy Code, Peopleware, The Mythical Man-Month, and SICP.

## When to Use

- Designing a new service, module boundary, domain model, API, or library
- Reviewing code for cohesion, coupling, testability, dependency direction, and pattern fit
- Choosing between GoF patterns without over-engineering
- Refactoring existing behavior safely
- Working with legacy code that lacks tests or hides dependencies
- Planning team structure, delivery scope, ownership, or coordination trade-offs
- Explaining architecture decisions in plain language

## Do Not Use When

- The task is a tiny one-off script or syntax question
- A direct function, small helper, or framework convention is clearly enough
- The user only wants a quick implementation and has not asked for design analysis
- There is no meaningful variation, boundary pressure, duplication, or testability problem
- Applying the skill would add jargon or indirection without improving maintainability

## Operating Procedure


When applying this skill, do not start by naming a pattern. Start by diagnosing the engineering situation.

1. **Classify the task**
   - New design: service, module, domain model, API, library
   - Review: PR, architecture, code smell, testability, coupling
   - Refactor: existing behavior must stay stable
   - Legacy change: code has weak/no tests or hidden dependencies
   - Team/process: scope, delivery, interruptions, ownership, coordination

2. **State the concrete problem in one sentence**
   - Good: “Provider fallback logic is duplicated across three modules and new providers require editing all callers.”
   - Bad: “We need Strategy.”

3. **Pick at most two candidate principles/patterns**
   - Use the decision matrix below.
   - Prefer the smallest pattern that solves the current problem.
   - If several patterns seem plausible, compare their trade-offs before choosing.

4. **Check the simpler alternative**
   - Could a plain function, dictionary dispatch, small value object, or module-level helper solve it?
   - Only introduce indirection when it buys testability, extensibility, clarity, or boundary protection.

5. **Define the boundary and contract**
   - What data crosses the boundary?
   - Who owns the abstraction?
   - What invariants must hold?
   - What dependencies are allowed to point where?

6. **Apply the smallest safe change**
   - New code: build the simplest structure that keeps the boundary clean.
   - Existing code: add characterization tests or seams first.
   - Large migration: use expand-contract, not a big-bang rewrite.

7. **Verify behavior and explain trade-offs**
   - Run relevant tests or define a manual verification path.
   - State why the chosen pattern beats the simpler option.
   - Document known costs: indirection, extra files/classes, performance, onboarding complexity.

---

## Quick Decision Matrix


| Situation | Start With |
|-----------|------------|
| New service/domain to model | DDD bounded contexts + ubiquitous language |
| Existing code, need to change safely | Feathers characterization tests + seams first |
| Code review / smell detection | Fowler code smells + SOLID checks |
| Architectural boundary decision | Clean Architecture dependency rule + Pragmatic orthogonality |
| API/library design | SICP abstraction + GoF Adapter/Facade/Strategy + Law of Demeter |
| Daily construction quality | Pragmatic DRY + Code Complete complexity rules |
| Legacy modernization | Feathers seams + DDD anticorruption layer + expand-contract changes |
| Team/schedule planning | Brooks's Law + Peopleware flow-state protection |
| Pattern choice | Problem intent first, GoF structure second |

---

## Code Review Checklist


Use this checklist during PR review or before claiming a refactor/design is complete.

### Problem and intent

- [ ] The code solves a clearly stated problem, not an imagined future requirement
- [ ] Names match the domain language used by users/stakeholders
- [ ] The design can be explained without pattern jargon
- [ ] The pattern, if any, is named after the problem is understood

### Cohesion and responsibility

- [ ] Each module/class/function has one primary reason to change
- [ ] Data and behavior that belong together are co-located
- [ ] No god object, manager class, or “utility dumping ground” is emerging
- [ ] Long functions/classes are split by behavior, not arbitrary line count

### Coupling and boundaries

- [ ] Dependencies point in the intended direction
- [ ] Domain/application logic does not depend on framework, ORM, UI, or provider details
- [ ] Boundary objects are stable DTOs/events/value objects, not leaked infrastructure models
- [ ] External APIs/legacy schemas are behind adapters or anticorruption layers

### Change safety

- [ ] Existing behavior is protected by tests before refactoring
- [ ] Legacy code changes identify a seam before invasive edits
- [ ] Public API changes use expand-contract where possible
- [ ] Tests assert behavior, not implementation trivia

### Simplicity and trade-offs

- [ ] A simpler alternative was considered
- [ ] New abstractions are justified by real variation, duplication, or boundary pressure
- [ ] The design avoids premature generality
- [ ] Trade-offs are documented when the design is non-obvious

### Common pattern signals

- [ ] Repeated `if provider == ...` or `switch type` logic suggests Strategy/Factory/Registry
- [ ] Many callers performing setup in the same order suggests Builder or Template Method
- [ ] Third-party/legacy interface mismatch suggests Adapter
- [ ] Complex subsystem used in one common way suggests Facade
- [ ] Cross-cutting behavior suggests Decorator or middleware
- [ ] State-dependent conditionals suggest State or an explicit state machine
- [ ] Undo/queue/deferred execution suggests Command

---

## Modern Python and TypeScript Pattern Translations


GoF patterns came from class-heavy OO languages. In Python and TypeScript, the cleanest implementation is often a function, protocol/interface, registry, middleware, or data object rather than a deep inheritance hierarchy.

| Classic Pattern | Modern Python Form | Modern TypeScript Form | Use This Instead Of |
|-----------------|--------------------|-------------------------|---------------------|
| Strategy | Callable, `Protocol`, provider class, dict of functions | Function type, interface, object map | Long provider `if/elif` chains |
| Factory Method | Function returning a protocol/interface implementation | Factory function returning interface | Constructors scattered across callers |
| Abstract Factory | Module/object that creates related implementations | Factory object with typed methods | Passing many concrete classes around |
| Builder | Dataclass/Pydantic builder, fluent object, validated config assembler | Builder object, schema parser, config composer | Huge constructors or partially valid objects |
| Adapter | Thin wrapper around SDK/API/legacy response | Wrapper/client class, mapping layer | Leaking third-party response shapes |
| Facade | Service module with a simple public function | Service/client facade | Callers orchestrating many subsystem details |
| Decorator | Function decorator, context manager, middleware | Higher-order function, middleware | Repeating logging/retry/auth behavior |
| Observer | Callback list, event bus, signal emitter | EventEmitter, observable, pub/sub | Direct calls from producer to every consumer |
| Command | Task object, dataclass command, queue message | Command object, action payload | Passing raw ad-hoc dicts through queues |
| State | Enum + transition table, state object | Discriminated union + transition map | Large stateful conditional blocks |
| Template Method | Pipeline function with hook callables | Pipeline function with typed hooks | Subclassing only to override one step |
| Chain of Responsibility | Middleware list, validator chain | Middleware/handler chain | One giant validation/orchestration function |
| Proxy | Lazy property, caching wrapper, authorization wrapper | Proxy object, wrapper service | Callers manually managing cache/auth/lazy load |
| Value Object (DDD) | Frozen dataclass/Pydantic model | readonly type/interface/class | Repeated primitive tuples/strings |

### Translation rules

- Prefer functions and composition before inheritance.
- Prefer `Protocol`/interface boundaries over concrete base classes when behavior matters more than shared implementation.
- Prefer registries for plugin/provider lookup when new implementations will be added over time.
- Prefer discriminated unions / enums for finite state when transitions are simple; use State objects when behavior per state is substantial.
- Prefer explicit DTOs/value objects at boundaries instead of passing raw provider/framework dictionaries everywhere.

---


## Canonical Pattern Heuristics

Use these as quick triggers; see `references/canonical-engineering-books.md` for the full book-by-book synthesis.

| Pressure | Heuristic |
|----------|-----------|
| Business/domain complexity | Use DDD bounded contexts, ubiquitous language, value objects, and aggregate boundaries only where invariants justify them |
| Framework/provider coupling | Apply Clean Architecture dependency direction; put policy inward and infrastructure at the edge |
| Repeated conditional behavior | Consider Strategy, Registry, State, or polymorphism after checking if a simple dictionary dispatch is enough |
| Unsafe legacy change | Add Feathers-style characterization tests and seams before invasive edits |
| Duplicated knowledge | Apply DRY to duplicated decisions, not every repeated line of similar-looking code |
| Complex construction | Consider Builder or a validated config assembler when objects have ordered steps or invariants |
| External API mismatch | Use Adapter or an anticorruption layer so provider schemas do not leak inward |
| Complex subsystem usage | Use Facade when most callers need one common path through many details |
| Cross-cutting behavior | Use Decorator, middleware, or wrapper services for retry/auth/logging/instrumentation |
| Team coordination pressure | Remember Brooks's Law and Peopleware: protect focus, limit coordination surfaces, and make ownership explicit |

## Pattern Selection Rules

1. Name the concrete problem before naming the pattern.
2. Prefer functions, modules, protocols/interfaces, value objects, and registries over inheritance-heavy structures in Python and TypeScript.
3. Introduce indirection only when it buys real extensibility, testability, boundary protection, or readability.
4. If two patterns both fit, choose the one with fewer moving parts and clearer failure modes.
5. Keep old public seams or re-exports during refactors when callers/tests monkeypatch them.
6. Use expand-contract for public API or schema migrations: add the new path, migrate callers, then remove the old path.
7. Document the cost of the pattern: extra files/classes, cognitive load, performance, and onboarding impact.

## Common Pitfalls


1. **Pattern worship** — applying a pattern before the matching problem exists.
2. **Conflating pattern with implementation** — a class named `Factory` that does 15 unrelated things is not a Factory.
3. **Ignoring composition** — modern code often implements GoF ideas with functions, composition, registries, and middleware rather than inheritance.
4. **Singleton abuse** — most “I need just one” cases are scoped instances or dependency injection, not globals.
5. **Forgetting trade-offs** — every pattern adds indirection and cognitive cost. State why the benefit is worth it.
6. **Visitor on unstable structures** — if element types change often, every visitor changes too.
7. **Premature optimization** — measure before optimizing; most performance problems live in a small part of the code.
8. **Big bang rewrites** — prefer incremental modernization behind seams and boundaries.
9. **Adding people to a late project** — this usually increases coordination overhead.
10. **Second-system over-engineering** — set explicit scope boundaries and success criteria before designing.
11. **Metric gaming** — optimize for outcomes, not proxy outputs.
12. **Feature envy** — move behavior toward the data it operates on, or introduce a proper domain object.
13. **Refactoring without preserving test hooks** — keep old module-level names/re-exports if tests or callers monkeypatch them.
14. **Schema-blind signal extraction** — handle flat and nested input schemas at boundaries; otherwise results fail silently.
15. **Forcing all modules into a pattern at once** — deploy registries/catalogs first, then convert modules one at a time.

---

## Worked Examples and References


This skill absorbed the detailed GoF examples from the old `design-patterns` skill. Reference files were copied into this skill:

- `references/worked-example-reconiq.md` — Strategy + Registry + Template Method applied to ReconIQ
- `references/competitor-search-case-study.md` — Builder + Strategy + Template Method for competitor query generation
- `references/case-study-3-fallback-chain.md` — Chain of Responsibility + Strategy + Factory for provider fallback chains
- `references/gof-pattern-catalog.md` — detailed GoF pattern catalog extracted from the original skill
- `references/canonical-engineering-books.md` — detailed book-by-book synthesis extracted from the original skill

Use those references when applying these patterns to real Python/FastAPI code.

---

## Architecture Decision Record Mini-Template


Use an ADR when the choice affects module boundaries, dependencies, deployment, data ownership, provider selection, or long-term maintenance. Keep it short enough that future maintainers will actually read it.

```markdown
# ADR-NNN: <decision title>

## Verification Checklist


Before claiming a design is good:

- [ ] The problem is clearly stated before the pattern is named
- [ ] The chosen pattern’s intent matches the actual problem
- [ ] A simpler alternative was considered
- [ ] Trade-offs are documented: indirection, complexity, testability, performance
- [ ] Dependencies point in the intended direction
- [ ] Boundaries pass stable DTOs/events/value objects, not framework internals
- [ ] Tests protect behavior before refactoring
- [ ] The team/process impact was considered, not only code structure
- [ ] The design can be explained in plain language without pattern jargon
