# Canonical Engineering Books Reference

Detailed synthesis moved out of the main skill to keep the default load concise.

## 1. Clean Architecture — Robert C. Martin


**Core thesis:** Separate code into layers with strict dependency rules. Inner layers know nothing about outer layers.

**Patterns and heuristics:**

- **Dependency Rule** — source dependencies point inward. Business policy must not depend on frameworks, databases, UIs, or APIs.
- **Boundary Lines** — define what crosses a boundary: DTOs, commands, events, or primitives, not ORM models or framework objects.
- **Ports and Adapters** — put abstract interfaces near the policy that owns them; implement details at the edge.
- **Humble Object** — split testable logic from hard-to-test infrastructure.
- **SOLID applied:**
  - Single Responsibility — one reason to change
  - Open/Closed — add behavior by extension, not invasive edits
  - Liskov Substitution — subtypes preserve contracts
  - Interface Segregation — many narrow interfaces beat one fat interface
  - Dependency Inversion — policy depends on abstractions, details implement them

**Example structure:**

```text
src/
  domain/       # entities, value objects; no external dependencies
  use_cases/    # application business rules
  interfaces/   # ports owned by inner policy
  adapters/     # DB/API/UI/framework implementations
```

---

## 2. Domain-Driven Design — Eric Evans


**Core thesis:** Model the domain. Let code reflect the ubiquitous language of the business.

**Patterns and heuristics:**

- **Bounded Context** — each context owns its model. Do not share one canonical model across different business languages.
- **Aggregate** — cluster related entities/value objects under one root. Treat the aggregate as the consistency boundary.
- **Entity vs. Value Object** — entities have identity; value objects are equal by attributes.
- **Ubiquitous Language** — use the same terms in code, docs, tickets, and stakeholder conversations.
- **Domain Events** — represent important domain facts as events: `InvoicePaid`, `LeadQualified`, `ReportGenerated`.
- **Anticorruption Layer** — wrap legacy/external schemas so they do not leak into your domain model.

**Use when:** the business domain is complex enough that nouns, workflows, lifecycle states, and invariants matter.

---

## 3. GoF Design Patterns — Gamma, Helm, Johnson, Vlissides


**Core thesis:** Reusable OO design patterns describe recurring problems, solution structures, and trade-offs. They are not templates to force into code.

Prefer pattern recognition over pattern application. First name the problem; only then choose the pattern.

### Creational patterns — object creation

| Pattern | Intent | Use When |
|---------|--------|----------|
| Singleton | Ensure exactly one instance and provide a global access point | A true process-wide singleton is required, e.g. config registry; beware global state |
| Factory Method | Let subclasses or functions decide which concrete class to instantiate | Caller knows the abstraction, not the implementation |
| Abstract Factory | Produce families of related objects without naming concrete classes | Multiple product families must vary together, e.g. themed UI widgets |
| Builder | Separate construction of a complex object from representation | Construction has steps, optional parameters, or invariants |
| Prototype | Create objects by copying an existing instance | Cloning is cheaper or more flexible than construction |

### Structural patterns — object composition

| Pattern | Intent | Use When |
|---------|--------|----------|
| Adapter | Convert one interface into another clients expect | Wrapping third-party, legacy, or incompatible APIs |
| Bridge | Decouple abstraction from implementation so both can vary | Two dimensions vary independently; avoid subclass explosion |
| Composite | Treat individual objects and compositions uniformly | Trees: filesystem, UI components, org charts |
| Decorator | Add responsibilities dynamically without subclassing | Stackable behavior: middleware, buffered IO, logging wrappers |
| Facade | Provide a simplified interface to a subsystem | Hide subsystem complexity behind a common path |
| Flyweight | Share common intrinsic state among many fine-grained objects | Many identical-ish objects: particles, glyphs, tiles |
| Proxy | Control access to another object | Lazy loading, caching, security, remote access, instrumentation |

### Behavioral patterns — interaction and responsibility

| Pattern | Intent | Use When |
|---------|--------|----------|
| Chain of Responsibility | Pass request along handlers until one handles it | Middleware, validation chains, event bubbling |
| Command | Encapsulate a request as an object | Undo/redo, queues, macros, transaction logs |
| Interpreter | Represent grammar + interpreter for a small language | Tiny fixed DSL; otherwise prefer parser generators |
| Iterator | Traverse collections without exposing internals | Language-level iteration or custom traversal |
| Mediator | Centralize complex communication | Many objects need coordination; avoid mesh dependencies |
| Memento | Capture/restore state without exposing internals | Undo, snapshots, checkpoints |
| Observer | Notify dependents when state changes | Events, pub/sub, reactive UI, MVC |
| State | Change behavior when internal state changes | Explicit state machines without giant conditionals |
| Strategy | Interchangeable algorithms behind one interface | Sort/search/payment/provider selection at runtime |
| Template Method | Define an algorithm skeleton with overridable steps | Fixed pipeline with customizable stages |
| Visitor | Add operations to a stable object structure | AST traversal, file trees; avoid if node classes change often |

### GoF selection flowchart

```text
Is plain code enough?
  yes -> use plain functions/classes
  no
Need controlled creation?
  complex construction -> Builder
  choose concrete type -> Factory Method / Abstract Factory
  clone existing object -> Prototype
  exactly one true global -> Singleton, but challenge this hard
Need composition/adaptation?
  incompatible interface -> Adapter
  complex subsystem -> Facade
  stackable behavior -> Decorator
  tree uniformity -> Composite
  two independent variation axes -> Bridge
  controlled/lazy access -> Proxy
Need behavior orchestration?
  interchangeable algorithms -> Strategy
  fixed algorithm skeleton -> Template Method
  request pipeline -> Chain of Responsibility
  action object / undo / queue -> Command
  state-dependent behavior -> State
  one-to-many notification -> Observer
  stable object structure + new operations -> Visitor
```

### Natural pattern combinations

- Decorator + Composite — UI trees with bordered/scrollable components
- Observer + Mediator — observers register with a coordinating mediator
- Command + Memento — commands implement undo using captured pre-state
- Strategy + State — both delegate behavior; State transitions itself, Strategy is selected externally
- Adapter + Facade — adapter converts an interface; facade simplifies a subsystem
- Strategy + Registry — provider/plugin systems selected by name or capability
- Builder + Template Method — fixed construction pipeline with overrideable steps

---

## 4. Refactoring — Martin Fowler


**Core thesis:** Improve internal structure without changing external behavior, guided by tests.

**Code smells checklist:**

- Duplicated code
- Long method / long function
- Large class / god object
- Long parameter list
- Divergent change — one class changes for many unrelated reasons
- Shotgun surgery — one change requires touching many files
- Feature envy — a method manipulates another object’s data more than its own
- Primitive obsession — missing small domain objects/value types
- Switch statements that want polymorphism or Strategy
- Temporary fields and speculative generality

**Core refactorings:** Extract Method, Inline Method, Rename Variable, Move Method, Introduce Parameter Object, Extract Class, Replace Conditional with Polymorphism, Split Phase, Encapsulate Variable.

**Rules:**

- **Rule of Three** — wait until duplication appears three times before extracting a general abstraction.
- **Small safe steps** — refactor in behavior-preserving increments; run tests between steps.
- **Expand-Contract** — add the new interface, migrate callers, then remove the old interface.

---

## 5. The Pragmatic Programmer — Thomas & Hunt


**Core thesis:** Take responsibility for craft, tools, design, and feedback loops.

**Principles:**

- **DRY** — every piece of knowledge has one authoritative representation.
- **Orthogonality** — independent components can change independently.
- **Tracer Bullets** — build thin end-to-end functionality early for feedback.
- **Design by Contract** — define preconditions, postconditions, and invariants.
- **Temporal Coupling** — if A must happen before B, make the sequence explicit.
- **Plain Text Power** — prefer human-readable, tool-friendly formats where possible.
- **Automation** — automate repeated tasks; do not rely on memory.
- **Law of Demeter** — talk only to immediate collaborators; avoid train wreck calls.

---

## 6. Code Complete 2 — Steve McConnell


**Core thesis:** Construction quality matters. Most software cost is paid in code comprehension, modification, testing, and debugging.

**Rules of thumb:**

- Routines should do one thing completely and well.
- Keep variable scope minimal; initialize close to first use.
- Prefer named constants over magic numbers.
- Use guard clauses to reduce nesting.
- Prefer positive conditionals over double negatives.
- Hide complexity behind well-named abstractions.
- Class cohesion matters: everything in a class should belong together.
- Subclasses should specialize, not generalize.

---

## 7. Working Effectively with Legacy Code — Michael Feathers


**Core thesis:** Legacy code is code without tests. Improve it by finding seams and adding characterization tests.

**Patterns:**

- **Seam** — a place where behavior can change without editing that location.
- **Characterization Test** — capture existing behavior before changing it, even if the behavior is odd.
- **Sprout Method/Class** — add new behavior beside old behavior rather than rewriting everything.
- **Wrap Method/Class** — wrap old behavior with a new boundary.
- **Extract Interface** — introduce a mockable seam.
- **Subclass and Override** — isolate hard dependencies for tests.
- **Lean Test Addition** — add the minimum tests needed for the change at hand.

---

## 8. Peopleware — DeMarco & Lister


**Core thesis:** Software project failures are usually sociological before they are technological.

**Patterns:**

- **Protect Flow State** — batch interruptions; focus blocks are productive infrastructure.
- **Team Room / High-Bandwidth Collaboration** — create dedicated synchronous space when needed.
- **Quality of Work Life → Quality of Product** — burnout creates defects.
- **Remove Demotivators** — bad tools, unclear requirements, and constant interruptions hurt more than perks help.
- **Measure Outcomes, Not Outputs** — lines of code and ticket counts invite gaming.

---

## 9. The Mythical Man-Month — Frederick Brooks


**Core thesis:** Adding people to a late software project makes it later.

**Patterns:**

- **Brooks's Law** — communication paths scale as N(N-1)/2.
- **Surgical Team** — small focused teams with clear technical leadership beat committees.
- **Second-System Effect** — the second version is at highest risk of over-engineering.
- **No Silver Bullet** — no single tool/method eliminates essential complexity.
- **Plan to Throw One Away** — prototype to learn; expect the first design to be wrong.
- **Essential vs. Accidental Difficulty** — complexity, conformance, changeability, and invisibility are essential.

---

## 10. Structure and Interpretation of Computer Programs — Abelson & Sussman


**Core thesis:** Build the right abstractions. Computation is abstraction at multiple levels.

**Patterns:**

- **Data Abstraction** — separate operations from representation.
- **Procedural Abstraction** — name reusable computation patterns.
- **Metalinguistic Abstraction** — build small languages/DSLs to express domain ideas directly.
- **Closures** — functions with captured environments power callbacks, memoization, and partial application.
- **Streams / Lazy Evaluation** — separate sequence description from evaluation.
- **Interpreters** — when behavior is data, build an evaluator.

---

## Cross-Book Synthesis


```text
Architecture Decisions
  Clean Architecture Dependency Rule
  + DDD Bounded Contexts
  + Pragmatic Orthogonality
  -> boundary protection and modularity

Code Quality
  Pragmatic DRY
  + SOLID
  + Code Complete complexity management
  + Fowler smells
  -> readable, cohesive, maintainable code

Change Safety
  Feathers characterization tests
  + Fowler refactorings
  + expand-contract changes
  -> confident incremental improvement

Team Design
  Peopleware flow state
  + Brooks's Law
  + clear ownership
  -> small teams, focus, less coordination tax

Abstraction
  SICP abstraction
  + GoF patterns
  + DDD value objects
  + Interface Segregation
  -> right abstraction at the right level
```

---

## What Not to Cargo-Cult


Every canonical book has sharp ideas that can become harmful when copied mechanically.

- **Clean Architecture:** do not create four layers for a tiny CRUD endpoint. Use boundaries where policy and details genuinely need separation.
- **DDD:** do not introduce aggregates, repositories, and domain events for a simple data-entry app with no domain complexity.
- **GoF:** do not recreate Java-style inheritance hierarchies in Python/TypeScript when functions, registries, or interfaces solve the problem.
- **Refactoring:** do not refactor endlessly without a behavioral goal or test safety net.
- **Pragmatic Programmer:** do not interpret DRY as “remove all textual similarity.” DRY is about duplicated knowledge, not every repeated line.
- **Code Complete:** do not turn local style preferences into universal laws. Optimize for team consistency and comprehensibility.
- **Legacy Code:** do not try to test the entire legacy system before making one change. Add targeted characterization tests around the seam you need.
- **Peopleware:** do not use “protecting flow” as an excuse to avoid necessary communication or slow feedback loops.
- **Mythical Man-Month:** do not treat small teams as a universal fix. Some work needs coordination; the point is to make coordination explicit and bounded.
- **SICP:** do not build a DSL just because you can. Build one when it makes domain ideas simpler than host-language code.

---

