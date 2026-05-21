# GoF Pattern Catalog

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
