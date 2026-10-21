# Mermaid Diagram Reference Guide

This reference provides comprehensive guidance on creating effective Mermaid diagrams for design documents.

---

## Diagram Type Selection Matrix

| When You Need To... | Use This Diagram | Best For |
|---------------------|------------------|----------|
| Show system boundaries and external actors | C4 Context | System context, stakeholder view |
| Document API calls and timing | Sequence Diagram | API flows, interactions, temporal behavior |
| Model object relationships and inheritance | Class Diagram | OOP design, code structure |
| Visualize database schema | ER Diagram | Data model, relationships |
| Show state transitions and lifecycle | State Diagram | Workflows, status changes |
| Document decision flows and algorithms | Flowchart | Business logic, processes |
| Map hierarchical concepts | Mind Map | Brainstorming, concept organization |
| Track project timeline | Gantt Chart | Project planning, milestones |
| Capture user experience | User Journey | UX flows, user interactions |
| Show infrastructure components | Architecture Diagram | Deployment, infrastructure |

---

## 1. C4 Context Diagrams

**Purpose:** Show system boundaries, users, and external dependencies

**When to use:**
- System overview
- Stakeholder presentations
- Architecture documentation

**Syntax:**
```mermaid
C4Context
    title System Context for [System Name]

    Person(personAlias, "Person Name", "Role description")
    System(systemAlias, "System Name", "System description")
    System_Ext(externalAlias, "External System", "External description")
    SystemDb(dbAlias, "Database", "Database description")

    Rel(personAlias, systemAlias, "Uses", "HTTPS")
    Rel(systemAlias, externalAlias, "Calls", "REST API")
    Rel(systemAlias, dbAlias, "Reads/Writes", "SQL")
```

**Best Practices:**
- Limit to 5-7 systems for clarity
- Focus on key external dependencies
- Use consistent naming (nouns for systems, verbs for relationships)
- Include protocol information in relationship labels

---

## 2. Sequence Diagrams

**Purpose:** Show interactions between components over time

**When to use:**
- API documentation
- Request/response flows
- Authentication flows
- Error scenarios

**Syntax:**
```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant Database

    Client->>Server: POST /api/resource
    activate Server
    Server->>Database: INSERT query
    activate Database
    Database-->>Server: Success
    deactivate Database
    Server-->>Client: 201 Created
    deactivate Server

    Note over Client,Server: Authentication required

    alt Success Case
        Client->>Server: Valid Request
        Server-->>Client: 200 OK
    else Error Case
        Client->>Server: Invalid Request
        Server-->>Client: 400 Bad Request
    end
```

**Best Practices:**
- Order participants left-to-right by interaction flow
- Use activation boxes for processing time
- Include both happy path and error scenarios
- Add notes for important context
- Use alt/opt/loop for conditional logic

**Common Patterns:**
```mermaid
sequenceDiagram
    %% Request-Response
    A->>B: Request
    B-->>A: Response

    %% Synchronous
    A->>B: Sync Call
    activate B
    B-->>A: Result
    deactivate B

    %% Asynchronous
    A->>Queue: Publish Event
    Queue->>B: Consume Event
```

---

## 3. Class Diagrams

**Purpose:** Model object-oriented structure and relationships

**When to use:**
- OOP design documentation
- Code architecture
- Domain model

**Syntax:**
```mermaid
classDiagram
    class Animal {
        +String name
        +int age
        +makeSound() void
        -metabolize() void
    }

    class Dog {
        +String breed
        +bark() void
    }

    class Cat {
        +String color
        +meow() void
    }

    Animal <|-- Dog : inherits
    Animal <|-- Cat : inherits
    Dog "1" --> "*" Toy : has
    Cat "1" --> "1" Owner : belongs to

    <<interface>> Flyable
    Bird ..|> Flyable : implements
```

**Relationships:**
- `<|--` : Inheritance
- `*--` : Composition
- `o--` : Aggregation
- `-->` : Association
- `..>` : Dependency
- `..|>` : Realization

**Visibility:**
- `+` : Public
- `-` : Private
- `#` : Protected
- `~` : Package/Internal

---

## 4. ER Diagrams

**Purpose:** Model database schema and relationships

**When to use:**
- Database design
- Data architecture
- Schema documentation

**Syntax:**
```mermaid
erDiagram
    CUSTOMER ||--o{ ORDER : places
    CUSTOMER {
        int id PK "Primary key"
        string email UK "Unique constraint"
        string name
        date created_at
    }

    ORDER ||--|{ ORDER_ITEM : contains
    ORDER {
        int id PK
        int customer_id FK
        decimal total
        enum status "pending, shipped, delivered"
    }

    PRODUCT ||--o{ ORDER_ITEM : "ordered in"
    PRODUCT {
        int id PK
        string sku UK
        string name
        decimal price
    }

    ORDER_ITEM {
        int id PK
        int order_id FK
        int product_id FK
        int quantity
    }
```

**Cardinality:**
- `||--||` : One to one
- `||--o{` : One to zero or more
- `||--|{` : One to one or more
- `}o--o{` : Zero or more to zero or more

**Best Practices:**
- Mark primary keys with "PK"
- Mark foreign keys with "FK"
- Mark unique constraints with "UK"
- Add field descriptions for clarity
- Include enum values in descriptions

---

## 5. State Diagrams

**Purpose:** Model state transitions and lifecycles

**When to use:**
- Workflow documentation
- Status management
- Lifecycle modeling

**Syntax:**
```mermaid
stateDiagram-v2
    [*] --> Draft
    Draft --> Review : Submit
    Review --> Approved : Approve
    Review --> Draft : Reject
    Approved --> Published : Publish
    Published --> Archived : Archive
    Archived --> [*]

    state Review {
        [*] --> PendingReview
        PendingReview --> InReview : Assign
        InReview --> [*]
    }

    note right of Approved
        Requires admin approval
    end note
```

**Best Practices:**
- Start with `[*]` for initial state
- End with `[*]` for final state
- Use nested states for complex workflows
- Add notes for important transitions
- Keep transitions labeled with trigger events

---

## 6. Flowcharts

**Purpose:** Document processes, algorithms, and decision logic

**When to use:**
- Business logic documentation
- Process flows
- Algorithm explanation

**Syntax:**
```mermaid
flowchart TD
    Start([Start])
    Input[/User Input/]
    Process[Process Data]
    Decision{Valid?}
    Success[/Success Message/]
    Error[/Error Message/]
    End([End])

    Start --> Input
    Input --> Process
    Process --> Decision
    Decision -->|Yes| Success
    Decision -->|No| Error
    Success --> End
    Error --> Input

    subgraph "Validation"
        Decision
    end
```

**Node Shapes:**
- `[Text]` : Rectangle (process)
- `([Text])` : Stadium (start/end)
- `{Text}` : Diamond (decision)
- `[/Text/]` : Parallelogram (input/output)
- `[(Text)]` : Cylinder (database)
- `((Text))` : Circle
- `>Text]` : Asymmetric (flag)

**Directions:**
- `TD` / `TB` : Top to bottom
- `BT` : Bottom to top
- `LR` : Left to right
- `RL` : Right to left

---

## 7. Gantt Charts

**Purpose:** Project planning and timeline visualization

**When to use:**
- Project schedules
- Milestone tracking
- Roadmap planning

**Syntax:**
```mermaid
gantt
    title Project Roadmap
    dateFormat YYYY-MM-DD
    section Phase 1
    Research           :done, r1, 2025-01-01, 30d
    Design            :active, d1, 2025-01-15, 45d
    section Phase 2
    Development       :dev1, after d1, 60d
    Testing          :test1, after dev1, 30d
    section Phase 3
    Deployment       :crit, deploy1, after test1, 15d
    Monitoring       :monitor1, after deploy1, 30d
```

**Keywords:**
- `done` : Completed task
- `active` : Currently in progress
- `crit` : Critical path
- `after [id]` : Dependency

---

## 8. User Journey Maps

**Purpose:** Document user experience and interactions

**When to use:**
- UX documentation
- User story mapping
- Experience design

**Syntax:**
```mermaid
journey
    title User Purchase Journey
    section Discovery
      Search for product: 5: User
      View details: 4: User
      Read reviews: 3: User
    section Decision
      Compare options: 3: User
      Add to cart: 5: User
    section Purchase
      Enter payment: 2: User
      Confirm order: 5: User
      Receive confirmation: 5: User, System
```

**Satisfaction Scores:**
- 5: Very satisfied
- 4: Satisfied
- 3: Neutral
- 2: Unsatisfied
- 1: Very unsatisfied

---

## Common Styling Patterns

### Subgraphs for Organization

```mermaid
graph TB
    subgraph "Frontend"
        UI[User Interface]
        Components[Components]
    end

    subgraph "Backend"
        API[API Layer]
        Database[(Database)]
    end

    UI --> API
    API --> Database
```

### Color Coding with High Contrast

**CRITICAL**: All Mermaid diagram styles MUST use high-contrast colors for accessibility.

**Rule**: Light backgrounds require dark text, dark backgrounds require light text.

✅ **Correct - High Contrast**:

```mermaid
graph LR
    A[Normal]
    B[Success]
    C[Warning]
    D[Error]

    style A fill:#F0F0F0,stroke:#333,color:black
    style B fill:#90EE90,stroke:#333,color:darkgreen
    style C fill:#FFD700,stroke:#333,color:black
    style D fill:#FF6B6B,stroke:#8B0000,color:white
```

**Using classDef (Recommended for consistency)**:

```mermaid
graph LR
    A[Normal]
    B[Success]
    C[Warning]
    D[Error]

    classDef normalStyle fill:#F0F0F0,stroke:#333,stroke-width:2px,color:black
    classDef successStyle fill:#90EE90,stroke:#2E7D2E,stroke-width:2px,color:darkgreen
    classDef warningStyle fill:#FFD700,stroke:#B8860B,stroke-width:2px,color:black
    classDef errorStyle fill:#FFB6C1,stroke:#DC143C,stroke-width:2px,color:black

    class A normalStyle
    class B successStyle
    class C warningStyle
    class D errorStyle
```

❌ **Incorrect - Poor Contrast**:

```mermaid
graph LR
    %% Missing color property - may be unreadable
    style B fill:#90EE90
    style C fill:#FFD700
    style D fill:#FF6B6B
```

**High-Contrast Color Palette**:

| State | Background Fill | Text Color | Stroke |
|-------|----------------|------------|--------|
| Normal | `#F0F0F0` | `color:black` | `#333` |
| Success | `#90EE90` | `color:darkgreen` | `#2E7D2E` |
| Warning | `#FFD700` | `color:black` | `#B8860B` |
| Error | `#FFB6C1` | `color:black` | `#DC143C` |
| Info | `#87CEEB` | `color:darkblue` | `#4682B4` |
| Public | `#FFE4B5` | `color:black` | `#FF8C00` |
| Private | `#E6E6FA` | `color:darkblue` | `#8A2BE2` |
| Dark | `#2C3E50` | `color:white` | `#34495E` |

### Link Styles

```mermaid
graph LR
    A --> B
    A -.-> C
    A ==> D

    linkStyle 0 stroke:#00ff00,stroke-width:2px
    linkStyle 1 stroke:#0000ff,stroke-width:2px
    linkStyle 2 stroke:#ff0000,stroke-width:4px
```

---

## Best Practices for Design Docs

### 1. Choose the Right Diagram

**Don't use:**
- Class diagrams for infrastructure
- Sequence diagrams for data models
- ER diagrams for workflows

**Do use:**
- The diagram type that best communicates your intent
- Multiple diagram types for different aspects
- Simpler diagrams over complex ones

### 2. Keep Diagrams Focused

**Bad:** One giant diagram showing everything
**Good:** Multiple focused diagrams showing specific aspects

**Guidelines:**
- Max 10-12 nodes per diagram
- Max 3-4 levels of nesting
- Break complex diagrams into multiple views

### 3. Use Consistent Naming

**Bad:**
```mermaid
graph LR
    usr --> sys
    system --> db1
```

**Good:**
```mermaid
graph LR
    User --> System
    System --> Database
```

### 4. Add Context

```mermaid
sequenceDiagram
    Note over Client,Server: OAuth 2.0 Authentication Flow

    Client->>AuthServer: Request Access Token
    Note right of AuthServer: Validates client credentials
    AuthServer-->>Client: Access Token (JWT)
```

### 5. Document Technical Decisions

```mermaid
graph TB
    A[Option A: REST API]
    B[Option B: GraphQL]
    C[Decision: REST API]

    note1[Pros: Simple, cacheable]
    note2[Cons: Over-fetching]

    C --> note1
    B --> note2
```

### 6. CRITICAL - Ensure High-Contrast Accessibility

**MANDATORY for ALL diagrams**:

Every diagram with custom styling MUST use high-contrast colors:

```mermaid
graph LR
    A[Component A]
    B[Component B]
    C[Component C]

    classDef primary fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef secondary fill:#FFD700,stroke:#333,stroke-width:2px,color:black
    classDef accent fill:#87CEEB,stroke:#333,stroke-width:2px,color:darkblue

    class A primary
    class B secondary
    class C accent
```

**Quick Accessibility Test**:
1. Can you easily read the text on each background color?
2. Would the diagram be readable if printed in grayscale?
3. Does every `classDef` include a `color:` property?

If answer is NO to any → Fix the contrast!

**Common Mistakes to Avoid**:
- ❌ `classDef myStyle fill:#FFD700` (missing `color:`)
- ❌ `style A fill:#F0F0F0,color:#E0E0E0` (light on light)
- ❌ `style B fill:#333,color:#222` (dark on dark)

**Always Include**:
- ✅ Explicit `color:` property in all `classDef` statements
- ✅ Explicit `color:` property in all `style` statements
- ✅ High-contrast combinations (see Color Coding section above)

---

## Syntax Validation Checklist

Before including a diagram, verify:

- [ ] All node IDs are unique
- [ ] All relationships use valid syntax
- [ ] Quotes are balanced in labels
- [ ] Special characters are escaped
- [ ] Subgraph syntax is correct
- [ ] No trailing commas
- [ ] Direction is specified (for flowcharts)
- [ ] Date format matches (for Gantt)
- [ ] **All `classDef` statements include `color:` property for high contrast**
- [ ] **All `style` statements include `color:` property for high contrast**
- [ ] **Text is readable on all background colors (accessibility test)**

---

## Common Errors and Fixes

### Error: "Parse error on line X"

**Cause:** Syntax error in Mermaid code

**Fix:**
- Check for missing quotes
- Verify relationship syntax
- Ensure all braces/parentheses match
- Remove trailing commas

### Error: Diagram doesn't render

**Cause:** Invalid node IDs or special characters

**Fix:**
- Use alphanumeric IDs (avoid spaces)
- Escape special characters in labels
- Use quotes for multi-word labels

### Error: Arrows pointing wrong direction

**Cause:** Wrong relationship syntax

**Fix:**
- `A-->B` : A to B
- `B<--A` : A to B (same as above)
- `A<-->B` : Bidirectional

---

## Quick Reference: When to Use Each Diagram

```mermaid
graph TB
    Start{What are you documenting?}
    Start -->|System boundaries| C4[C4 Context Diagram]
    Start -->|API interactions| Seq[Sequence Diagram]
    Start -->|Code structure| Class[Class Diagram]
    Start -->|Database| ER[ER Diagram]
    Start -->|Workflow/Status| State[State Diagram]
    Start -->|Process flow| Flow[Flowchart]
    Start -->|Timeline| Gantt[Gantt Chart]
    Start -->|User experience| Journey[User Journey]
    Start -->|Infrastructure| Arch[Architecture Diagram]
```

---

## Resources

- [Mermaid Official Docs](https://mermaid.js.org/)
- [Mermaid Live Editor](https://mermaid.live/)
- [C4 Model](https://c4model.com/)
