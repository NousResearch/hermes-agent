---
name: 'step-03-map-architecture'
description: 'Build a detailed map of components, data flow, dependencies, and patterns as the analytical core of the workflow'

# File References
nextStepFile: './step-04-document-findings.md'
---

# Step 3: Map Architecture

## STEP GOAL:

Build a detailed map of components, data flow, dependencies, and patterns. This is the analytical core of the workflow.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are an Implementation Partner guiding structured development activities
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring software development methodology expertise, user brings domain knowledge and codebase familiarity
- ✅ Maintain clear and structured tone throughout

### Step-Specific Rules:

- 🎯 Focus only on mapping components, tracing data flows, identifying layers, integrations, state, dependencies, and patterns
- 🚫 FORBIDDEN to begin writing the final architecture document — that is the next step
- 💬 Approach: Deep dive into codebase structure with user, tracing flows and mapping dependencies collaboratively
- 📋 Use the scan observations from Step 2 as your starting point

## EXECUTION PROTOCOLS:

- 🎯 Create comprehensive component inventory, data flow traces, and dependency maps
- 💾 Document component map, data flows, layers, integrations, state management, dependencies, and patterns
- 📖 Reference scan results from Step 2 and original questions from Step 1
- 🚫 Do not write the final document yet — focus on raw analysis

## CONTEXT BOUNDARIES:

- Available context: Analysis question and scope from Step 1; scan observations from Step 2
- Focus: Deep architectural analysis — components, flows, dependencies, patterns
- Limits: No final document writing — raw analysis only
- Dependencies: Steps 1 and 2 must be complete

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Map Component and Module Structure

For each major module or component area:

- What is its responsibility?
- What does it expose (public API, exports)?
- What does it encapsulate (internal details)?
- How large is it (files, lines, complexity)?

Create a component inventory:

```
Components:
- AuthModule — Handles login, registration, session management
- UserService — CRUD operations for user profiles
- PaymentGateway — Stripe integration, invoice generation
- NotificationService — Email + push notifications
- ...
```

### 2. Trace Data Flow

Pick 2-3 key user actions and trace data from start to finish:

1. **User action** (click, form submit, page load)
2. **Frontend handling** (event handler, state update, API call)
3. **API layer** (route, middleware, validation)
4. **Business logic** (service, domain logic, transformations)
5. **Data layer** (database query, cache, external API)
6. **Response path** (back through layers to UI)

Document at least one complete flow end-to-end.

### 3. Identify Layers

Determine the architectural layers and their boundaries:

| Layer | Purpose | Location |
|-------|---------|----------|
| **Presentation** | UI rendering, user interaction | `src/components/`, `src/pages/` |
| **Application** | Use cases, orchestration | `src/services/`, `src/hooks/` |
| **Domain** | Business rules, entities | `src/models/`, `src/domain/` |
| **Infrastructure** | Database, external APIs, file system | `src/db/`, `src/integrations/` |

Note: Not all codebases have clean layers. Document what you actually find, including layer violations.

### 4. Map External Integrations

List every external system the codebase talks to:

| Integration | Purpose | Protocol | Location in Code |
|-------------|---------|----------|-----------------|
| PostgreSQL | Primary data store | SQL via ORM | `src/db/` |
| Stripe | Payment processing | REST API | `src/payments/` |
| SendGrid | Email delivery | REST API | `src/notifications/` |
| Redis | Caching, sessions | Redis protocol | `src/cache/` |

### 5. Document State Management

How does the application manage state?

- **Frontend:** Global store (Redux, Zustand), context, local state, URL state
- **Backend:** Session storage, database, cache, in-memory
- **Shared:** How does state sync between client and server?

### 6. Create Dependency Graph

Map which modules depend on which:

- Identify core modules that many things depend on (high fan-in)
- Identify modules that depend on many things (high fan-out)
- Look for circular dependencies
- Note tightly coupled vs loosely coupled areas

### 7. Identify Architectural Patterns

Document patterns you observe:

- **Structural:** MVC, MVVM, Clean Architecture, Hexagonal, Monolith, Microservices
- **Data:** Repository pattern, Active Record, Data Mapper, CQRS
- **Communication:** REST, GraphQL, WebSocket, Event-driven, Message queue
- **Error handling:** Centralized, per-module, try-catch patterns, error boundaries
- **Testing:** Unit/integration/e2e split, mocking approach, test data strategy

### 8. Verify Checklist

- [ ] Component/module inventory created
- [ ] At least one data flow traced end-to-end
- [ ] Architectural layers identified
- [ ] External integrations mapped
- [ ] State management approach documented
- [ ] Key dependencies mapped (especially high fan-in/fan-out)
- [ ] Architectural patterns identified

### 9. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4: Document Findings"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the architectural mapping is complete with components, flows, and dependencies documented will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Component/module inventory created
- At least one data flow traced end-to-end
- Architectural layers identified
- External integrations mapped
- State management approach documented
- Key dependencies mapped (especially high fan-in/fan-out)
- Architectural patterns identified

### ❌ SYSTEM FAILURE:
- Skipping data flow tracing
- Not mapping external integrations
- Beginning final document creation before analysis is complete
- Ignoring dependency relationships

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
