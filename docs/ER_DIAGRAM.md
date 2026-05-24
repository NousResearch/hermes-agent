---
title: Hermes Agent 3-Tier Knowledge Center — ER Diagram & Implementation Plan
tags:
  - hermes-agent
  - knowledge-center
  - architecture
  - implementation-plan
status: active
updated: 2026-05-23
---

# Hermes Agent 3-Tier Knowledge Center

## Entity-Relationship Diagram

```mermaid
erDiagram
    PROJECT ||--o{ CONTEXT_PACK : has
    PROJECT ||--o{ PROJECT_KB : "local knowledge (Tier 1)"
    PROJECT ||--o{ DOMAIN_TAG : "belongs to"

    DOMAIN ||--o{ DOMAIN_KB : "shared knowledge (Tier 2)"
    DOMAIN ||--o{ PROJECT : "categorizes"

    PLAYBOOK ||--o{ KNOWLEDGE_ENTRY : "promoted to (Tier 3)"

    KNOWLEDGE_ENTRY }o--|| PROJECT : "originated from"
    KNOWLEDGE_ENTRY }o--|| DOMAIN : "classified under"
    KNOWLEDGE_ENTRY ||--|| REVIEW_STATUS : "has"

    USER ||--o{ KNOWLEDGE_ENTRY : "approves/rejects"
    USER ||--o{ PROMOTE_PREFERENCE : "sets"

    PROMOTE_PREFERENCE }o--|| DOMAIN : "applies to"
    PROMOTE_PREFERENCE }o--|| PROJECT : "scoped to"

    CURATOR ||--o{ KNOWLEDGE_ENTRY : "reviews"
    CURATOR ||--o{ DOMAIN_KB : "archives stale"
    CURATOR ||--o{ PLAYBOOK : "reviews"

    PROJECT {
        string slug PK
        string name
        string path
        string stack
        string risk
        string role
    }

    CONTEXT_PACK {
        string slug FK
        string scripts
        string top_files
        string top_dirs
        string first_gate
    }

    PROJECT_KB {
        string id PK
        string project_slug FK
        string title
        text content
        timestamp created_at
        timestamp updated_at
    }

    DOMAIN {
        string slug PK
        string name
        string description
    }

    DOMAIN_KB {
        string id PK
        string domain_slug FK
        string title
        text content
        string[] relevant_projects
        timestamp promoted_at
        string origin_project
    }

    PLAYBOOK {
        string id PK
        string title
        text content
        timestamp created_at
        timestamp last_reviewed
        string status
    }

    KNOWLEDGE_ENTRY {
        string id PK
        string title
        text content
        string tier "1|2|3"
        string origin_project FK
        string domain_slug FK
        string status "pending|approved|rejected|archived"
        timestamp created_at
        timestamp promoted_at
    }

    REVIEW_STATUS {
        string knowledge_id FK
        string status "draft|review|approved|rejected|archived"
        string reviewer "agent|user|curator"
        timestamp reviewed_at
        string reason
    }

    PROMOTE_PREFERENCE {
        string id PK
        string user_id
        string domain_slug FK
        string project_slug FK
        string pattern
        boolean allow
        string reason
        timestamp created_at
    }

    USER {
        string id PK
        string name
    }

    CURATOR {
        string id PK
        string last_run
        int interval_hours
        int stale_after_days
    }
```

## Data Flow

```mermaid
flowchart TD
    A[Agent works on Project X] --> B{Creates new knowledge?}
    B -- No --> C[Continue work]
    B -- Yes --> D{Match existing preference?}
    D -- Yes: allow --> E[Auto-promote to Domain KB]
    D -- Yes: deny --> F[Keep in Project KB only]
    D -- No preference --> G{Cross-project relevance?}
    G -- No match --> F
    G -- Match found --> H[Ask user: promote to shared?]
    H -- Approve --> E
    H -- Reject --> F
    H -- Reject + remember --> I[Save deny preference]
    E --> J[Write to domains/{domain}/note.md]
    J --> K[Update domain index]
    K --> L[Update project note backlink]

    M[Agent starts work on Project Y] --> N[Load Tier 1: Project Y context pack]
    N --> O[Load Tier 2: domains matching Project Y's domains]
    O --> P[Load Tier 3: playbooks always in system prompt]
    P --> Q[Agent works with relevant knowledge]
```

## Token Budget Model

```mermaid
flowchart LR
    T1[Tier 1: Project-Local] -->|~500 tokens| SUM[Total per turn]
    T2[Tier 2: Domain-Shared] -->|~800-1500 tokens| SUM
    T3[Tier 3: Global Playbooks] -->|~300 tokens| SUM
    MEM[Memory System] -->|~200-500 tokens| SUM
    SUM -->|~1800-2800 tokens| TOTAL[Total: 1800-2800 tokens]
```

## Implementation Plan

See [11-knowledge-center-plan.md](./11-knowledge-center-plan.md) for the full 7-phase, 70-issue implementation plan with compliance tracking.

---
