# {scenario-number}-{scenario-name}

**Project:** {project-name}  
**Created:** {date}  
**Method:** Whiteport Design Studio (WDS)

---

## Scenario Overview

**User Journey:** {High-level description of what users accomplish in this scenario}

**Entry Point:** {Where users begin this scenario}  
**Success Exit:** {Where users end after successful completion}  
**Alternative Exits:** {Other possible endpoints - errors, cancellations, etc.}

**Target Personas:** {Which personas from Trigger Map use this scenario}

---

## Pages in This Scenario

| Page # | Page Name   | Status           | Purpose         |
| ------ | ----------- | ---------------- | --------------- |
| {n}.1  | {page-name} | {draft/complete} | {Brief purpose} |
| {n}.2  | {page-name} | {draft/complete} | {Brief purpose} |
| {n}.3  | {page-name} | {draft/complete} | {Brief purpose} |

---

## User Flow

```mermaid
flowchart TD
    A[{Entry Point}] --> B[{Page n.1}]
    B --> C[{Page n.2}]
    C --> D{{Decision Point?}}
    D -->|Yes| E[{Page n.3}]
    D -->|No| F[{Alternative Path}]
    E --> G[{Success Exit}]
    F --> G
```

---

## Scenario Steps

### Step 1: {Step Name}

**Page:** {n.1-Page-Name}  
**User Action:** {What the user does}  
**System Response:** {How the system responds}  
**Success Criteria:** {What defines success for this step}

### Step 2: {Step Name}

**Page:** {n.2-Page-Name}  
**User Action:** {What the user does}  
**System Response:** {How the system responds}  
**Success Criteria:** {What defines success for this step}

{Repeat for all steps}

---

## Trigger Map Connections

### Positive Drivers Addressed

From Trigger Map analysis, this scenario serves these user goals:

- ✅ {Positive goal from Trigger Map}
- ✅ {Positive goal from Trigger Map}

### Negative Drivers Avoided

This scenario helps users avoid:

- ❌ {Negative outcome from Trigger Map}
- ❌ {Negative outcome from Trigger Map}

---

## Success Metrics

**Primary Metric:** {Main measure of scenario success}

**Secondary Metrics:**

- {Metric 1}
- {Metric 2}

**User Satisfaction Indicators:**

- {What indicates good user experience}

---

## Edge Cases & Error Handling

| Edge Case               | How Handled         | Page(s) Affected  |
| ----------------------- | ------------------- | ----------------- |
| {edge-case-description} | {handling-approach} | {page-references} |

---

## Technical Requirements

### Data Flow

```
{Entry} → [Fetch Data] → {Display} → [User Action] → [Validate] → [API Call] → {Success}
```

### API Endpoints Used

| Endpoint        | Page(s)     | Purpose        |
| --------------- | ----------- | -------------- |
| {endpoint-path} | {page-refs} | {what-it-does} |

### State Management

{How state is managed across pages in this scenario}

---

## Design Assets

**Scenario Folder:** `C-UX-Scenarios/{scenario-number}-{scenario-name}/`

**Page Specifications:**

- {n}.1-{page-name}/{page-name}.md
- {n}.2-{page-name}/{page-name}.md
- {n}.3-{page-name}/{page-name}.md

**Prototypes:**

- {n}.1-{page-name}/Prototype/
- {n}.2-{page-name}/Prototype/
- {n}.3-{page-name}/Prototype/

---

## Development Notes

{Any scenario-level technical considerations, performance requirements, security notes, etc.}

---

## Revision History

| Date   | Changes                  | Author   |
| ------ | ------------------------ | -------- |
| {date} | Initial scenario created | {author} |

---

_Created using Whiteport Design Studio (WDS) methodology_
