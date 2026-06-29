# UX Scenarios: {{project_name}}

> Design experiences, not screens — every page serves a user with a goal and an emotion.

**Created:** {{date}}
**Phase:** 3 (Scenario Outline) + Phase 4 (UX Design)
**Agents:** Saga (Scenario Outline), Freya (Page Specifications)

---

## What Belongs Here

Scenarios organize the product into meaningful user journeys. Each scenario groups related pages. Each page gets a full specification that a developer can build from.

**Folder structure per scenario:**
```
C-UX-Scenarios/
├── 00-ux-scenarios.md          ← This file (scenario guide + page index)
├── 01-scenario-name/
│   ├── 1.1-page-name/
│   │   ├── 1.1-page-name.md   ← Page specification
│   │   └── Sketches/           ← Wireframes and concepts
│   ├── 1.2-page-name/
│   │   ├── 1.2-page-name.md
│   │   └── Sketches/
│   └── ...
├── 02-scenario-name/
│   └── ...
├── Components/                  ← Shared component specs
└── Features/
    └── Storyboards/             ← Multi-step interaction flows
```

**Learn more:**
- WDS Course Module 08: Outline Scenarios — Design Experiences Not Screens
- WDS Course Module 09: Conceptual Sketching
- WDS Course Module 10: Storyboarding
- WDS Course Module 11: Conceptual Specifications
- WDS Course Tutorial 08: From Trigger Map to Scenarios

---

## For Agents

### Scenario Outline (Saga)
**Workflow:** `skill:wds-3-scenarios`
**Agent trigger:** `SC` (Saga)

### Page Specifications (Freya)
**Workflow:** `skill:wds-4-ux-design`
**Agent trigger:** `UX` (Freya)
**Page template:** `./resources/wds-4-ux-design/templates/page-specification.template.md`
**Scenario template:** `./resources/wds-4-ux-design/templates/scenario-overview.template.md`
**Quality guide:** `./resources/agent-guides/freya/specification-quality.md`
**Object types:** `./resources/wds-4-ux-design/object-types/`

### Specification Audit (Freya)
**Workflow:** `skill:wds-4-ux-design`
**Agent trigger:** `SA` (Freya)

**Before writing any page specification:**
1. Read `B-Trigger-Map/` — know the personas and their driving forces
2. Read the page specification template — use it as your scaffold, not memory
3. Discuss the page purpose with the user before filling in details
4. Each page folder needs a `Sketches/` subfolder for wireframes

**Harm:** Producing page specs from memory of what the template "roughly" contains. Plausible-looking specs that use wrong structure break the pipeline — developers can't trust them, audits can't validate them, and the user must correct what should have been right.

**Help:** Reading the actual template into context, discussing page purpose with the user, then filling the template with specific content. Specs that follow the template work across projects, pass audits, and give developers confidence.

---

## Scenarios

_This section will be updated as scenarios are outlined during Phase 3._

---

## Page Index

_This section will be updated as page specifications are created during Phase 4._

---

_Created using Whiteport Design Studio (WDS) methodology_
