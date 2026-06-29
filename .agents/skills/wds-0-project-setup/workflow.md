---
name: wds-0-project-setup
description: Project onboarding - determine project type, complexity, tech stack, and route to correct phase
---

# Phase 0: Project Setup

**The starting point for every WDS project.**


## Purpose

Phase 0 ensures you start on the right path. Before diving into design work, we need to understand:

1. **What is WDS?** - So you know what you're getting
2. **What type of project?** - New product vs existing product
3. **How complex?** - Landing page vs web application
4. **What tech?** - Framework and component library choices
5. **What's the right workflow?** - Route to the correct phase with right config

---

## Why This Matters

**The #1 mistake**: Starting Phase 1 with an existing codebase.

- **Phase 1-7** = Building something NEW (Greenfield)
- **Phase 8** = Improving something EXISTING (Brownfield, Product Evolution)

Wrong path = wasted work. Phase 0 prevents this.

---

## The Flow

```
Phase 0: Project Setup
    │
    ├─→ Step 01: Welcome
    │       ├─→ "What is WDS?" (quick intro)
    │       └─→ "Greenfield or Brownfield?"
    │
    └─→ Step 02: Configuration
            ├─→ Project name
            ├─→ Product complexity (landing/website/app)
            ├─→ Tech stack (optional)
            ├─→ Component library (optional)
            ├─→ Brief level (complete/simplified)
            ├─→ Strategic analysis (full/simplified/skip)
            ├─→ Create folder structure
            └─→ Generate project outline
                    │
                    ├─→ Greenfield → Phase 1 (→ Phase 2 if full analysis)
                    └─→ Brownfield → Phase 8
```

---

## Steps

| Step | Name | Purpose |
|------|------|---------|
| 01 | [Welcome & Orientation](steps/step-01-welcome.md) | Introduce WDS, determine greenfield vs brownfield |
| 02 | [Configuration & Structure](steps/step-02-structure.md) | Configure project, create folders, generate outline |

---

## Entry Point

**Start here**: `steps/step-01-welcome.md`

---

## When to Use Phase 0

- First time using WDS
- Starting a new project
- Joining an existing project
- Unsure which workflow to use

---

## When to Skip Phase 0

- Project outline already exists (`.wds-project-outline.yaml`)
- You know exactly which phase you need
- Continuing work on established WDS project

---

**Phase 0 takes 3-5 minutes. It saves hours of wrong-path work.**

---

## Configuration Options

| Option | Values | Impact |
|--------|--------|--------|
| Project Type | greenfield / brownfield | Determines Phase 1-7 (greenfield) vs Phase 8 (brownfield) |
| Complexity | simple / standard / complex | Which phases are enabled |
| Tech Stack | react / vue / wordpress / etc. | Delivery format guidance |
| Component Library | shadcn / tailwind / custom | Skip or enable Phase 5 |
| Brief Level | complete / simplified | Depth of Phase 1 |
| Strategic Analysis | full / simplified / skip | Full Trigger Map or simplified in brief |
