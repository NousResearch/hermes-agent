# Product Brief Dialog: {{project_name}}

**Agent:** Saga (Product Brief Analyst)
**Project:** {{project_name}}
**Started:** {{start_date}}
**Status:** {{in_progress|completed}}
**Last Updated:** {{current_date}}

---

## About This Dialog

This dialog tracks the Product Brief discovery process - the conversations, reflections, decisions, and synthesis that led to the documented brief.

---

## Project Context

**Client/Stakeholder:** {{client_name}} ({{relationship}})
**Designer:** {{designer_name}}
**Sign-off Authority:** {{who_approves}}
**Project Type:** {{internal|external|agency}}

**Working Relationship:**
{{Brief description of stakes, involvement level, how directive to be}}

---

## Progress Tracker

- [ ] [Vision Capture](02-vision.md) — What we're building and why
- [ ] [User Definition](03-users.md) — Who we're building for
- [ ] [Product Concept](04-concept.md) — The founding structural idea
- [ ] [Core Features](05-features.md) — Essential functionality
- [ ] [Inspiration & References](06-inspiration.md) — Visual preferences and references
- [ ] [Positioning](07-positioning.md) — Market position and differentiation
- [ ] [Success Metrics](08-metrics.md) — How we measure success
- [ ] [Constraints](09-constraints.md) — Limitations and boundaries
- [ ] [Launch Requirements](10-launch.md) — What's needed to ship
- [ ] [Timeline & Phases](11-timeline.md) — Roadmap and milestones
- [ ] [Review & Synthesis](12-synthesis.md) — Final review and signoff

---

## Key Decisions

See [decisions.md](decisions.md) for detailed decision log.

**Major decisions:**
1. {{decision_summary_1}}
2. {{decision_summary_2}}
3. {{decision_summary_3}}

---

## Reflection Quality

**Total Checkpoints:** {{count}}
**Confirmed First Try:** {{count}}
**Required Correction:** {{count}}

This measures how well the agent understood the user's intent.

---

## Dialog Artifacts

All dialog files are timestamped and track the natural conversation flow, not just the final outputs.

**Purpose:** Enable future agents (or humans) to understand WHY decisions were made, not just WHAT was decided.

---

**Generated Artifacts:**
- [wds-project-outline.yaml](../../projects/{{project_slug}}/wds-project-outline.yaml)
- [Product Brief documentation](../../projects/{{project_slug}}/A-Product-Brief/)
