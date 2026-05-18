# Session Workflow — Plan Lifecycle

This document describes how business-specifiers, technical-architects, and
implementors create, work with, and complete plans.

## Directory Structure

```
docs/
├── plans-active/      # In-progress plans (specifications, designs, implementation plans)
├── plans-complete/    # Completed plans, prefixed with sortable completion date
└── agent-index.md     # Index of active plans (this file is the entry point)
```

## Roles and Responsibilities

### Business Specifiers

When working on a business specification or feature spec:

1. **Write the plan** into `docs/plans-active/` using a descriptive slug as the
   filename (e.g., `stripe-integration-spec.md`).
2. **Index the plan** by adding an entry to the Active Plans table in
   `docs/agent-index.md`.
3. Use the `plan` skill for the writing workflow if available.
4. When the spec is ready for review, block the task and request human review.

### Technical Architects

When working on a technical design or architecture plan:

1. **Write the plan** into `docs/plans-active/` using a descriptive slug as the
   filename (e.g., `rate-limiter-architecture.md`).
2. **Index the plan** by adding an entry to the Active Plans table in
   `docs/agent-index.md`.
3. Reference any upstream business specs that this plan builds on.
4. When the design is ready for review, block the task and request human review.

### Implementors (Coders)

When implementing from an approved plan:

1. **Work from the plan** in `docs/plans-active/`. Read it before starting.
2. **Annotate changes** — if you deviate from the plan, do NOT overwrite the
   original content. Instead, append annotations:
   - `[CHANGED: <brief description>]` — marks something done differently than
     specified, with an explanation of what was actually done and why.
   - `[DOWNSTREAM: <brief description>]` — marks a change that impacts downstream
     plans or later implementation steps, explaining what downstream work will
     need to adjust.
3. **On completion:**
   - Move the plan file from `docs/plans-active/` to `docs/plans-complete/`.
   - Rename the file to prefix it with the completion date in `YYYY-MM-DD-`
     format (e.g., `2026-05-20-rate-limiter-architecture.md`).
   - Remove the plan's entry from the Active Plans table in `docs/agent-index.md`.
     (Do not replace it with references to completed plans — the index is for
     active work only.)
4. If the implementation uncovers issues that affect other active plans, update
   those plans with `[DOWNSTREAM]` annotations as well.

## Change Annotation Format

Changes to plans must be additive annotations, not overwrites. This preserves the
original intent and makes it clear what actually happened vs. what was planned.

### [CHANGED] — Local deviation

Use when the implementation differs from the plan for this component only.

```markdown
### Original Plan
Use Redis for rate limit counters.

### [CHANGED: Used in-memory counters instead]
Redis added an extra dependency and operational burden. For the current scale
(≤10k req/s), a per-process token bucket with `threading.Lock` is sufficient.
Can be swapped to Redis later if horizontal scaling is needed.
```

### [DOWNSTREAM] — Impacts other plans

Use when a change in this plan means downstream plans or later steps need to
adjust.

```markdown
### [DOWNSTREAM: Auth middleware needs updating]
The rate limiter now keys on `user_id` with IP fallback, not IP-only. The auth
middleware spec (see `session-management-spec.md`) assumed IP-only keys and will
need to be updated to pass `user_id` context.
```

## File Naming Conventions

- **Active plans:** `<descriptive-slug>.md` (e.g., `stripe-integration-spec.md`)
- **Completed plans:** `YYYY-MM-DD-<descriptive-slug>.md`
  (e.g., `2026-05-20-stripe-integration-spec.md`)
- Use lowercase, hyphen-separated slugs.
- The date prefix on completed plans is sortable and makes it easy to see
  completion order.
