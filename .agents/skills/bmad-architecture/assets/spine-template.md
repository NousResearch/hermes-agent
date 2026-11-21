---
name: '{name}'
type: architecture-spine
purpose: build-substrate    # build-substrate (default) · discussion · report · deck
altitude: feature           # initiative (keeps features) · feature (keeps epics) · epic (keeps stories)
paradigm: '{named design pattern, e.g. hexagonal, layered, pipes-and-filters, actor}'
scope: '{what this spine governs}'
status: draft               # draft · final
created: '{date}'
updated: '{date}'
binds: []                   # capability / unit IDs governed (from the driving spec; at epic altitude, also the inherited parent AD ids)
sources: []
companions: []
---

# Architecture Spine — {name}

<!-- TEMPLATE GUIDE — act on these comments, then delete them; never emit a comment in the finished spine. This is a shape, not a script: keep only the sections this spine needs and cut the rest (no empty headers). A small intent may be just paradigm + a few ADs + conventions; a platform earns more. An inherited epic spine is usually mostly Inherited Invariants + a thin Deferred. Decisions, not rationale (rationale lives in the memlog). Carry shape in diagrams; prose only where it must. -->

## Design Paradigm

<!-- Name the pattern (a known one loads a whole model for free) and map its layers to namespaces/directories. The smallest, most durable thing here. -->

## Inherited Invariants

<!-- Only when this spine inherits a higher-altitude parent. The parent's ADs/conventions/paradigm that bind here, by their ORIGINAL ids — read-only, never renumbered, not re-derived. A local decision that contradicts one is a conflict to surface, not an override. Cut this section otherwise. -->

| Inherited | From parent | Binds here |
| --- | --- | --- |
| {AD-id / convention} | {parent spine} | {what it constrains in this scope} |

## Invariants & Rules

<!-- The durable heart: calls a future builder can't read off compliant code. One block per decision: stable ascending id (never reused/renumbered), Binds, Prevents (the divergence), Rule (enforceable). Tag [ADOPTED] when the user or existing reality settled it. Include a dependency-direction diagram (who may depend on whom) — it IS a rule; author it as valid mermaid, never an empty graph. -->

### AD-1 — {decision}

- **Binds:** {capability / unit ids / fr/nfr's, areas, or `all`}
- **Prevents:** {the divergence this stops}
- **Rule:** {the constraint downstream must follow}

## Consistency Conventions

<!-- Defaults that bind where independent builders would drift. Cut rows that don't apply; add rows the project needs. -->

| Concern | Convention |
| --- | --- |
| Naming (entities, files, interfaces, events) | |
| Data & formats (ids, dates, error shapes, envelopes) | |
| State & cross-cutting (mutation, errors, logging, config, auth) | |

## Stack

<!-- SEED — verified current at authoring; the code owns this once it exists. Name + version only; the why lives in the memlog. One row per language, framework, key dependency, platform, or chain that's pinned. -->

| Name | Version |
| --- | --- |
| {language / framework / key dep / platform / chain} | {pinned version} |

## Structural Seed

<!-- The shapes worth fixing at cold-start — not a fixed list. Include only what's non-obvious at this altitude, and use as many diagrams as convey it, each as VALID mermaid (never a placeholder or empty graph). Candidates: system/container/context view; DEPLOYMENT & ENVIRONMENTS and external provider/infra topology (cover the operational envelope here when this altitude owns it — don't let it fall through); core-entity ERD (names + relationships only; an attribute that's itself an invariant is an AD, not a diagram); a minimal source tree. The code owns the detail — this is scaffold, not a mirror to maintain. -->

```text
{root}/
  {dir}/   # {what lives here}
```

## Capability → Architecture Map

<!-- Present when a spec drove this run. Bridges the spec's capabilities to where they live + what governs them; the consistency auditor's checklist. Cut otherwise. -->

| Capability / Area | Lives in | Governed by |
| --- | --- | --- |
| {CAP-id / area} | {component / module} | {AD-id, convention, paradigm} |

## Deferred

<!-- Decisions intentionally pushed down, each with the reason it can wait — including whole dimensions this altitude doesn't own yet. The half of the contract that keeps the spine lean. -->
