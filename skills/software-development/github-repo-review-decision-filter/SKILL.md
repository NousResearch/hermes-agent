---
name: github-repo-review-decision-filter
description: Use when you need a very fast yes/no filter on whether a GitHub repo is worth deeper review for Hermes. Separates reusable structure from reference-only content and enforces a safe default surface.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [github, repo-review, triage, curation, decision-filter]
    related_skills: [external-skill-library-curation, code-review, tool-selection-discipline]
---

# GitHub Repo Review Decision Filter

## Overview

Use this skill when a GitHub repository needs a very fast yes/no filter before deeper analysis. The goal is to decide whether the repo deserves Hermes attention, whether it should be captured only as reference material, or whether it should become a reusable bundle or skill source.

This skill is intentionally conservative. It privileges clear purpose, active maintenance, reusable structure, and a safe default surface over novelty or raw content volume.

## When to Use

- A user shares a GitHub repo and asks whether it is worth exploring
- You need a quick triage before a deeper repository review
- You want to separate structure from content before building Hermes notes or skills
- You need to decide whether a repo should become a Hermes bundle, a reference note, or be skipped
- You want to avoid importing unsafe, stale, duplicated, or monolithic material into the default Hermes surface

## Fast Decision Flow

### 1) Check the repo’s purpose

Ask:

- Is the repo clearly about a problem Hermes actually cares about?
- Is the domain relevant to the current task, fleet, or knowledge base?

If not, stop unless the user explicitly wants broad research.

### 2) Check whether the structure is reusable

Look for:

- README or front door
- index or catalog
- taxonomy or categories
- repeated workflow shape
- verification steps
- metadata or coverage map

If the repo is mostly content without structure, it is harder to adapt safely.

### 3) Check for current activity

Prefer repos with signs of usefulness now:

- recent commits or releases
- maintained docs
- obvious active organization
- current references or packaging metadata

A stale repo can still be useful, but only if the content is exceptional and the structure is reusable.

### 4) Check the safety boundary

Ask:

- Can the useful material be separated from unsafe material?
- Is the default surface safe for Hermes to import?
- Is offensive or dual-use content clearly isolatable?

If the default surface is unsafe, keep it out of the main Hermes lane.

### 5) Check the reuse boundary

Classify the repo into one of four buckets:

- reusable now
- reusable after adaptation
- reference only
- do not import

If the repo fails the filters, keep only the useful reference notes and do not build a Hermes bundle around it.

## Keep

- strong metadata
- a clear front door, README, or index
- repeatable workflows
- verification steps
- taxonomy or coverage mapping
- role-specific organization
- packaging or discovery artifacts

## Skip

- stale repos with no clear update signal
- monolithic dumps with no discovery layer
- offensive material in the default surface
- duplicate repos that add no new structure
- content that cannot be safely separated into reusable pieces

## Hermes Role Split

Use the fleet split to route follow-up work:

- Hermes00: catalog, propagation, note structure
- Hermes01: canary validation, spot checks, live verification
- Hermes02: monitoring, admin validation, low-noise reporting

## Output Shape

After triage, return one of these conclusions:

- Keep going: deeper review warranted
- Reference only: useful material, but no Hermes bundle
- Adaptable: extract structure first, then content
- Skip: not worth further Hermes attention

Include a one-line reason and the specific evidence that drove the decision.

## Common Pitfalls

1. Confusing content volume with usefulness.
2. Importing offensive or dual-use material into the default Hermes surface.
3. Reviewing content before identifying the repo’s structure.
4. Skipping the front door / index check.
5. Treating a stale repo as unusable without checking whether it has a reusable pattern.
6. Building a Hermes bundle around a repo that should only be reference material.

## Verification Checklist

- [ ] Repo purpose is clear
- [ ] Structure is reusable or clearly not reusable
- [ ] Activity level has been checked
- [ ] Safety boundary has been evaluated
- [ ] Reuse bucket is explicit
- [ ] Hermes role split is clear
- [ ] Output includes a concise decision and reason
- [ ] Unsafe material is not placed in the default Hermes surface
