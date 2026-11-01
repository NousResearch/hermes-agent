---
name: github-repo-review-deep-review
description: Use when a GitHub repo has already passed the fast filter and needs a deeper evidence-based review for Hermes. Covers structure, content, security, maintainability, and Hermes-fit recommendations.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [github, repo-review, deep-review, evidence, hermes]
    related_skills: [github-repo-review-decision-filter, external-skill-library-curation, code-review, verification-first-operations]
---

# GitHub Repo Deep Review

## Overview

Use this skill after a repository has passed the fast repo-review filter and deserves a deeper, evidence-based assessment. The goal is to answer not just “is this repo interesting?” but “what exactly is reusable, what is risky, what is missing, and how should Hermes use it?”

This skill is intentionally structured around evidence gathering. Do not jump to conclusions based on the README alone, and do not promote a repo into Hermes without checking structure, maintenance signals, safety boundaries, and implementation quality.

## When to Use

- A repo already passed the fast yes/no filter and needs a deeper review
- You need a Hermes-oriented assessment of a repo’s structure, quality, and reusability
- You want to decide whether to import, reference, adapt, or skip specific parts
- You need a review that includes security, maintainability, and operational fit
- You want to convert the review into a note, bundle, or reusable skill source

## Prerequisite

Run the fast filter first:
- [[github-repo-review-decision-filter]]

If the repo fails the fast filter, stop here unless the user explicitly asked for deeper analysis anyway.

## Review Workflow

### 1) Establish the source facts

Capture the basic identifiers first:

- repository owner/name
- purpose statement
- license
- default branch
- recent activity signal
- packaging or distribution model
- whether the repo is library, app, reference set, tool, or catalog

Prefer live evidence over assumptions. If the repo has a README, changelog, releases, or generated index, read them first.

### 2) Inspect the front door and discovery surface

Check:

- README or landing page
- index or catalog files
- topic organization
- folder layout
- examples or quick starts
- contribution guidance
- changelog or release notes

Ask whether a newcomer can understand the repo without reading source files immediately.

### 3) Evaluate the structure

Look for reusable patterns:

- one concept per file or folder
- consistent naming
- clear boundaries between domains
- sidecar references, templates, or scripts
- generated indexes or coverage maps
- versioning or release discipline

Also note structural debt:

- duplicated flows
- overlong monolithic documents
- unclear category boundaries
- mixed-purpose folders
- missing front doors

### 4) Evaluate the content quality

For the core files, ask:

- Are the workflows concrete and executable?
- Are prerequisites and verification steps explicit?
- Is the prose operational or just descriptive?
- Are examples current and actually useful?
- Do commands, paths, and names look realistic?

For code or scripts, check whether the implementation matches the surrounding docs.

### 5) Check safety and risk

Identify:

- secrets or credentials
- destructive or destructive-adjacent actions
- offensive or dual-use material
- dependency or supply-chain risk
- unsafe defaults
- licensing friction
- misleading claims

If the repo mixes safe and unsafe material, record the boundary clearly.

### 6) Assess Hermes fit

Classify the repo into one or more Hermes uses:

- reference only
- note extraction
- skill adaptation
- front-door / index pattern
- taxonomy source
- operational runbook source
- deny-list / restricted-lane material

Then decide how Hermes should handle it:

- import structure, not content
- import selected workflows
- keep as external reference
- skip entirely

### 7) Form a recommendation

Return a concise recommendation with:

- decision
- confidence
- why it matters to Hermes
- what to extract next, if anything
- what to avoid importing

## Evidence Checklist

Use evidence from the repo itself whenever possible:

- README / front door
- index or catalog
- changelog / releases / tags
- folder layout
- representative skill or workflow file
- scripts or templates
- license and contribution docs
- generated coverage or mapping files

If you make a claim, attach the exact file or section that supports it.

## Output Shape

Return the review in this order:

1. Repository snapshot
2. Structure assessment
3. Content assessment
4. Safety and risk
5. Hermes fit
6. Recommendation
7. Evidence list

Keep the recommendation short and action-oriented. Distinguish clearly between:
- reusable now
- reusable after adaptation
- reference only
- skip

## Common Pitfalls

1. Reading source files before understanding the repo’s discovery surface.
2. Confusing active maintenance with superficial freshness.
3. Treating documentation quality as proof of implementation quality.
4. Ignoring licensing or redistribution constraints.
5. Importing the whole repo instead of extracting the reusable part.
6. Failing to note the line between safe and unsafe material.
7. Making a recommendation without quoting or pointing to evidence.

## Verification Checklist

- [ ] Repo identity and purpose are captured
- [ ] Discovery surface was reviewed first
- [ ] Structure was evaluated separately from content
- [ ] Safety and licensing were checked
- [ ] Hermes fit was classified explicitly
- [ ] Recommendation includes confidence and next action
- [ ] Evidence list points to concrete files or sections
- [ ] Unsafe or irrelevant material is not promoted into the default Hermes surface
