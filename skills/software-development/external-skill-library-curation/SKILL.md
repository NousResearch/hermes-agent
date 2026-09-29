---
name: external-skill-library-curation
description: Use when evaluating an external skill library or prompt bundle for Hermes. Turns the source into a safe import subset, a reusable curation plan, and a front-door note.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [skills, curation, taxonomy, knowledge-management, hermes]
    related_skills: [hermes-agent-skill-authoring, memory-hygiene, response-quality, tool-selection-discipline]
---

# External Skill Library Curation

## Overview

Use this skill when a repository, package, or catalog contains many reusable workflows and you need to decide what belongs in Hermes, what should stay external, and how to present the result cleanly.

The goal is not to mirror the source blindly. The goal is to extract the reusable structure: discovery metadata, workflow shape, taxonomy, coverage gaps, safety boundaries, and the smallest useful import set.

## When to Use

- A repo presents many agent skills, prompts, playbooks, or workflows
- You need to decide what should become Hermes skills versus note-only references
- You want to adapt a large external library into a smaller Hermes-native bundle
- You need a defensive-only subset and a clear exclusion boundary for offensive material
- You want a front door, index, or taxonomy for discovery and routing

## Workflow

### 1) Identify the source pattern

Extract the structural elements that make the source useful:

- skill or note metadata
- domain or topic taxonomy
- discovery index
- coverage map or gap analysis
- per-item workflow shape
- references or sidecars
- packaging or distribution metadata

Write down the source’s organizing principle in one sentence.

If the source is a GitHub repo, do a quick front-door pass first: README, index/catalog, update cadence, and safety boundary before reading the details.

See `references/github-repo-review-ladder.md` for the session-derived GitHub repo triage ladder.

### 2) Separate structure from content

Classify each useful element into one of four buckets:

- reusable now
- reusable after adaptation
- reference only
- do not import

Default to conservative cleanup. If a source includes offensive, destructive, or dual-use material, keep it out of the default Hermes surface unless the user explicitly requests a restricted lane.

### 3) Derive the Hermes-native shape

For each candidate import, define:

- purpose
- when to use
- prerequisites
- inputs
- outputs
- verification
- related skills
- risk class

Prefer one workflow per skill. If multiple tasks share the same skeleton, make an umbrella skill and keep specifics in references.

### 4) Build the discovery surface

Create a compact front door that answers:

- what this bundle is
- what is safe to open first
- what is excluded
- what the best next note is
- how the bundle maps to Hermes roles

A good pattern is:

- front door / README
- index / catalog
- detailed adaptation note
- role-specific follow-ups
- changelog entry

### 5) Map to Hermes roles

Use the fleet split to route work:

- Hermes00: control plane, catalog, propagation, documentation
- Hermes01: canary validation, runtime correctness, live checks
- Hermes02: stable monitoring, admin validation, low-noise reporting

Do not give every role the same surface. Only load the skills that each role actually needs.

### 6) Verify the bundle

Before declaring the adaptation done, verify that:

- the new note or skill is linked from a front door
- the import set is conservative and safe
- the taxonomy is searchable
- the role split is explicit
- the bundle has a clear next-open path

## Common Pitfalls

1. Importing the whole source library instead of curating it.
2. Mixing offensive material into the default Hermes surface.
3. Creating too many narrow skills when one umbrella skill would do.
4. Skipping the index/front-door layer and leaving the bundle hard to rediscover.
5. Losing the source taxonomy when adapting the content.
6. Forgetting to add a verification step or risk boundary.

## Verification Checklist

- [ ] Source pattern identified in one sentence
- [ ] Reusable vs reference-only vs do-not-import split completed
- [ ] Conservative default set for Hermes chosen
- [ ] Front door or index exists
- [ ] Hermes role split is explicit
- [ ] Verification step included
- [ ] Changelog or front-door link added
- [ ] No offensive material was imported into the default surface
- [ ] If the source was a GitHub repo, the README/index/safety boundary were checked before deep extraction
