---
name: check-resolvable
description: "Use when checking reachability of all skills from the resolver chain — walk RESOLVER.md → skill descriptions → files, find dark skills (no resolver path), dead triggers (wrong description), and unreachable capabilities."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [resolver, audit, reachability, meta-skill, linter]
    related_skills: [skill-resolver, hermes-agent-skill-authoring]
---

# Check-Resolvable — Resolver Reachability Auditor

## Purpose

This meta-skill audits the entire resolver chain to find dark capabilities, dead paths, and unreachable skills. It is the resolver equivalent of a lint pass. Run it after creating new skills or when routing seems wrong.

## Audit Procedure

### Step 1: Gather the Ground Truth

1. Read all RESOLVER.md files across the skills tree
2. Read `skill-resolver`'s decision table
3. List all installed skills via `skills_list`

### Step 2: Build the Reachability Matrix

For every skill found:

**Check A — Is there a path in any RESOLVER.md?**
- Does the top-level or any category-level RESOLVER.md mention this skill?
- Does `skill-resolver`'s decision table mention it?
- If NO to all three → **DARK SKILL**

**Check B — Does the resolver trigger match the skill description?**
- Read the resolver's trigger phrase
- Read the skill's `description` field
- If they diverge → **DRIFT WARNING**

**Check C — Does the skill file actually load?**
- Can `skill_view(name)` return the SKILL.md?
- Does the frontmatter parse cleanly?
- If not → **DEAD SKILL**

### Step 3: Report Results

Output a structured report with:
- N reachable / N total
- Dark skills list with remediation
- Dead triggers list
- Drift warnings list

### Step 4: Fix

For each finding, either:
1. **Fix immediately** — patch the RESOLVER.md or skill-resolver
2. **Create a Kanban task** — for gaps needing deeper work

## Trigger Evals

| Input | Expected skill |
|---|---|
| "setup hermes gateway" | hermes-agent |
| "review this PR" | github-code-review |
| "debug this python crash" | python-debugpy |
| "design an architecture diagram" | architecture-diagram |
| "search for papers on RLHF" | arxiv |
| "serve this model with vllm" | serving-llms-vllm |
| "create a new skill for PDF extraction" | hermes-agent-skill-authoring |
| "make a p5.js generative art sketch" | p5js |

## Common Pitfalls

1. **RESOLVER.md and skill-resolver get out of sync.** Both must match.
2. **Forgetting to re-check after creating a skill.** Every create should trigger a check-resolvable pass.
3. **Old deleted skills still in RESOLVER.md.** Dead triggers confuse routing.
4. **Category resolvers going stale.**

## Verification Checklist

- [ ] Every skill in skills_list has a path from at least one RESOLVER.md
- [ ] Every entry in every RESOLVER.md points to a skill that exists
- [ ] skill-resolver decision table matches canonical RESOLVER.md
- [ ] No unresolved dark skills
- [ ] All gaps fixed (patch) or ticketed (Kanban task)
