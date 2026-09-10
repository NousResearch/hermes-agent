---
name: doubt-driven-development
description: 'Fresh-context adversarial review before non-trivial decisions stand. Use when correctness matters more than speed, in unfamiliar code, high-stakes changes, or when a confident output would be cheaper to verify now than debug later.'
version: 1.0.0
author: Hermes Agent (adapted from community)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [review, quality, verification, adversarial, safety]
    related_skills: [gstack-review, verification-loop, safety-guard]
---

# Doubt-Driven Development

A confident answer is not a correct one. Long sessions accumulate context that quietly turns assumptions into "facts" without anyone noticing. Doubt-driven development is the discipline of materializing a fresh-context reviewer — biased to **disprove**, not approve — before any non-trivial output stands.

## When to Use

A decision is **non-trivial** when at least one of these is true:

- It introduces or modifies branching logic
- It crosses a module or service boundary
- It asserts a property the type system or compiler cannot verify (thread safety, idempotence, ordering, invariants)
- Its correctness depends on context the future reader cannot see
- Its blast radius is irreversible (production deploy, data migration, public API change)

Apply the skill when:

- About to make an architectural decision under uncertainty
- About to commit non-trivial code
- About to claim a non-obvious fact ("this is safe", "this scales")
- Working in code you don't fully understand

**When NOT to use:**

- Mechanical operations (renaming, formatting, file moves)
- Following a clear, unambiguous user instruction
- Reading or summarizing existing code
- One-line changes with obvious correctness

## The Process

### Step 1: CLAIM

Write the claim + why-it-matters in one sentence.

### Step 2: EXTRACT

Isolate the artifact (code, design, decision) and its contract. Strip all reasoning — just the artifact and what it promises.

### Step 3: DOUBT

Spawn a fresh-context reviewer via `delegate_task` with an adversarial prompt: "Find what's wrong with this. Assume it's incorrect. Prove it."

### Step 4: RECONCILE

Classify every finding against the artifact text:

- **True positive**: the artifact is wrong → fix it
- **False positive**: the reviewer misunderstood → document why
- **Clarification needed**: the artifact is ambiguous → sharpen it

### Step 5: STOP

Stop when: all findings are trivial, 3 cycles completed, or user overrides.

## Hermes Integration

- Use `delegate_task` to spawn the fresh-context adversarial reviewer
- Use `read_file` to extract the artifact
- Use `write_file` / `replace_string_in_file` to apply fixes
- Combine with `gstack-review` for comprehensive code review
- Combine with `safety-guard` for high-stakes production changes
