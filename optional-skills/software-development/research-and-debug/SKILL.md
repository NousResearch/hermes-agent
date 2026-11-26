---
name: research-and-debug
description: "Use when validating ideas before building (spikes) or investigating bugs systematically (root cause before fix)."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [spike, prototype, feasibility, debug, investigation, root-cause, troubleshooting, exploration]
    related_skills: [test-driven-development, writing-plans, runtime-debugger]
---

# Research and Debug

Investigate before building. Find root cause before fixing. Both skills share the same philosophy: don't guess.

## Overview

Two complementary skills:

- **Spike** (`spike`): Throwaway experiments to validate ideas before committing to a real build
- **Systematic Debugging** (`systematic-debugging`): 4-phase root cause investigation before fixing

**Core principles:**
- Spikes: "Build cheap experiments to answer questions no amount of research can"
- Debugging: "NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST"
## When to Use

### Spike

- "Let me try this", "I want to see if X works", "spike this out"
- Validating feasibility, comparing approaches, surfacing unknowns
- Before committing to a real build

**Don't use when:**
- The answer is knowable from docs — just research
- The work is production path — use `writing-plans` / `planning`
- The idea is already validated — jump straight to implementation

### Systematic Debugging

- Any technical issue: test failures, bugs, unexpected behavior, performance problems
- Especially when under time pressure (emergencies make guessing tempting)
- When you've already tried multiple fixes without success

**Phase 2 checklist — "Before assuming you understand":**
- [ ] All module-level imports present: `functools.lru_cache`, `re`, `pathlib`, `dataclass`
- [ ] Regex patterns compile without error (including parentheses, character classes)
- [ ] No keyword collisions with third-party libraries (`event=` in structlog v25+ reserves this kwarg)
- [ ] Functions actually exist where you think they do (not just in your mental model)
- [ ] Compare import blocks between working files and broken files
- [ ] **Module-level regex errors cascade**: A broken `re.compile()` at module load causes ALL imports of that module (and anything that imports IT) to fail at collection time — look for the root module first
- [ ] **Sibling subagent file state**: When multiple subagents write the same file, your read may see the sibling's version. Always re-read before patching if you know other agents touched the file
- [ ] **Pydantic schema drift**: If a test creates a schema object without a required field that wasn't there before, check the schema definition — fields are added over time. Always include all required fields in test object construction
- [ ] **Return-type mismatches**: Pipeline methods may return tuples (`results, score`) not lists, or `QueryResponse` objects not dicts. Check the actual return statement, not assumptions
**Don't skip when:**
- Issue seems simple (simple bugs have root causes too)
- You're in a hurry (systematic is faster than thrashing)

## Spike Method

### Core Loop

```
decompose  →  research  →  build  →  verdict
   ↑__________________________________________↓
                  iterate on findings
```

### 1. Decompose

Break the idea into **2-5 independent feasibility questions**. Present as table:

| # | Spike | Validates | Risk |
|---|-------|-----------|------|
| 001 | websocket-streaming | Given WS, when LLM streams, client receives chunks < 100ms | High |
| 002a | pdf-parse-pdfjs | Given PDF, when parsed with pdfjs, structured text extractable | Medium |

**Order by risk.** The spike most likely to kill the idea runs first.

### 2. Research (before building)

- Brief it (2-3 sentences)
- Surface competing approaches
- Pick one with justification
- Skip research for pure logic

### 3. Build

One directory per spike. Keep standalone. Bias toward something the user can interact with:

1. Runnable CLI with observable output
2. Minimal HTML page
3. Small web server with one endpoint
4. Unit test with recognizable assertions

**Depth over speed.** Test edge cases. Follow surprises.

### 4. Verdict

Each spike's README closes with:

```markdown
## Verdict: VALIDATED | PARTIAL | INVALIDATED

### What worked
- ...

### What didn't
- ...

### Surprises
- ...

### Recommendation for the real build
- ...
```

## Systematic Debugging Method

### The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

### Phase 1: Root Cause Investigation

**BEFORE attempting ANY fix:**
1. Read error messages carefully
2. Reproduce consistently
3. Check recent changes
4. Gather evidence across component boundaries
5. Trace data flow upstream

**Completion checklist:**
- [ ] Error messages fully read and understood
- [ ] Issue reproduced consistently
- [ ] Recent changes identified
- [ ] Root cause hypothesis formed

### Phase 2: Pattern Analysis

- Find working examples in codebase
- Compare against reference implementations
- Identify differences
- Understand dependencies

### Phase 3: Hypothesis and Testing

- Form single hypothesis: "I think X is root cause because Y"
- Test minimally — one variable at a time
- Verify before continuing
- If no: form new hypothesis

### Phase 4: Implementation

1. Create failing test case (regression test)
2. Implement ONE fix at root cause
3. Verify — run test, run full suite
4. If 3+ fixes failed: question architecture

### Red Flags — STOP

- "Quick fix for now, investigate later"
- "Just try changing X and see if it works"
- "One more fix attempt" (after 2+ failures)
- Each fix reveals new problem in different place

## Quick Reference

| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **Spike** | Decompose, research, build, verdict | Validated/PARTIAL/INVALIDATED |
| **Debug Phase 1** | Read errors, reproduce, trace | Understand WHAT and WHY |
| **Debug Phase 2** | Find working examples, compare | Know what's different |
| **Debug Phase 3** | Form theory, test minimally | Confirmed or new hypothesis |
| **Debug Phase 4** | Regression test, fix root cause | Bug resolved, tests pass |

## Detailed References

See:
- `references/spike.md` — full spike skill
- `references/systematic-debugging.md` — full systematic debugging skill
- `references/wsl-windows-python-structlog.md` — WSL + Windows Python interop, structlog v25 keyword collision, module-level regex cascade, API-key testing patterns