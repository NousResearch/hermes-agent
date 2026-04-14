---
name: systematic-debugging
description: Use when encountering any bug, test failure, or unexpected behavior — especially when the user seems frustrated, stuck, or something isn't working despite previous attempts. 4-phase root cause investigation with confidence tagging and explicit backtracking. NO fixes without understanding the problem first.
version: 1.2.0
author: Hermes Agent (adapted from obra/superpowers) + Jeff
license: MIT
metadata:
  hermes:
    tags: [debugging, troubleshooting, problem-solving, root-cause, investigation, frustration, backtracking, confidence]
    activation_signals: [frustrated, stuck, not working, that didn't work, it's broken, still broken, tried that, nothing works, why isn't this working, fix this, this is annoying, i give up, wasting my time]
    related_skills: [test-driven-development, writing-plans, subagent-driven-development]
---

# Systematic Debugging

## Overview

Random fixes waste time and create new bugs. Quick patches mask underlying issues.

**Core principle:** ALWAYS find root cause before attempting fixes. Symptom fixes are failure.

**Violating the letter of this process is violating the spirit of debugging.**

## Frustration Detection — Early Activation

**Activate this skill IMMEDIATELY when you detect user frustration.** Don't wait for the user to explicitly say "I have a bug."

Early warning signs:
- "that didn't work" or "still broken"
- "I tried X and Y"
- "why isn't this working"
- "this is annoying" or "i give up"
- "nevermind" or "forget it"
- User is correcting you repeatedly
- User sounds impatient or short

**When you detect frustration:** Acknowledge it directly and switch to structured debugging immediately. Do not continue guessing. Do not propose another fix without following the process.

Example:
```
User: Still broken. I already tried that.
Agent: I hear you — let's stop guessing and be systematic.
  === DEBUG LOG STARTED ===
  I'll track exactly what we've tried so we don't repeat the same paths.
```

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

If you haven't completed Phase 1, you cannot propose fixes.

## Confidence Levels (Mandatory)

Every claim about the problem must be tagged with a confidence level. Do not use absolute language when uncertain.

| Level | Tag | Meaning | Language |
|-------|-----|---------|----------|
| 🟢 KNOW | `KNOW` | Direct evidence — error message, log output, docs, test failure | "The error says X" |
| 🟡 BELIEVE | `BELIEVE` | Pattern match from experience, similar issues seen before | "In my experience, X usually causes Y" |
| 🔴 GUESS | `GUESS` | Plausible but unproven | "X could be causing this" |

**Rules:**
- Every hypothesis must state its confidence level
- No "try this" without a confidence tag on the preceding theory
- A GUESS followed by another GUESS is not allowed — must verify one first
- Don't say KNOW when you mean BELIEVE or GUESS

## Debugging Log (Running)

Start a log when frustration is detected and maintain it throughout:

```
=== DEBUG LOG: [brief issue description] ===
  Attempt 1: [what was tried] → [KNOW/BELIEVE/GUESS it was wrong] because [specific reason]
  Attempt 2: [what was tried] → [KNOW/BELIEVE/GUESS it was wrong] because [specific reason]
  Dead end reached: [when 2+ fails without clear diagnostic signal]
===
```

Show this log at the start of each response during active debugging.

## When to Use

Use for ANY technical issue:
- Test failures
- Bugs in production
- Unexpected behavior
- Performance problems
- Build failures
- Integration issues

**Use this ESPECIALLY when:**
- Under time pressure (emergencies make guessing tempting)
- "Just one quick fix" seems obvious
- You've already tried multiple fixes
- Previous fix didn't work
- You don't fully understand the issue
- User sounds frustrated

**Don't skip when:**
- Issue seems simple (simple bugs have root causes too)
- You're in a hurry (rushing guarantees rework)
- Someone wants it fixed NOW (systematic is faster than thrashing)

## The Four Phases

You MUST complete each phase before proceeding to the next.

---

## Phase 1: Root Cause Investigation

**BEFORE attempting ANY fix:**

### 1. Read Error Messages Carefully

- Don't skip past errors or warnings
- They often contain the exact solution
- Read stack traces completely
- Note line numbers, file paths, error codes

**Action:** Use `read_file` on the relevant source files. Use `search_files` to find the error string in the codebase.

### 2. Reproduce Consistently

- Can you trigger it reliably?
- What are the exact steps?
- Does it happen every time?
- If not reproducible → gather more data, don't guess

**Action:** Use the `terminal` tool to run the failing test or trigger the bug:

```bash
# Run specific failing test
pytest tests/test_module.py::test_name -v

# Run with verbose output
pytest tests/test_module.py -v --tb=long
```

### 3. Check Recent Changes

- What changed that could cause this?
- Git diff, recent commits
- New dependencies, config changes

**Action:**

```bash
# Recent commits
git log --oneline -10

# Uncommitted changes
git diff

# Changes in specific file
git log -p --follow src/problematic_file.py | head -100
```

### 4. Gather Evidence in Multi-Component Systems

**WHEN system has multiple components (API → service → database, CI → build → deploy):**

**BEFORE proposing fixes, add diagnostic instrumentation:**

For EACH component boundary:
- Log what data enters the component
- Log what data exits the component
- Verify environment/config propagation
- Check state at each layer

Run once to gather evidence showing WHERE it breaks.
THEN analyze evidence to identify the failing component.
THEN investigate that specific component.

### 5. Trace Data Flow

**WHEN error is deep in the call stack:**

- Where does the bad value originate?
- What called this function with the bad value?
- Keep tracing upstream until you find the source
- Fix at the source, not at the symptom

**Action:** Use `search_files` to trace references:

```python
# Find where the function is called
search_files("function_name(", path="src/", file_glob="*.py")

# Find where the variable is set
search_files("variable_name\\s*=", path="src/", file_glob="*.py")
```

### Phase 1 Completion Checklist

- [ ] Error messages fully read and understood
- [ ] Issue reproduced consistently
- [ ] Recent changes identified and reviewed
- [ ] Evidence gathered (logs, state, data flow)
- [ ] Problem isolated to specific component/code
- [ ] Root cause hypothesis formed

**STOP:** Do not proceed to Phase 2 until you understand WHY it's happening.

---

## Phase 2: Pattern Analysis

**Find the pattern before fixing:**

### 1. Find Working Examples

- Locate similar working code in the same codebase
- What works that's similar to what's broken?

**Action:** Use `search_files` to find comparable patterns:

```python
search_files("similar_pattern", path="src/", file_glob="*.py")
```

### 2. Compare Against References

- If implementing a pattern, read the reference implementation COMPLETELY
- Don't skim — read every line
- Understand the pattern fully before applying

### 3. Identify Differences

- What's different between working and broken?
- List every difference, however small
- Don't assume "that can't matter"

### 4. Understand Dependencies

- What other components does this need?
- What settings, config, environment?
- What assumptions does it make?

---

## Phase 3: Hypothesis and Testing

**Scientific method with confidence tagging:**

### 1. Form a Tagged Hypothesis

State clearly with confidence level:

```
=== HYPOTHESIS CARD ===
Hypothesis: [What you think is happening]
Confidence: [KNOW | BELIEVE | GUESS]
Evidence: [What you actually know, not what you assume]
Verification: [Specific thing to check — command, log, file content]
If Confirmed: [What this means for the fix]
If Ruled Out: [What we know is NOT the problem]
===
```

### 2. Test Minimally

- Make the SMALLEST possible change to test the hypothesis
- One variable at a time
- Don't fix multiple things at once

### 3. Backtrack on Failure — Explicit Protocol

When a fix fails, STOP and output before suggesting ANYTHING else:

```
=== BACKTRACK ===
Fix Attempt: [what was tried]
Why I Thought It Would Work: [specific reason]
Why It Failed: [specific failure mode observed — not just "didn't work"]
What This Rules Out: [what we now know is NOT the problem]
What This Doesn't Rule Out: [what could still be true]
New Confidence: [updated based on what we learned]
===

What I now know for certain:
1. [Fact learned from the failure — be specific]
2. [Fact learned from the failure — be specific]

I will not suggest another fix until I verify [specific thing].
```

**The backtrack is mandatory after any failed fix attempt.** Do not propose the next fix without the backtrack first.

### 4. When You Don't Know

- Say "I don't understand X"
- Don't pretend to know
- Ask the user for help
- Research more

---

## Phase 4: Implementation

**Fix the root cause, not the symptom:**

### 1. Create Failing Test Case

- Simplest possible reproduction
- Automated test if possible
- MUST have before fixing
- Use the `test-driven-development` skill

### 2. Implement Single Fix

- Address the root cause identified
- ONE change at a time
- No "while I'm here" improvements
- No bundled refactoring

### 3. Verify Fix

```bash
# Run the specific regression test
pytest tests/test_module.py::test_regression -v

# Run full suite — no regressions
pytest tests/ -q
```

### 4. If Fix Doesn't Work — The Rule of Three

- **STOP.**
- Count: How many fixes have you tried?
- If < 3: Return to Phase 1, re-analyze with new information
- **If ≥ 3: STOP and question the architecture (step 5 below)**
- DON'T attempt Fix #4 without architectural discussion

### 5. If 3+ Fixes Failed: Question Architecture

**Pattern indicating an architectural problem:**
- Each fix reveals new shared state/coupling in a different place
- Fixes require "massive refactoring" to implement
- Each fix creates new symptoms elsewhere

**STOP and question fundamentals:**
- Is this pattern fundamentally sound?
- Are we "sticking with it through sheer inertia"?
- Should we refactor the architecture vs. continue fixing symptoms?

**Discuss with the user before attempting more fixes.**

This is NOT a failed hypothesis — this is a wrong architecture.

---

## Red Flags — STOP and Follow Process

If you catch yourself thinking:
- "Quick fix for now, investigate later"
- "Just try changing X and see if it works"
- "Add multiple changes, run tests"
- "Skip the test, I'll manually verify"
- "It's probably X, let me fix that"
- "I don't fully understand but this might work"
- "Pattern says X but I'll adapt it differently"
- "Here are the main problems: [lists fixes without investigation]"
- Proposing solutions before tracing data flow
- **"One more fix attempt" (when already tried 2+)**
- **Each fix reveals a new problem in a different place**

**ALL of these mean: STOP. Return to Phase 1.**

**If 3+ fixes failed:** Question the architecture (Phase 4 step 5).

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Issue is simple, don't need process" | Simple issues have root causes too. Process is fast for simple bugs. |
| "Emergency, no time for process" | Systematic debugging is FASTER than guess-and-check thrashing. |
| "Just try this first, then investigate" | First fix sets the pattern. Do it right from the start. |
| "I'll write test after confirming fix works" | Untested fixes don't stick. Test first proves it. |
| "Multiple fixes at once saves time" | Can't isolate what worked. Causes new bugs. |
| "Reference too long, I'll adapt the pattern" | Partial understanding guarantees bugs. Read it completely. |
| "I see the problem, let me fix it" | Seeing symptoms ≠ understanding root cause. |
| "One more fix attempt" (after 2+ failures) | 3+ failures = architectural problem. Question the pattern, don't fix again. |

## Quick Reference

| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Root Cause** | Read errors, reproduce, check changes, gather evidence, trace data flow | Understand WHAT and WHY |
| **2. Pattern** | Find working examples, compare, identify differences | Know what's different |
| **3. Hypothesis** | Form theory with confidence tag, test minimally, backtrack explicitly | Confirmed or new hypothesis |
| **4. Implementation** | Create regression test, fix root cause, verify | Bug resolved, all tests pass |

## Confidence Level Quick Tags

| Tag | Use when |
|-----|---------|
| `KNOW` | Error message, stack trace, log output, direct observation |
| `BELIEVE` | Pattern you've seen before, similar issues, likely cause |
| `GUESS` | Plausible but unverified, multiple possibilities |

## Hermes Agent Integration

### Investigation Tools

Use these Hermes tools during Phase 1:

- **`search_files`** — Find error strings, trace function calls, locate patterns
- **`read_file`** — Read source code with line numbers for precise analysis
- **`terminal`** — Run tests, check git history, reproduce bugs
- **`web_search`/`web_extract`** — Research error messages, library docs

### With delegate_task

For complex multi-component debugging, dispatch investigation subagents:

```python
delegate_task(
    goal="Investigate why [specific test/behavior] fails",
    context="""
    Follow systematic-debugging skill:
    1. Read the error message carefully
    2. Reproduce the issue
    3. Trace the data flow to find root cause
    4. Report findings — do NOT fix yet

    Error: [paste full error]
    File: [path to failing code]
    Test command: [exact command]
    """,
    toolsets=['terminal', 'file']
)
```

### With test-driven-development

When fixing bugs:
1. Write a test that reproduces the bug (RED)
2. Debug systematically to find root cause
3. Fix the root cause (GREEN)
4. The test proves the fix and prevents regression

## Real-World Impact

From debugging sessions:
- Systematic approach: 15-30 minutes to fix
- Random fixes approach: 2-3 hours of thrashing
- First-time fix rate: 95% vs 40%
- New bugs introduced: Near zero vs common

**No shortcuts. No guessing. Systematic always wins.**
