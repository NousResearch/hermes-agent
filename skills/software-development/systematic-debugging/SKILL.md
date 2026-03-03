---
name: systematic-debugging
description: Use when encountering any bug, test failure, or unexpected behavior. 4-phase root cause investigation process - NO fixes without understanding the problem first.
version: 1.0.0
author: Hermes Agent (adapted from Superpowers)
license: MIT
metadata:
  hermes:
    tags: [debugging, troubleshooting, problem-solving, root-cause, investigation]
    related_skills: [test-driven-development, writing-plans, subagent-driven-development]
---

# Systematic Debugging

## Overview

Random fixes waste time and create new bugs. Quick patches mask underlying issues.

**Core principle:** ALWAYS find root cause before attempting fixes. Symptom fixes are failure.

**Violating the letter of this process is violating the spirit of debugging.**

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

If you haven't completed Phase 1, you cannot propose fixes.

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

**Don't skip when:**
- Issue seems simple (simple bugs have root causes too)
- You're in a hurry (rushing guarantees rework)
- Manager wants it fixed NOW (systematic is faster than thrashing)

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

**Action:** Copy full error message to your notes.

### 2. Reproduce Consistently

- Can you trigger it reliably?
- What are the exact steps?
- Does it happen every time?
- If not reproducible → gather more data, don't guess

**Action:** Write down exact reproduction steps.

### 3. Check Recent Changes

- What changed that could cause this?
- Git diff, recent commits
- New dependencies, config changes
- Environmental differences

**Commands:**
```bash
# Recent commits
git log --oneline -10

# Uncommitted changes
git diff

# Changes in specific file
git log -p --follow src/problematic_file.py
```

### 4. Gather Evidence in Multi-Component Systems

**WHEN system has multiple components (CI pipeline, API service, database layer):**

**BEFORE proposing fixes, add diagnostic instrumentation:**

For EACH component boundary:
- Log what data enters component
- Log what data exits component
- Verify environment/config propagation
- Check state at each layer

Run once to gather evidence showing WHERE it breaks
THEN analyze evidence to identify failing component
THEN investigate that specific component

**Example (multi-layer system):**
```python
# Layer 1: Entry point
def entry_point(input_data):
    print(f"DEBUG: Input received: {input_data}")
    result = process_layer1(input_data)
    print(f"DEBUG: Layer 1 output: {result}")
    return result

# Layer 2: Processing
def process_layer1(data):
    print(f"DEBUG: Layer 1 received: {data}")
    # ... processing ...
    print(f"DEBUG: Layer 1 returning: {result}")
    return result
```

**Action:** Add logging, run once, analyze output.

### 5. Isolate the Problem

- Comment out code until problem disappears
- Binary search through recent changes
- Create minimal reproduction case
- Test with fresh environment

**Action:** Create minimal reproduction case.

### Phase 1 Completion Checklist

- [ ] Error messages fully read and understood
- [ ] Issue reproduced consistently
- [ ] Recent changes identified and reviewed
- [ ] Evidence gathered (logs, state)
- [ ] Problem isolated to specific component/code
- [ ] Root cause hypothesis formed

**STOP:** Do not proceed to Phase 2 until you understand WHY it's happening.

---

## Phase 2: Solution Design

**Given the root cause, design the fix:**

### 1. Understand the Fix Area

- Read relevant code thoroughly
- Understand data flow
- Identify affected components
- Check for similar issues elsewhere

**Action:** Read all relevant code files.

### 2. Design Minimal Fix

- Smallest change that fixes root cause
- Avoid scope creep
- Don't refactor while fixing
- Fix one issue at a time

**Action:** Write down the exact fix before coding.

### 3. Consider Side Effects

- What else could this change affect?
- Are there dependencies?
- Will this break other functionality?

**Action:** Identify potential side effects.

### Phase 2 Completion Checklist

- [ ] Fix area code fully understood
- [ ] Minimal fix designed
- [ ] Side effects identified
- [ ] Fix approach documented

---

## Phase 3: Implementation

**Make the fix:**

### 1. Write Test First (if possible)

```python
def test_should_handle_empty_input():
    """Regression test for bug #123"""
    result = process_data("")
    assert result == expected_empty_result
```

### 2. Implement Fix

```python
# Before (buggy)
def process_data(data):
    return data.split(",")[0]

# After (fixed)
def process_data(data):
    if not data:
        return ""
    return data.split(",")[0]
```

### 3. Verify Fix

```bash
# Run the specific test
pytest tests/test_data.py::test_should_handle_empty_input -v

# Run all tests to check for regressions
pytest
```

### Phase 3 Completion Checklist

- [ ] Test written that reproduces the bug
- [ ] Minimal fix implemented
- [ ] Test passes
- [ ] No regressions introduced

---

## Phase 4: Verification

**Confirm it's actually fixed:**

### 1. Reproduce Original Issue

- Follow original reproduction steps
- Verify issue is resolved
- Test edge cases

### 2. Regression Testing

```bash
# Full test suite
pytest

# Integration tests
pytest tests/integration/

# Check related areas
pytest -k "related_feature"
```

### 3. Monitor After Deploy

- Watch logs for related errors
- Check metrics
- Verify fix in production

### Phase 4 Completion Checklist

- [ ] Original issue cannot be reproduced
- [ ] All tests pass
- [ ] No new warnings/errors
- [ ] Fix documented (commit message, comments)

---

## Debugging Techniques

### Root Cause Tracing

Ask "why" 5 times:
1. Why did it fail? → Null pointer
2. Why was it null? → Function returned null
3. Why did function return null? → Missing validation
4. Why was validation missing? → Assumed input always valid
5. Why was that assumption wrong? → API changed

**Root cause:** API change not accounted for

### Defense in Depth

Don't fix just the symptom:

**Bad:** Add null check at crash site
**Good:** 
1. Add validation at API boundary
2. Add null check at crash site
3. Add test for both
4. Document API behavior

### Condition-Based Waiting

For timing/race conditions:

```python
# Bad - arbitrary sleep
import time
time.sleep(5)  # "Should be enough"

# Good - wait for condition
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(5))
def wait_for_service():
    response = requests.get(health_url)
    assert response.status_code == 200
```

---

## Common Debugging Pitfalls

### Fix Without Understanding

**Symptom:** "Just add a try/catch"
**Problem:** Masks the real issue
**Solution:** Complete Phase 1 before any fix

### Shotgun Debugging

**Symptom:** Change 5 things at once
**Problem:** Don't know what fixed it
**Solution:** One change at a time, verify each

### Premature Optimization

**Symptom:** Rewrite while debugging
**Problem:** Introduces new bugs
**Solution:** Fix first, refactor later

### Assuming Environment

**Symptom:** "Works on my machine"
**Problem:** Environment differences
**Solution:** Check environment variables, versions, configs

---

## Language-Specific Tools

### Python

```python
# Add debugger
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()

# Log state
import logging
logging.debug(f"Variable state: {variable}")

# Stack trace
import traceback
traceback.print_exc()
```

### JavaScript/TypeScript

```javascript
// Debugger
debugger;

// Console with context
console.log("State:", { var1, var2, var3 });

// Stack trace
console.trace("Here");

// Error with context
throw new Error(`Failed with input: ${JSON.stringify(input)}`);
```

### Go

```go
// Print state
fmt.Printf("Debug: variable=%+v\n", variable)

// Stack trace
import "runtime/debug"
debug.PrintStack()

// Panic with context
if err != nil {
    panic(fmt.Sprintf("unexpected error: %v", err))
}
```

---

## Integration with Other Skills

### With test-driven-development

When debugging:
1. Write test that reproduces bug
2. Debug systematically
3. Fix root cause
4. Test passes

### With writing-plans

Include debugging tasks in plans:
- "Add diagnostic logging"
- "Create reproduction test"
- "Verify fix resolves issue"

### With subagent-driven-development

If subagent gets stuck:
1. Switch to systematic debugging
2. Analyze root cause
3. Provide findings to subagent
4. Resume implementation

---

## Remember

```
PHASE 1: Investigate → Understand WHY
PHASE 2: Design → Plan the fix
PHASE 3: Implement → Make the fix
PHASE 4: Verify → Confirm it's fixed
```

**No shortcuts. No guessing. Systematic always wins.**
