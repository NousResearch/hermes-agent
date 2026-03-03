---
name: subagent-driven-development
description: Use when executing implementation plans with independent tasks. Dispatches fresh delegate_task per task with two-stage review (spec compliance then code quality).
version: 1.0.0
author: Hermes Agent (adapted from Superpowers)
license: MIT
metadata:
  hermes:
    tags: [delegation, subagent, implementation, workflow, parallel]
    related_skills: [writing-plans, requesting-code-review, test-driven-development]
---

# Subagent-Driven Development

## Overview

Execute implementation plans by dispatching fresh subagents per task with systematic two-stage review.

**Core principle:** Fresh subagent per task + two-stage review (spec then quality) = high quality, fast iteration

## When to Use

Use this skill when:
- You have an implementation plan (from writing-plans skill)
- Tasks are mostly independent
- You want to stay in the current session
- Quality and spec compliance are important

**vs. Manual execution:**
- Parallel task execution possible
- Automated review process
- Consistent quality checks
- Better for complex multi-step plans

## The Process

### 1. Read and Parse Plan

```markdown
[Read plan file once: docs/plans/feature-plan.md]
[Extract all tasks with full text and context]
[Create todo list with all tasks]
```

**Action:** Read plan, extract tasks, create todo list.

### 2. Per-Task Workflow

For EACH task in the plan:

#### Step 1: Dispatch Implementer Subagent

Use `delegate_task` with:
- **goal:** Implement [specific task from plan]
- **context:** Full task description from plan, project structure, relevant files
- **toolsets:** ['terminal', 'file', 'web'] (or as needed)

**Example:**
```python
# Task: Add user authentication middleware
delegate_task(
    goal="Implement JWT authentication middleware as specified in Task 3 of the plan",
    context="""
    Task from plan:
    - Create: src/middleware/auth.py
    - Validate JWT tokens from Authorization header
    - Return 401 for invalid tokens
    - Attach user info to request object
    
    Project structure:
    - Flask app in src/app.py
    - Uses PyJWT library
    - Existing middleware pattern in src/middleware/
    """,
    toolsets=['terminal', 'file']
)
```

#### Step 2: Implementer Subagent Works

The subagent will:
1. Ask questions if needed (you answer)
2. Implement the task following TDD
3. Write tests
4. Run tests to verify
5. Self-review
6. Report completion

**Your role:** Answer questions, provide context.

#### Step 3: Spec Compliance Review

Dispatch reviewer subagent:

```python
delegate_task(
    goal="Review if implementation matches spec from plan",
    context="""
    Original task spec: [copy from plan]
    Implementation: [file paths and key code]
    
    Check:
    - All requirements from spec implemented?
    - File paths match spec?
    - Behavior matches spec?
    - Nothing extra added?
    """,
    toolsets=['file']
)
```

**If spec issues found:**
- Subagent fixes gaps
- Re-run spec review
- Continue only when spec-compliant

#### Step 4: Code Quality Review

Dispatch quality reviewer:

```python
delegate_task(
    goal="Review code quality and best practices",
    context="""
    Code to review: [file paths]
    
    Check:
    - Follows project style?
    - Proper error handling?
    - Good naming?
    - Test coverage adequate?
    - No obvious bugs?
    """,
    toolsets=['file']
)
```

**If quality issues found:**
- Subagent fixes issues
- Re-run quality review
- Continue only when approved

#### Step 5: Mark Complete

Update todo list, mark task complete.

### 3. Final Review

After ALL tasks complete:

```python
delegate_task(
    goal="Review entire implementation for consistency",
    context="All tasks completed, review for integration issues",
    toolsets=['file']
)
```

### 4. Branch Cleanup

Use `finishing-a-development-branch` skill:
- Verify all tests pass
- Present merge options
- Clean up worktree

## Task Granularity

**Good task size:** 2-5 minutes of focused work

**Examples:**

**Too big:**
- "Implement user authentication system"

**Right size:**
- "Create User model with email and password fields"
- "Add password hashing function"
- "Create login endpoint"
- "Add JWT token generation"

## Communication Pattern

### You to Subagent

**Provide:**
- Clear task description
- Exact file paths
- Expected behavior
- Success criteria
- Relevant context

**Example:**
```
Task: Add email validation
Files: Create src/validators/email.py
Expected: Function returns True for valid emails, False for invalid
Success: Tests pass for 10 test cases including edge cases
Context: Used in user registration flow
```

### Subagent to You

**Expect:**
- Questions for clarification
- Progress updates
- Completion report
- Self-review summary

**Respond to:**
- Answer questions promptly
- Provide missing context
- Approve approach decisions

## Two-Stage Review Details

### Stage 1: Spec Compliance

**Checks:**
- [ ] All requirements from plan implemented
- [ ] File paths match specification
- [ ] Function signatures match spec
- [ ] Behavior matches expected
- [ ] No scope creep (nothing extra)

**Output:** PASS or list of spec gaps

### Stage 2: Code Quality

**Checks:**
- [ ] Follows language conventions
- [ ] Consistent with project style
- [ ] Clear variable/function names
- [ ] Proper error handling
- [ ] Adequate test coverage
- [ ] No obvious bugs/edge cases missed
- [ ] Documentation if needed

**Output:** APPROVED or list of issues (critical/important/minor)

## Handling Issues

### Critical Issues

**Examples:** Security vulnerability, broken functionality, data loss risk

**Action:** Must fix before proceeding

### Important Issues

**Examples:** Missing tests, poor error handling, unclear code

**Action:** Should fix before proceeding

### Minor Issues

**Examples:** Style inconsistency, minor refactoring opportunity

**Action:** Note for later, optional fix

## Integration with Other Skills

### With test-driven-development

Subagent should:
1. Write failing test first
2. Implement minimal code
3. Verify test passes
4. Commit

### With systematic-debugging

If subagent encounters bugs:
1. Pause implementation
2. Debug systematically
3. Fix root cause
4. Resume

### With writing-plans

This skill EXECUTES plans created by writing-plans skill.

**Sequence:**
1. brainstorming → writing-plans → subagent-driven-development

### With requesting-code-review

After subagent completes task, use requesting-code-review skill for final validation.

## Common Patterns

### Pattern: Fresh Subagent Per Task

**Why:** Prevents context pollution
**How:** New delegate_task for each task
**Result:** Each subagent has clean context

### Pattern: Two-Stage Review

**Why:** Catch issues early, ensure quality
**How:** Spec review → Quality review
**Result:** High-quality, spec-compliant code

### Pattern: Frequent Checkpoints

**Why:** Catch issues before they compound
**How:** Review after each task
**Result:** Issues don't cascade

## Best Practices

1. **Clear Task Boundaries**
   - One task = one focused change
   - Independent where possible
   - Clear success criteria

2. **Complete Context**
   - Provide all needed files
   - Explain project conventions
   - Share relevant examples

3. **Review Discipline**
   - Don't skip spec review
   - Address critical issues immediately
   - Keep quality bar consistent

4. **Communication**
   - Answer subagent questions quickly
   - Clarify when needed
   - Provide feedback on reviews

## Example Workflow

```markdown
User: Implement user authentication

You: I'll use subagent-driven development. Let me create a plan first.
[Uses writing-plans skill]

Plan created with 5 tasks:
1. Create User model
2. Add password hashing
3. Implement login endpoint
4. Add JWT middleware
5. Create registration endpoint

--- Task 1 ---
[Dispatch implementer subagent for User model]
[Subagent asks: "Should email be unique?"]
You: Yes, email must be unique
[Subagent implements]
[Dispatch spec reviewer - PASS]
[Dispatch quality reviewer - APPROVED]
Task 1 complete

--- Task 2 ---
[Dispatch implementer for password hashing]
...

[After all tasks]
[Final review]
[Merge branch]
```

## Remember

```
Fresh subagent per task
Two-stage review every time
Spec compliance first
Code quality second
Never skip reviews
Catch issues early
```

**Quality is not an accident. It's the result of systematic process.**
