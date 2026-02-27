---
name: requesting-code-review
description: Use when completing tasks, implementing major features, or before merging. Validates work meets requirements through systematic review process.
version: 1.0.0
author: Hermes Agent (adapted from Superpowers)
license: MIT
metadata:
  hermes:
    tags: [code-review, quality, validation, workflow, review]
    related_skills: [subagent-driven-development, writing-plans, test-driven-development]
---

# Requesting Code Review

## Overview

Systematic code review catches issues before they cascade. Review early, review often.

**Core principle:** Fresh perspective finds issues you'll miss.

## When to Request Review

**Mandatory Reviews:**
- After each task in subagent-driven development
- After completing major features
- Before merge to main
- After bug fixes

**Optional but Valuable:**
- When stuck (fresh perspective)
- Before refactoring (baseline check)
- After complex logic implementation
- When touching critical code (auth, payments, data)

**Don't skip because:**
- "It's simple" (simple bugs compound)
- "I'm in a hurry" (reviews save time)
- "I tested it" (you have blind spots)

## Review Process

### Step 1: Prepare Context

Gather:
- What was implemented
- Original requirements/plan
- Files changed
- Test results

```bash
# Get changed files
git diff --name-only HEAD~1

# Get diff summary
git diff --stat HEAD~1

# Get commit messages
git log --oneline HEAD~5
```

### Step 2: Self-Review First

Before requesting external review:

**Checklist:**
- [ ] Code follows project conventions
- [ ] All tests pass
- [ ] No debug print statements
- [ ] No hardcoded secrets
- [ ] Error handling in place
- [ ] Documentation updated
- [ ] Commit messages are clear

```bash
# Run full test suite
pytest

# Check for debug code
grep -r "print(" src/ --include="*.py"
grep -r "debugger" src/ --include="*.js"

# Check for TODOs
grep -r "TODO" src/ --include="*.py"
```

### Step 3: Request Review

Use `delegate_task` to dispatch a reviewer subagent:

```python
delegate_task(
    goal="Review implementation for quality and correctness",
    context="""
    WHAT WAS IMPLEMENTED: [Brief description]
    
    ORIGINAL REQUIREMENTS: [From plan or issue]
    
    FILES CHANGED:
    - src/feature.py (added X function)
    - tests/test_feature.py (added tests)
    
    COMMIT RANGE: [SHA range or branch]
    
    CHECK FOR:
    - Correctness (does it do what it should?)
    - Edge cases handled?
    - Error handling adequate?
    - Code quality and style
    - Test coverage
    - Security issues
    - Performance concerns
    
    OUTPUT FORMAT:
    - Summary: [brief assessment]
    - Critical Issues: [must fix]
    - Important Issues: [should fix]
    - Minor Issues: [nice to have]
    - Verdict: [APPROVE / REQUEST_CHANGES / NEEDS_WORK]
    """,
    toolsets=['file']
)
```

### Step 4: Act on Feedback

**Critical Issues (Block merge):**
- Security vulnerabilities
- Broken functionality
- Data loss risk
- Test failures

**Action:** Fix immediately before proceeding

**Important Issues (Should fix):**
- Missing edge case handling
- Poor error messages
- Unclear code
- Missing tests

**Action:** Fix before merge if possible

**Minor Issues (Nice to have):**
- Style preferences
- Refactoring suggestions
- Documentation improvements

**Action:** Note for later or quick fix

## Review Dimensions

### Correctness

**Questions:**
- Does it implement the requirements?
- Are there logic errors?
- Do edge cases work?
- Are there race conditions?

**Check:**
- Read implementation against requirements
- Trace through edge cases
- Check boundary conditions

### Code Quality

**Questions:**
- Is code readable?
- Are names clear?
- Is it too complex?
- Is there duplication?

**Check:**
- Function length (aim <20 lines)
- Cyclomatic complexity
- DRY violations
- Naming clarity

### Testing

**Questions:**
- Are there tests?
- Do they cover edge cases?
- Are they meaningful?
- Do they pass?

**Check:**
- Test coverage
- Edge case coverage
- Test clarity
- Assertion quality

### Security

**Questions:**
- Any injection vulnerabilities?
- Proper input validation?
- Secrets handled correctly?
- Access control in place?

**Check:**
- Input sanitization
- Authentication/authorization
- Secret management
- SQL/query safety

### Performance

**Questions:**
- Any N+1 queries?
- Unnecessary computation?
- Memory leaks?
- Scalability concerns?

**Check:**
- Database queries
- Algorithm complexity
- Resource usage
- Caching opportunities

## Review Output Format

Standard review format:

```markdown
## Review Summary

**Assessment:** [Brief overall assessment]

**Verdict:** [APPROVE / REQUEST_CHANGES / NEEDS_WORK]

---

## Critical Issues (Fix Required)

1. **[Issue title]**
   - Location: `file.py:45`
   - Problem: [Description]
   - Suggestion: [How to fix]

---

## Important Issues (Should Fix)

1. **[Issue title]**
   - Location: `file.py:67`
   - Problem: [Description]
   - Suggestion: [How to fix]

---

## Minor Issues (Optional)

1. **[Issue title]**
   - Suggestion: [Improvement idea]

---

## Strengths

- [What was done well]
```

## Integration with Other Skills

### With subagent-driven-development

Review after EACH task:
1. Subagent implements task
2. Request code review
3. Fix issues
4. Proceed to next task

### With test-driven-development

Review checks:
- Tests written first?
- Tests are meaningful?
- Edge cases covered?
- All tests pass?

### With writing-plans

Review validates:
- Implementation matches plan?
- All tasks completed?
- Quality standards met?

## Common Review Patterns

### Pre-Merge Review

Before merging feature branch:

```bash
# Create review checkpoint
git log --oneline main..feature-branch

# Get summary of changes
git diff --stat main..feature-branch

# Request review
delegate_task(
    goal="Pre-merge review of feature branch",
    context="[changes, requirements, test results]"
)

# Address feedback
# Merge when approved
```

### Continuous Review

During subagent-driven development:

```python
# After each task
if task_complete:
    review_result = request_review(task)
    if review_result.has_critical_issues():
        fix_issues(review_result.critical)
        re_review()
    proceed_to_next_task()
```

### Emergency Review

When fixing production bugs:

1. Fix with tests
2. Self-review
3. Quick peer review
4. Deploy
5. Full review post-deploy

## Review Best Practices

### As Review Requester

**Do:**
- Provide complete context
- Highlight areas of concern
- Ask specific questions
- Be responsive to feedback
- Fix issues promptly

**Don't:**
- Rush the reviewer
- Argue without evidence
- Ignore feedback
- Take criticism personally

### As Reviewer (via subagent)

**Do:**
- Be specific about issues
- Suggest improvements
- Acknowledge what works
- Prioritize issues

**Don't:**
- Nitpick style (unless project requires)
- Make vague comments
- Block without explanation
- Be overly critical

## Quality Gates

**Must pass before merge:**
- [ ] No critical issues
- [ ] All tests pass
- [ ] Review approved
- [ ] Requirements met

**Should pass before merge:**
- [ ] No important issues
- [ ] Documentation updated
- [ ] Performance acceptable

**Nice to have:**
- [ ] No minor issues
- [ ] Extra polish

## Remember

```
Review early
Review often
Be specific
Fix critical issues first
Quality over speed
```

**A good review catches what you missed.**
