---
name: code-review
description: "Comprehensive code review guidance for PRs — structured four-phase review with severity labels (blocking/important/nit/suggestion/praise). Covers 20+ languages, security, architecture, performance. From awesome-skills/code-review-skill."
version: 1.0.0
author: awesome-skills / Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [code-review, pr-review, quality, security, review]
    related_skills: [requesting-code-review-umbrella]
---

# Code Review Skill

Transform code reviews from gatekeeping to knowledge sharing through constructive feedback, systematic analysis, and collaborative improvement.

## When to Use This Skill

- Reviewing pull requests and code changes
- Establishing code review standards for teams
- Conducting architecture reviews
- Security audits

## Core Principles

### 1. The Review Mindset

**Goals:**
- Catch bugs and edge cases
- Ensure code maintainability
- Share knowledge across team
- Enforce coding standards
- Improve design and architecture

**Not the Goals:**
- Show off knowledge
- Nitpick formatting (use linters)
- Block progress unnecessarily
- Rewrite to your preference

### 2. Effective Feedback

**Good Feedback is:**
- Specific and actionable
- Educational, not judgmental
- Focused on the code, not the person
- Balanced (praise good work too)
- Prioritized (critical vs nice-to-have)

### 3. Severity Labels

| Label | Meaning |
|-------|---------|
| 🔴 blocking | Must fix before merge |
| 🟡 important | Should fix, discuss if disagree |
| 🟢 nit | Nice to have, not blocking |
| 💡 suggestion | Alternative approach |
| 📚 learning | Educational comment |
| 🎉 praise | Good work |

## Review Process

### Phase 1: Context Gathering (2-3 min)
1. Read PR description and linked issue
2. Check PR size (>400 lines? Ask to split)
3. Review CI/CD status
4. Understand the business requirement

### Phase 2: High-Level Review (5-10 min)
1. Architecture & Design — SOLID, coupling/cohesion, anti-patterns
2. Performance — algorithm complexity, N+1 queries
3. File organization — new files in right places
4. Testing strategy — edge cases covered

### Phase 3: Line-by-Line (10-20 min)
For each file:
- Logic & correctness — edge cases, race conditions
- Security — input validation, injection, XSS
- Performance — N+1 queries, unnecessary loops
- Maintainability — clear names, single responsibility

### Phase 4: Summary & Decision (2-3 min)
1. Summarize key concerns
2. Highlight what you liked
3. Clear decision: Approve / Comment / Request Changes

## Review Techniques

**The Checklist Method:** Use checklists for consistent reviews
**The Question Approach:** "What happens if items is empty?" not "This fails if empty"
**Suggest, Don't Command:** "async/await might be more readable" not "You must use async/await"
