# Code Review Output Template

Use this template when posting a code review summary on a GitLab Merge Request.

---

## Code Review Summary

**Verdict: [Approved / Changes Requested / Comment]**

### 🔴 Critical
- **file/path:line** — Description of the critical issue.
  Suggestion: How to fix it.

### ⚠️ Warnings
- **file/path:line** — Description of the warning.
  Suggestion: How to fix it.

### 💡 Suggestions
- **file/path:line** — Description of the suggestion.
  Rationale: Why this improvement helps.

### ✅ Looks Good
- Positive observations about the code
- Well-designed patterns worth highlighting
- Good test coverage in specific areas

---
*Reviewed by Hermes Agent*

---

## Inline Comment Format

Use these prefixes for inline comments:

- 🔴 **Critical:** — Must fix before merge (security, correctness, data loss)
- ⚠️ **Warning:** — Should fix before merge (potential bugs, missing error handling)
- 💡 **Suggestion:** — Nice to have (style, refactoring, minor improvements)
- ✅ **Nice:** — Positive feedback on good patterns

## Decision Guide

| Finding Level | Review Action |
|---------------|--------------|
| No critical/warning issues | **Approve** |
| Any critical issue | **Request Changes** |
| Any warning issue | **Request Changes** (unless author acknowledges and will fix separately) |
| Only suggestions | **Comment** or **Approve** with suggestions |
