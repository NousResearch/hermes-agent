# Accessibility Audit Workflow

**Purpose:** Agent-led accessibility review with explanations and suggestions

---

## How This Works

1. Agent reads the page specification and/or prototype code
2. Agent evaluates each area against WCAG 2.1 AA
3. Agent explains findings in plain language
4. Agent proposes specific fixes
5. User approves, rejects, or asks for alternatives

---

## Agent Instructions

### Step 1: Analyze Color Contrast

Read the design system colors and check:

```
For each text element:
- Calculate contrast ratio against background
- WCAG AA requires: 4.5:1 for normal text, 3:1 for large text (18px+)

Report:
"The button text (#FFFFFF) on primary background (#2563EB)
has a contrast ratio of 8.6:1 ✓ Passes WCAG AA

The helper text (#9CA3AF) on white (#FFFFFF)
has a contrast ratio of 2.9:1 ✗ Fails WCAG AA
→ Suggestion: Darken to #6B7280 (4.6:1) or #4B5563 (7:1)"
```

**Ask user:** "Should I darken the helper text to #6B7280?"

---

### Step 2: Analyze Keyboard Navigation

Read the page structure and determine logical tab order:

```
Looking at the page layout, the logical keyboard flow should be:

1. Skip to main content link (hidden until focused)
2. Logo (if clickable) → Home
3. Navigation items left to right
4. Main content, top to bottom
5. Form fields in visual order
6. Submit/action buttons
7. Footer links

Current spec has these interactive elements:
- `header-logo` ✓ Has behavior defined
- `nav-home` ✓ Link
- `form-email` ✗ Missing keyboard info
- `submit-btn` ✗ Missing focus state

→ Suggestion: Add to form-email:
  | Keyboard | Tab to focus, type to enter |
  | Focus | 2px blue ring (#2563EB) |
```

**Ask user:** "Should I add keyboard properties to these elements?"

---

### Step 3: Analyze Screen Reader Experience

Check each element has appropriate labels:

```
Reviewing interactive elements:

`booking-submit` - Button with text "Book"
→ Issue: "Book" alone may be unclear out of context
→ Suggestion: aria-label="Book this walk slot"

`booking-cancel` - Icon button with X
→ Issue: No text, screen reader says nothing
→ Suggestion: aria-label="Cancel booking"

`status-indicator` - Colored dot (green/red/gray)
→ Issue: Color only, no text alternative
→ Suggestion: Add aria-label="Status: Available" and visually hidden text
```

**Ask user:** "Should I add these aria-labels to the spec?"

---

### Step 4: Analyze Dynamic Content

Check state changes and notifications:

```
This page has dynamic content:

State changes (drawer opens/closes):
→ Need: aria-expanded on trigger, focus management
→ Suggestion: When drawer opens, move focus to drawer header

Loading states:
→ Need: aria-busy="true" on container, "Loading..." announcement
→ Suggestion: Add aria-live="polite" region for status updates

Error messages:
→ Need: aria-live="assertive" so errors are announced immediately
→ Suggestion: Link error to field with aria-describedby
```

**Ask user:** "Should I add these dynamic content specifications?"

---

### Step 5: Summary Report

```
## Accessibility Audit Summary

### Passes ✓
- Color contrast on primary buttons (8.6:1)
- Semantic HTML structure (header, main, nav)
- Form labels present

### Needs Attention ⚠
- Helper text contrast (2.9:1 → needs 4.5:1)
- 3 buttons missing aria-labels
- Tab order not documented
- Focus states not specified

### Recommendations
1. Darken helper text to #6B7280
2. Add aria-labels to icon buttons
3. Document keyboard flow
4. Specify focus ring style (2px #2563EB)

Implement these changes? [Yes to all / Review each / Skip]
```

---

## Quick Reference for Agent

| Check | WCAG Rule | Requirement |
|-------|-----------|-------------|
| Text contrast | 1.4.3 | 4.5:1 normal, 3:1 large |
| Focus visible | 2.4.7 | Clear visual indicator |
| Labels | 1.3.1 | All inputs labeled |
| Keyboard | 2.1.1 | All functions keyboard accessible |
| Error ID | 3.3.1 | Errors identified and described |
| Name/Role | 4.1.2 | Interactive elements have accessible names |

---

## Agent Prompts

Use these to guide the conversation:

- "I found {N} contrast issues. Want me to explain each one?"
- "This button has no accessible name. Should I suggest one based on its purpose?"
- "The tab order seems unclear. Can you confirm the intended flow?"
- "Screen readers won't announce this status change. Should I add aria-live?"
