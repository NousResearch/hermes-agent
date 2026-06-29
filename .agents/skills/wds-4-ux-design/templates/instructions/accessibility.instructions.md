# Accessibility Specification

**Include when:** Specifying interactive elements, forms, or navigation

---

## For Each Interactive Element

When documenting buttons, links, inputs, add:

```markdown
| Property | Value |
|----------|-------|
| aria-label | "{What it does}" |
| Keyboard | {Enter / Space / Arrow keys} |
| Focus style | {ring / outline / highlight} |
```

**Example:**
```markdown
#### Submit Button
**OBJECT ID:** `form-submit`

| Property | Value |
|----------|-------|
| aria-label | "Submit booking request" |
| Keyboard | Enter or Space |
| Focus | 2px blue ring |
| Disabled state | aria-disabled="true", gray, no focus |
```

---

## Tab Order

Document the logical sequence:

```markdown
## Keyboard Flow

1. `header-logo` → Home link
2. `header-nav` → Main navigation
3. `main-content` → Skip to here
4. `form-name` → First input
5. `form-email` → Second input
6. `form-submit` → Submit button
```

---

## Dynamic Content

When content changes without page reload:

```markdown
| Element | Announces |
|---------|-----------|
| `toast-success` | aria-live="polite" — "Booking confirmed" |
| `error-message` | aria-live="assertive" — Error text |
| `loading-spinner` | aria-busy="true" on parent |
```

---

## Color Independence

For status indicators, ensure alternatives:

| Status | Color | Also Has |
|--------|-------|----------|
| Success | Green | Checkmark icon + "Complete" text |
| Error | Red | Warning icon + error message |
| Active | Blue | Bold text + underline |

---

## Form Errors

Link errors to fields:

```markdown
#### Email Error
**OBJECT ID:** `form-email-error`

| Property | Value |
|----------|-------|
| aria-describedby | Links to `form-email` |
| Role | "alert" |
| Content | "Please enter a valid email" |
```

---

## Quick Checks

Before finalizing:

- [ ] Every button has aria-label or visible text
- [ ] Every image has alt (or alt="" if decorative)
- [ ] Every input has associated label
- [ ] Focus visible on all interactive elements
- [ ] Status not conveyed by color alone
