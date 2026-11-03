# Accessibility Rule IDs and the Fix Partition

The finding vocabulary. Prose findings are not comparable across reruns and cannot be handed off in
parts; a rule ID, a severity, a WCAG criterion and a fix class can.

## How a finding is written

Read the target surface completely, collect every finding, then report. One row per finding:

`rule ID | severity | location | WCAG criterion | fix class | the fix`

Severity is `critical` (the surface is unusable for someone), `major` (a task is degraded or a
control is mislabelled to assistive technology) or `minor` (comprehension or convention). Severity
ranks findings; the audit verdict stays PASS/HOLD/BLOCK on observed evidence.

## The fix partition

A fix is `auto` **only when the correct output is derivable from the markup itself**. It is `manual`
whenever producing it requires knowing what the content *means*.

- Derivable: an image inside a control that already carries a text label is decorative, so its text
  alternative is the empty one. The structure decided it.
- Not derivable: what a meaningful image depicts, where a link goes, what a field asks for.

When a fix is partly structural and partly semantic, split the row into its `auto` half and its
`manual` half rather than rounding to either.

## Rules

### Images

| ID | Finding | Severity | WCAG | Fix class |
| --- | --- | --- | --- | --- |
| IMG-1 | Image with no text alternative | critical | 1.1.1 | `auto` when decorative by context, else `manual` |
| IMG-2 | Decorative image not marked decorative to assistive technology | minor | 1.1.1 | `auto` |

### Links and buttons

| ID | Finding | Severity | WCAG | Fix class |
| --- | --- | --- | --- | --- |
| LNK-1 | Link with no accessible name | critical | 2.4.4, 4.1.2 | `manual` |
| LNK-2 | Link text that does not describe its destination | minor | 2.4.4 | `manual` |
| LNK-3 | Link opening a new context with no warning | minor | 3.2.5 | `manual` |
| BTN-1 | Button with no accessible name | critical | 4.1.2 | `manual` |

### Forms

| ID | Finding | Severity | WCAG | Fix class |
| --- | --- | --- | --- | --- |
| FORM-1 | Control with no programmatically associated label | critical | 1.3.1, 3.3.2, 4.1.2 | `manual` |
| FORM-2 | Label associated with a control that does not exist | major | 1.3.1 | `manual` |
| FORM-3 | Personal-data field with no autofill purpose declared | minor | 1.3.5 | `manual` |
| FORM-4 | Error not announced or not associated with its field | major | 3.3.1, 4.1.3 | `manual` |

### Roles and states

| ID | Finding | Severity | WCAG | Fix class |
| --- | --- | --- | --- | --- |
| ARIA-1 | Role that is not a valid role name | major | 4.1.2 | `manual` |
| ARIA-2 | Focusable element hidden from assistive technology | critical | 2.4.3, 4.1.2 | `auto` (unhide; it is keyboard-reachable and must be announced) |
| ARIA-3 | Decorative vector graphic exposed to assistive technology | major | 1.1.1 | `auto` when decorative by context, else `manual` |
| ARIA-4 | Status update outside a live region | major | 4.1.3 | `manual` |

### Keyboard and focus

| ID | Finding | Severity | WCAG | Fix class |
| --- | --- | --- | --- | --- |
| KEY-1 | Positive tab index overriding document order | major | 2.4.3 | `auto` (neutralize), then re-walk the order |
| KEY-2 | Pointer handler on a non-interactive element with no keyboard path | major | 2.1.1 | split: role and focusability `auto`, the key handler `manual` |
| KEY-3 | Keyboard trap | critical | 2.1.2 | `manual` |
| KEY-4 | Focus indicator not visible | major | 2.4.7 | `manual` |
| KEY-5 | Focused element hidden behind sticky content | major | 2.4.11 | `manual` |

### Structure

| ID | Finding | Severity | WCAG | Fix class |
| --- | --- | --- | --- | --- |
| SEM-1 | Page language not declared | critical | 3.1.1 | `auto` (the attribute), with the language value verified |
| SEM-2 | Heading level skipped | major | 1.3.1 | `manual` |
| SEM-3 | No main landmark | major | 1.3.1, 2.4.1 | `manual` |
| SEM-4 | Embedded frame with no title | major | 2.4.1, 4.1.2 | `manual` |
| SEM-5 | Layout table exposed as a data table | minor | 1.3.1 | `auto` |

### Color, size and reflow

| ID | Finding | Severity | WCAG | Fix class |
| --- | --- | --- | --- | --- |
| COL-1 | Color as the only carrier of meaning | minor | 1.4.1 | `manual` |
| COL-2 | Text contrast below 4.5:1 (3:1 large text) | major | 1.4.3 | `manual` (measure, then pick the color) |
| COL-3 | UI component or focus indicator contrast below 3:1 | major | 1.4.11 | `manual` |
| SIZE-1 | Pointer target under 24×24 CSS px without spacing exception | major | 2.5.8 | `manual` |
| FLOW-1 | Content clipped or two-dimensional scroll at 320 CSS px | major | 1.4.10 | `manual` |

Detection criteria are written against structure, not one framework's syntax. A syntax that hides
the condition (a fully dynamic binding, a name computed at runtime) is reported as
unverifiable-from-source and checked in the live accessibility tree.

## Rerun and handoff

A finding is resolved when its ID no longer matches at that location, carried when it still does,
and new otherwise. Only `auto` rows can be handed off as a batch; `manual` rows go back with the
question each one needs answered. A fix class is a property of the fix, never evidence it was
applied — the verdict still needs observed evidence after the change.
