# Review Output Format

The format for a standalone accessibility review. When `better-interface` orchestrates, it owns the format, severity, consolidation, the cap, and the verdict; hand it domain evidence and findings instead.

Present the standalone review in two parts.

## Findings

Group all confirmed findings by principle. Use a markdown table with **Severity**, **Location**, **Before**, **After**, and **Why** columns. Never use separate "Before:" / "After:" lines.

- **Severity**: `HIGH` prevents a task, hides content from assistive technology, or creates a systemic accessibility failure; `MEDIUM` makes an interaction meaningfully harder; `LOW` is isolated polish.
- **Location**: cite `path/to/file:line`. If the artifact has no source files, cite the exact screen and component instead.
- **Before / After**: show the current implementation and an actionable replacement.
- **Why**: name the violated principle and its user impact.

Consolidate a repeated systemic issue into one row and list every affected location. Omit principles with no findings.

### Example

#### Accessible names everywhere
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| HIGH | `src/Dialog.tsx:42` | `<button><XIcon /></button>` | Add `aria-label="Close"`; mark the icon `aria-hidden="true"` | The icon-only control has no accessible name |
| HIGH | `src/Nav.tsx:18` | `<a href="/settings"><GearIcon /></a>` | Add `aria-label="Settings"` | The link destination is unavailable to screen readers |

#### Visible focus rings
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| HIGH | `src/button.css:12` | `button:focus { outline: none; }` | `button:focus-visible { outline: 2px solid; outline-offset: 2px; }` | Keyboard users cannot see focus |
| HIGH | `src/Menu.tsx:31` | `focus:outline-none` | `focus-visible:outline-2 focus-visible:outline-offset-2` | Menu navigation has no visible focus indicator |

#### Errors that announce
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| HIGH | `src/EmailField.tsx:27` | Error shown only as `border-red-500` | Add `aria-invalid="true"` + `aria-describedby="email-error"` with inline error text | Color alone neither explains nor announces the error |
| MEDIUM | `src/SignupForm.tsx:64` | Submit disabled until the form is valid | Keep submit enabled; on failure, focus the first invalid field | A disabled action hides what must be fixed |

#### Minimum hit area
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| MEDIUM | `src/Toolbar.tsx:22` | `size-4` icon-only button | Extend the hit area to 44×44px with `after:absolute after:size-11` | The target is too small for reliable touch input |

## Verification and Verdict

After the findings:

1. **Verification**: list the exact checks run and their observed results, including keyboard traversal, accessible-name inspection, and screen-reader or automated checks when applicable. If a check was not run, state what still needs verification.
2. **Verdict**: `Block` if any `HIGH` finding remains, `Needs changes` if only `MEDIUM` or `LOW` findings remain, and `Approve` only when no actionable findings remain.

When there are no findings, omit the tables, state "No actionable accessibility findings", report verification, and end with `Approve`.
