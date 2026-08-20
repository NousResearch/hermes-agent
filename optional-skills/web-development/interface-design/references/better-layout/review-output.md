# Review Output Format

The format for a standalone layout review. When `better-interface` orchestrates, it owns the format, severity, consolidation, the cap, and the verdict; hand it domain evidence and findings instead.

Present the standalone review in two parts.

## Findings

Group all confirmed findings by principle. Use a markdown table with **Severity**, **Location**, **Before**, **After**, and **Why** columns. Never use separate "Before:" / "After:" lines.

- **Severity**: `HIGH` blocks content or an action at a supported viewport; `MEDIUM` harms hierarchy, reading order, or adaptability; `LOW` is isolated alignment or spacing polish.
- **Location**: cite `path/to/file:line`. If the artifact has no source files, cite the exact screen and component instead.
- **Before / After**: show the current layout and an actionable replacement.
- **Why**: name the violated principle and its effect on comprehension or adaptability.

Consolidate a repeated systemic issue into one row and list every affected location. Omit principles with no findings.

### Example

#### Group with space, not lines
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| LOW | `src/Settings.tsx:41` | `border-b` on every settings row | Remove borders; use `space-y-2` within groups and `space-y-8` between groups | Spacing communicates grouping with less visual noise |
| LOW | `src/ProfileForm.tsx:58` | `<hr>` between form sections | Replace with `mt-10` on each section heading | Section hierarchy should not depend on repeated rules |

#### Align to shared edges
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| LOW | `src/Card.tsx:24` | Card text at `pl-4`, card icon at `pl-3` | Align both to the same `pl-4` edge | Shared edges create a legible structure |
| MEDIUM | `src/Nav.css:19` | `margin-left: 16px` | `margin-inline-start: 16px` | Physical properties break direction-aware layouts |

## Verification and Verdict

After the findings:

1. **Verification**: list the exact checks run and their observed results across the relevant viewport widths, reading order, zoom, and RTL state. If a check was not run, state what still needs verification.
2. **Verdict**: `Block` if any `HIGH` finding remains, `Needs changes` if only `MEDIUM` or `LOW` findings remain, and `Approve` only when no actionable findings remain.

When there are no findings, omit the tables, state "No actionable layout findings", report verification, and end with `Approve`.
