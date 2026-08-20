# Review Output Format

The format for a standalone writing review. When `better-interface` orchestrates, it owns the format, severity, consolidation, the cap, and the verdict; hand it domain evidence and findings instead.

Present the standalone review in two parts.

## Findings

Group all confirmed findings by principle. Use a markdown table with **Severity**, **Location**, **Before**, **After**, and **Why** columns. Never use separate "Before:" / "After:" lines.

- **Severity**: `HIGH` misleads users, obscures a consequence, or prevents recovery; `MEDIUM` makes a task harder to understand; `LOW` is isolated voice or consistency polish.
- **Location**: cite `path/to/file:line`. If the artifact has no source files, cite the exact screen and component instead.
- **Before / After**: quote the current copy and its complete replacement.
- **Why**: name the violated principle and explain the comprehension or trust cost.

Consolidate a repeated systemic issue into one row and list every affected location. Omit principles with no findings.

### Example

#### Errors say how to fix
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| MEDIUM | `src/PasswordField.tsx:36` | "Invalid password" | "Choose a password with at least 8 characters" | The error must say how to fix the problem |
| HIGH | `src/Editor.tsx:81` | "We couldn't process your request" toast | Inline "Unable to save. Check your connection and try again." | The current message neither locates the failure nor offers recovery |

#### Verb-first buttons
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| HIGH | `src/DeleteDialog.tsx:29` | "OK" on the delete confirmation | "Delete project" | A consequential action must repeat the consequence |
| MEDIUM | `src/Signup.tsx:54` | "Let's go!" | "Create account" | The label must name the action |

## Verification and Verdict

After the findings:

1. **Verification**: list the exact checks run and their observed results, including the complete flow, variable interpolation, pluralization, and narrow-width wrapping when applicable. If a check was not run, state what still needs verification.
2. **Verdict**: `Block` if any `HIGH` finding remains, `Needs changes` if only `MEDIUM` or `LOW` findings remain, and `Approve` only when no actionable findings remain.

When there are no findings, omit the tables, state "No actionable writing findings", report verification, and end with `Approve`.
