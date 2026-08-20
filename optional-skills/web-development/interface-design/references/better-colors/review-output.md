# Review Output Format

The format for a standalone color review. When `better-interface` orchestrates, it owns the format, severity, consolidation, the cap, and the verdict; hand it domain evidence and findings instead.

Present the standalone review in two parts.

## Findings

Group all confirmed findings by principle. Use a markdown table with **Severity**, **Location**, **Before**, **After**, and **Why** columns. Never use separate "Before:" / "After:" lines.

- **Severity**: `HIGH` makes content unreadable or assigns a misleading semantic color; `MEDIUM` creates a noticeable theme, gamut, or consistency failure; `LOW` is isolated polish.
- **Location**: cite `path/to/file:line`. If the artifact has no source files, cite the exact screen and component instead.
- **Before / After**: show the current value or token and the exact replacement.
- **Why**: name the violated principle and include measured contrast or gamut evidence when relevant.

Consolidate a repeated systemic issue into one row and list every affected location. Omit principles with no findings.

### Example

| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| MEDIUM | `src/theme.css:18` | `color: #3b82f6` | `color: oklch(0.623 0.188 259.815)` | New project colors use OKLCH tokens |
| MEDIUM | `src/palette.ts:31` | Same absolute C across hues | Same C% of each hue's maximum chroma | Equal chroma values do not appear equally vivid across hues |
| HIGH | `src/theme.css:52` | P3 color with no fallback | Add an sRGB fallback before `@media (color-gamut: p3)` | The color fails on non-P3 displays |

## Verification and Verdict

After the findings:

1. **Verification**: list the exact checks run and their observed results, including contrast measurements, gamut checks, and both light and dark appearances when applicable. If a check was not run, state what still needs verification.
2. **Verdict**: `Block` if any `HIGH` finding remains, `Needs changes` if only `MEDIUM` or `LOW` findings remain, and `Approve` only when no actionable findings remain.

When there are no findings, omit the table, state "No actionable color findings", report verification, and end with `Approve`.
