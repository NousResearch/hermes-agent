# Review Output Format

The format for a standalone typography review. When `better-interface` orchestrates, it owns the format, severity, consolidation, the cap, and the verdict; hand it domain evidence and findings instead.

Present the standalone review in two parts.

## Findings

Group all confirmed findings by principle. Use a markdown table with **Severity**, **Location**, **Before**, **After**, and **Why** columns. Never use separate "Before:" / "After:" lines.

- **Severity**: `HIGH` makes text unreadable, unavailable, or structurally misleading; `MEDIUM` harms hierarchy, wrapping, or scanning; `LOW` is isolated typographic polish.
- **Location**: cite `path/to/file:line`. If the artifact has no source files, cite the exact screen and component instead.
- **Before / After**: show the current typography and an actionable replacement.
- **Why**: name the violated principle and its effect on readability or hierarchy.

Consolidate a repeated systemic issue into one row and list every affected location. Omit principles with no findings.

### Example

#### Tabular numbers
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| MEDIUM | `src/Price.tsx:17` | `<span>{price}</span>` on a live price | `<span className="tabular-nums">{price}</span>` | Proportional digits cause changing values to shift |
| LOW | `src/numbers.css:8` | `font-feature-settings: "tnum" 1` | `font-variant-numeric: tabular-nums` | The high-level property preserves fallback behavior |

#### Line-height and measure
| Severity | Location | Before | After | Why |
| --- | --- | --- | --- | --- |
| MEDIUM | `src/Article.tsx:33` | `leading-none` on a body paragraph | `leading-normal` (`1.5`–`1.6`) | Wrapped body text needs enough vertical separation |
| MEDIUM | `src/article.css:12` | Full-width article column | `max-width` near 65 characters at `16px` | Long measures make lines hard to track |

## Verification and Verdict

After the findings:

1. **Verification**: list the exact checks run and their observed results, including wrapping, hierarchy, text resizing, font loading, and dynamic-value stability when applicable. If a check was not run, state what still needs verification.
2. **Verdict**: `Block` if any `HIGH` finding remains, `Needs changes` if only `MEDIUM` or `LOW` findings remain, and `Approve` only when no actionable findings remain.

When there are no findings, omit the tables, state "No actionable typography findings", report verification, and end with `Approve`.
