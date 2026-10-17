# Desktop shared-component accessibility audit

Baseline: `1744a19e0d` on `main`. This replaces the stale audit behind #49413, not a claim of complete Desktop accessibility.

Scope: the native Electron/React Desktop in `apps/desktop/`. Source review using the accessibility-auditor checklist, rendered React tests and a real Chromium keyboard smoke. Independent GPT-6-Astra review, followed by fixes and a second review. The Tauri bootstrap installer, web dashboard and Ink TUI are separate surfaces.

## Confirmed root causes addressed

- **Confirmation dialogs:** an ancestor Enter/Space handler confirmed even when Cancel, a secondary action or an input owned focus. Removed that interception so native buttons own activation. Destructive dialogs initially focus Cancel. Inline failures expose one alert. WCAG risk mapping: 2.1.1, 3.2.2, 4.1.3.
- **Statusbar names:** shared button/link/menu rendering did not expose existing title metadata for ordinary icon-only items. Added a fallback without overriding visible text names. WCAG: 4.1.2.
- **Settings relationships:** shared ListRow visually placed titles/descriptions beside segmented controls without programmatic association. Added explicit render-action relations and migrated direct Appearance, Billing and Keep Awake consumers. Existing ToggleRow explicit labels remain unchanged. WCAG: 1.3.1, 3.3.2, 4.1.2.
- **Current view:** shared button-style TextTab selection was visual-only. Added pressed state without claiming an unsupported ARIA tab/arrow-key contract. WCAG: 4.1.2.
- **Loading:** added an explicit decorative mode to the shared glyph spinner. The autocomplete loading row retains one localized status announcement instead of a separate generic spinner announcement. Meaningful spinner-only status defaults remain unchanged. WCAG: 4.1.3.
- **Terminals:** both interactive PTY and read-only agent xterm construction enable the supported screen-reader mode. Tests cover both constructors. WCAG risk mapping: 4.1.2; spoken behavior remains unverified.

These fixes target common primitives and their direct integrations. No runtime DOM rewriting, accessibility overlay, new plugin or new dependency is introduced.

## Existing behavior retained

Electron already enables renderer accessibility on Windows/macOS. CSS already respects reduced motion. Radix supplies modal semantics and trapping; no duplicate dialog framework is added. Settings toggles already have explicit names. User-visible names reuse existing localized copy.

## Verification

- Focused/shared UI tests: 38 files, 150 tests passed.
- Renderer, Electron and E2E TypeScript checks, plus checked Electron builder JavaScript.
- Changed-file ESLint, Prettier and `git diff --check`.
- Real Chromium native keyboard smoke using the actual ConfirmDialog: Enter/Space on Cancel, secondary and Confirm; input-key isolation; pending repeat suppression; destructive initial focus.

Run the portable browser smoke from the repository root:

```sh
node apps/desktop/scripts/verify-confirm-dialog-keyboard.mjs
```

It requires Playwright Chromium (`npx playwright install chromium`). An existing Chromium may instead be selected with `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`. The script uses an isolated fixture without an agent backend or user profile and cleans up its fixture and server.

## Deferred and manual validation

This is a bounded shared-component remediation, not full WCAG conformance.

- Validate actual NVDA, VoiceOver and Orca output, announcements, keyboard shortcuts and terminal focus/output. Browser keyboard tests do not establish spoken behavior or terminal performance.
- Measure rendered contrast, target sizes and zoom/reflow across themes.
- Composer completion input/listbox active-descendant relationships, layout tab/resize/drag alternatives, and broader navigation need their own integration-level repros and follow-up.
- Controlled dialog restoration to the actual opener is a pre-existing gap; this PR does not claim to fix it. The existing `dismissOnConfirm` pending path also warrants separate behavioral review.
- Bootstrap installer accessibility remains a separate task.
