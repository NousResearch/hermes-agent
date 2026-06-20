# Hermes Desktop accessibility audit — v0.17.0

Target: Hermes Agent Desktop, tag `v2026.6.19` / v0.17.0.

Scope:

- `apps/desktop/` — Electron + React desktop app.
- `apps/bootstrap-installer/` — Tauri bootstrap installer.

Method: static source audit against WCAG 2.1 AA with desktop-app checks. This audit intentionally deduplicates findings by reusable component/pattern instead of listing every repeated button instance.

Limitations: static review cannot confirm actual NVDA/Narrator/Orca output, rendered contrast in every state, final tab order, or runtime keyboard traps. Those items need manual validation on Windows and Linux.

## Findings fixed in this PR

### Composer autocomplete combobox/listbox relationship

Files:

- `apps/desktop/src/app/chat/composer/index.tsx`
- `apps/desktop/src/app/chat/composer/trigger-popover.tsx`

Issue: the contenteditable composer exposed only `role="textbox"`; the `/` and `@` suggestion list was visually connected but not programmatically connected.

Fix:

- Add `aria-autocomplete="list"`, `aria-expanded`, `aria-controls`, `aria-activedescendant`, and `aria-multiline` to the composer textbox.
- Add stable listbox/option ids.
- Mark the active suggestion with `aria-selected`.
- Keep the loading spinner decorative where visible text already says that suggestions are loading.

WCAG: 1.3.1, 4.1.2.

### Preview tabs and panel semantics

File: `apps/desktop/src/app/chat/right-rail/preview.tsx`

Issue: preview tabs had partial tab semantics but no tablist label, no `aria-controls`, and no associated `tabpanel`.

Fix:

- Label the tablist.
- Connect active tabs to the panel via `aria-controls` / `aria-labelledby`.
- Add a focus-visible ring for tab buttons.

Remaining manual check: verify ArrowLeft/ArrowRight/Home/End tab navigation at runtime.

WCAG: 1.3.1, 2.1.1, 4.1.2.

### Terminal screen-reader support

Files:

- `apps/desktop/src/app/right-sidebar/terminal/index.tsx`
- `apps/desktop/src/app/right-sidebar/terminal/use-terminal-session.ts`

Issue: xterm was visually integrated but the host region had no accessible region name and xterm screen-reader mode was not enabled.

Fix:

- Label the terminal host as a terminal region.
- Enable `screenReaderMode` for xterm.

Remaining manual check: validate actual xterm behavior with NVDA/Narrator on Windows and Orca on Linux.

WCAG: 4.1.2.

### Statusbar icon-only controls

File: `apps/desktop/src/app/shell/statusbar-controls.tsx`

Issue: statusbar items could provide `title`, but the renderer did not pass it to interactive button/link/menu trigger elements. Icon-only items therefore risked unnamed controls.

Fix:

- Derive an accessible title from `item.title`, text label, or detail.
- Pass it as `aria-label` and `title` to button/link/menu trigger variants.

WCAG: 2.4.4, 4.1.2.

### Destructive confirmation dialog keyboard behavior and error announcement

File: `apps/desktop/src/components/ui/confirm-dialog.tsx`

Issue: the shared confirmation dialog intercepted Enter/Space anywhere in the dialog and ran the confirm action, even if focus was on Cancel. Inline errors were visual-only.

Fix:

- Remove the dialog-wide Enter/Space override and rely on native button semantics.
- Expose inline errors with `role="alert"` and decorative alert icons as `aria-hidden`.

WCAG: 2.1.1, 3.2.2, 3.3.1, 4.1.3.

### Cron delete icon button accessible name

File: `apps/desktop/src/app/cron/index.tsx`

Issue: the cron delete button rendered only a trash icon.

Fix: add an accessible name using the existing cron deletion copy.

WCAG: 4.1.2.

### Bootstrap installer progress and details semantics

Files:

- `apps/bootstrap-installer/src/routes/progress.tsx`
- `apps/bootstrap-installer/src-tauri/tauri.conf.json`
- `apps/bootstrap-installer/src/styles.css`

Issues:

- Tauri window title was generic (`Hermes`) instead of installer-specific.
- Progress bar was visual-only.
- Stage state was icon/color-only.
- Details disclosure lacked `aria-expanded` / `aria-controls`.
- Installer fade-in animation ignored reduced-motion preference.

Fixes:

- Rename the window title to `Hermes Setup`.
- Add `role="progressbar"` with value attributes and `aria-valuetext`.
- Add a polite status region for current progress.
- Add text equivalents for stage states and mark status icons decorative.
- Add disclosure relationship semantics for the log panel.
- Disable the installer fade animation under `prefers-reduced-motion: reduce`.

WCAG: 1.3.1, 1.4.1, 2.2.2, 2.4.2, 4.1.2, 4.1.3.

## Deduplicated findings still recommended for follow-up

### Shared settings row labels

Pattern: `ListRow` visually pairs labels/descriptions with controls, but many generated switches, selects, inputs, and textareas are not programmatically associated with the row text.

Representative files:

- `apps/desktop/src/app/settings/primitives.tsx`
- `apps/desktop/src/app/settings/config-settings.tsx`
- `apps/desktop/src/app/settings/notifications-settings.tsx`
- `apps/desktop/src/app/skills/index.tsx`

Recommendation: create a shared labeled-row field API that emits ids and connects `aria-labelledby` / `aria-describedby` to the child control.

### Active navigation state semantics

Pattern: several custom navigation rows expose selection only visually.

Representative files:

- `apps/desktop/src/components/ui/text-tab.tsx`
- `apps/desktop/src/app/settings/primitives.tsx`
- `apps/desktop/src/app/overlays/overlay-split-layout.tsx`
- `apps/desktop/src/app/profiles/index.tsx`
- `apps/desktop/src/app/messaging/index.tsx`

Recommendation: use `aria-current`, `aria-selected`, or proper tab/radio semantics depending on the control pattern.

### Pointer/drag-only workflows

Pattern: some reference insertion and preview line-selection flows rely on mouse selection or drag/drop.

Representative files:

- `apps/desktop/src/app/chat/right-rail/preview-file.tsx`
- `apps/desktop/src/app/chat/sidebar/session-row.tsx`

Recommendation: add keyboard-equivalent commands for selecting source lines/ranges and inserting session/file/path references.

### Resizable preview console separator

File: `apps/desktop/src/app/chat/right-rail/preview-console.tsx`

Recommendation: make the separator focusable and expose `aria-orientation`, `aria-valuemin`, `aria-valuemax`, `aria-valuenow`, and keyboard resize behavior.

### Live regions for streaming surfaces

Representative files:

- `apps/desktop/src/components/assistant-ui/thread-list.tsx`
- `apps/desktop/src/app/chat/right-rail/preview-console.tsx`

Recommendation: use a labeled `role="log"` for chat/console output and targeted polite/assertive status messages for progress/errors without announcing every streamed token.

## Manual validation checklist

Validate on Windows with NVDA/Narrator and on Linux with Orca:

- Composer `/` and `@` suggestions: announcement of suggestions, active option, selection, and dismissal.
- Preview tabs: Tab reachability and arrow-key tab switching.
- Terminal: xterm screen-reader output, focus entry/exit, selection-to-chat shortcut, and paste/drop alternatives.
- Statusbar: icon-only controls have names in the screen reader object model.
- Confirm dialogs: Space/Enter on Cancel does not confirm destructive actions.
- Installer: title in task switcher, progressbar announcements, details disclosure, failure state announcement, and reduced-motion behavior.
