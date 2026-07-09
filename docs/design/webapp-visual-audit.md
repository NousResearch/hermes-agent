# Hermes Webapp visual audit notes

Date: 2026-07-08
Scope: browser Webapp (`web/`) chat and files surfaces. This is not a Dashboard rename and not a full Desktop replacement; the goal is a browser-native workspace that borrows sensible Desktop affordances where they apply.

## Underlying product goal

The Webapp should feel like a coherent chat + workspace surface:

- Chat stays usable at desktop, tablet, and phone widths.
- Model/session controls are reachable without clipping or stealing focus while hidden.
- File management should behave like a basic browser/IDE file browser: path, refresh, search, upload, new folder, open/download/delete, usable mobile layout, and recovery when the default root is bad.
- Hidden/offscreen panels must not leave ghost controls in the tab order.

## Visual audit evidence

Captured responsive screenshots with Chrome DevTools device metrics at:

- 1440x900 desktop
- 768x1024 tablet
- 390x844 phone
- 320x720 narrow phone

Pre-fix and post-fix screenshots should be kept as local audit artifacts, not committed to the repository.

Authoritative post-fix screenshots should be CDP-forced captures, because plain headless `--window-size=390` can keep `window.innerWidth=500` and crop the result.

## Issues found

### Fixed in this patch

1. **Chat mobile header action clipped/missing in screenshots**
   - The Model/tools action did not reliably fit the phone header.
   - It now uses a compact `Panel` label on mobile and is constrained inside the header slot.

2. **Chat copy-last action clipped on phone widths**
   - The floating `copy last response` control could extend past the right edge.
   - It is now icon-only below `sm` and max-width constrained.

3. **Closed mobile Model/tools sheet was visually hidden but still focusable**
   - The sheet was translated offscreen while its internal controls remained in the DOM.
   - Closed sheet contents are no longer rendered; the closed container is aria-hidden.

4. **Files bad-root state was a dead end**
   - A missing default root rendered a raw 500, disabled upload/create, and gave no path recovery.
   - The Files page now keeps a path textbox visible, extracts the failed path, and offers Retry/Open parent.

5. **Files lacked expected file-browser controls in the main surface**
   - Refresh was only in the page header; there was no search/filter; `Create` was vague.
   - The page now exposes Refresh, Search files, Upload, and New folder together with the path bar.

6. **Files mobile layout clipped the table**
   - The table had a hard `min-w-[42rem]`, which caused phone screenshots to crop columns/actions.
   - Desktop keeps the table; mobile now renders stacked file cards/actions.

### Remaining product gaps, intentionally not fixed here

1. **No Webapp integrated preview/code-editor rail yet**
   - Desktop has richer preview/editor/file-browser panes. Webapp still primarily embeds the TUI plus separate Files page.
   - A future Webapp-specific pass should define browser-native preview/editor tabs rather than blindly copying Desktop internals.

2. **No backend create-file endpoint in the managed-files API**
   - The UI now says `New folder`; it does not fake a `New file` button because the current supported write affordances are upload and mkdir.

3. **Files page still downloads files instead of previewing/editing in-place**
   - The existing managed-files API supports read/download. A separate feature should add safe text preview/edit UX if desired.

4. **Global mobile navigation likely needs an inert/focus audit**
   - This patch fixed the chat Model/tools sheet. The app sidebar uses a similar offscreen translation pattern and should get a dedicated accessibility pass.

## Verification checklist

- `npm --workspace web run test -- src/lib/files-ui.test.ts src/lib/api.test.ts src/lib/chat-title.test.ts`
- `npm --workspace web run typecheck`
- Targeted eslint on touched files
- `npm --workspace web run build`
- CDP responsive visual pass at 1440, 768, 390, and 320 widths
