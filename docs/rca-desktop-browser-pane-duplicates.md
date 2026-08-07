# Desktop in-app Browser pane duplication

Status: fixed locally; draft PR [#80929](https://github.com/NousResearch/hermes-agent/pull/80929) is awaiting contributor review.

Date investigated: 2026-08-07

## User-visible symptoms

- Hermes Desktop could show two in-app `BROWSER` panes for one Browser surface.
- Right-clicking one Browser pane and choosing Remove could remove both panes.
- The layout could look correct after restart, then gain a second Browser pane when an agent opened another URL.

## Root causes

There were two related persistence problems:

1. The persisted layout tree could contain the same pane contribution ID in more than one group. The Browser pane uses a shared contribution ID, so closing that ID affected every persisted occurrence.
2. The persisted preview decoder called `previewTabId()` while the `BROWSER_TAB_ID` constant was still in the module's temporal dead zone. `persistentAtom()` caught the initialization error and fell back to an empty preview-tab list. The next agent-driven URL open then added a new Browser pane beside the stale layout entry.

## Fix

- Normalize pane IDs globally across the whole layout tree, keeping the first occurrence in tree order.
- Initialize `BROWSER_TAB_ID` before persisted preview state is decoded.
- Keep all restored URL previews on the singleton `url:browser` ID and retain the latest URL.
- Add regression tests for duplicate layout entries and multiple persisted URL tabs.

Changed files:

- `apps/desktop/src/components/pane-shell/tree/model.ts`
- `apps/desktop/src/components/pane-shell/tree/remove-pane.test.ts`
- `apps/desktop/src/store/preview.ts`
- `apps/desktop/src/store/preview-persistence.test.ts`

## Local verification

Completed before updating PR #80929:

- UI tests: 59 suites / 450 tests passed.
- Desktop typecheck passed.
- ESLint passed for all changed Desktop files.
- `git diff --check` passed.
- Rebuilt and launched the local macOS `Hermes.app`.
- Confirmed one Browser after restart.
- Confirmed one Browser after the agent opened one URL and then a different URL.

## Update warning

Until PR #80929 is merged and included in an official release, installing an older official Hermes build can overwrite the local packaged app and reintroduce this bug. Use the locally built app for verification until the fixed release is available.
