# Reuben Remote Filesystem Audit

## Summary

This is a read-only audit of Reuben Desktop's remote-mode filesystem and
workspace semantics. No app code was changed in this pass.

The renderer is mostly on the right track: workspace file reads, text writes,
project tree loading, and git/review actions usually flow through remote-aware
facades. The remaining risk is at UI edges inherited from local-desktop Hermes:
local file-manager reveal actions, local rename/trash helpers, local PTY
terminals, and ambiguous local file drops or `file://` preview/open paths.

## Files/Functions Inspected

- `apps/desktop/src/lib/desktop-fs.ts`: `isDesktopFsRemoteMode`,
  `readDesktopDir`, `readDesktopFileText`, `writeDesktopFileText`,
  `readDesktopFileDataUrl`, `desktopGitRoot`, `desktopDefaultCwd`,
  `revealDesktopPath`, `renameDesktopPath`, `trashDesktopPath`,
  `selectDesktopPaths`.
- `apps/desktop/src/lib/desktop-git.ts`: `desktopGit`, `remoteGit`,
  `remoteGitRequest`, `remoteGitRequestOptional`.
- `apps/desktop/src/app/right-sidebar/file-actions.tsx`:
  `FileEntryContextMenu`, `pickRevealLabel`, `InlineRenameInput`.
- `apps/desktop/src/app/right-sidebar/files/use-project-tree.ts`:
  `fallbackRootFor`, `revalidateTree`, project tree loading keyed by
  connection mode/profile/backend.
- `apps/desktop/src/app/right-sidebar/files/ipc.ts`: `readProjectDir`.
- `apps/desktop/src/app/right-sidebar/review/file-tree.tsx`: review file
  context menu reveal, copy, stage/unstage, and revert actions.
- `apps/desktop/src/app/chat/hooks/use-composer-actions.ts`:
  `extractDroppedFiles`, `partitionDroppedFiles`, `pickContextPaths`,
  `attachDroppedItems`, `pickImages`, `pasteClipboardImage`.
- `apps/desktop/src/app/chat/index.tsx`: `ChatView` drop handling,
  especially `onDropFiles`.
- `apps/desktop/src/components/assistant-ui/thread.tsx`: edit-composer
  `insertDroppedRefs`, `uploadOsDropRefs`, `handleDrop`.
- Related inspected files: `apps/desktop/src/app/session/hooks/use-prompt-actions.ts`
  (`uploadComposerAttachment`), `apps/desktop/src/store/projects.ts`
  (`revealPath`, project/git actions), project menu components, terminal
  modules under `apps/desktop/src/app/right-sidebar/terminal/`, and relevant
  IPC handlers in `apps/desktop/electron/main.cjs`.

## Correct Current Remote-Mode Behavior

- `apps/desktop/src/lib/desktop-fs.ts` uses `$connection.mode` through
  `isDesktopFsRemoteMode()`. In remote mode, `readDesktopDir`,
  `readDesktopFileText`, `writeDesktopFileText`, `readDesktopFileDataUrl`,
  `desktopGitRoot`, `desktopDefaultCwd`, and `desktopFileDiff` call gateway REST
  routes such as `/api/fs/list`, `/api/fs/read-text`, `/api/fs/write-text`,
  `/api/fs/read-data-url`, `/api/fs/git-root`, `/api/fs/default-cwd`, and
  `/api/git/file-diff`.
- `apps/desktop/src/lib/desktop-git.ts` returns the `remoteGit` implementation
  from `desktopGit()` when remote mode is active. Worktree, branch, status,
  diff, review, commit, push, and PR operations go through `/api/git/*`
  endpoints rather than Electron-local git IPC.
- `desktopGit().scanRepos()` is a remote-mode no-op. That avoids crawling the
  user's local disk to discover repositories for a remote backend.
- `apps/desktop/src/app/right-sidebar/files/use-project-tree.ts` does not fall
  back to local sanitized paths in remote mode. `fallbackRootFor()` returns
  `null`, and tree reload keys include connection mode, profile, and backend
  URL so stale local/remote roots are not silently reused across connection
  changes.
- `apps/desktop/src/app/right-sidebar/files/ipc.ts` loads directories through
  `readDesktopDir()`, so the file tree and `.gitignore` filtering use backend
  paths in remote mode.
- `apps/desktop/src/app/right-sidebar/file-actions.tsx` already hides
  file-manager reveal, rename, and delete actions in `FileEntryContextMenu`
  when `isDesktopFsRemoteMode()` is true. Copy path and copy relative path
  remain available, which is appropriate for backend paths.
- `apps/desktop/src/app/right-sidebar/review/file-tree.tsx` keeps "Reveal in
  filetree" available for remote workspaces while hiding the OS file-manager
  reveal item when `isDesktopFsRemoteMode()` is true.
- `apps/desktop/src/app/chat/hooks/use-composer-actions.ts` explicitly
  separates OS/Finder drops from in-app file tree or gutter drags via
  `partitionDroppedFiles()`. The comment correctly states that OS drops are
  local imports/uploads, while in-app path-only drags are backend-resolvable
  `@file:`, `@folder:`, or `@line:` refs.
- `apps/desktop/src/app/chat/index.tsx` follows that split in `onDropFiles()`:
  in-app refs are inserted inline, and OS drops are routed to the attachment
  pipeline.
- `apps/desktop/src/components/assistant-ui/thread.tsx` applies the same split
  in the message edit composer. When a session/gateway is available,
  `uploadOsDropRefs()` stages OS drops through `uploadComposerAttachment()` and
  inserts the gateway-side ref instead of a raw local path.
- `apps/desktop/src/app/session/hooks/use-prompt-actions.ts`
  `uploadComposerAttachment()` uploads bytes for remote file and image
  attachments, using Electron only to read the explicitly chosen local file
  before sending it to gateway methods such as `file.attach` or
  `image.attach_bytes`.

## Local Filesystem Assumptions Still Present

- `apps/desktop/src/lib/desktop-fs.ts` still exposes local-only operations
  without an internal remote-mode guard: `revealDesktopPath()`,
  `renameDesktopPath()`, and `trashDesktopPath()`. The right-sidebar file menu
  currently guards its use, but the facade itself will still call Electron IPC
  if another caller invokes it with a backend path.
- `apps/desktop/electron/main.cjs` still contains local IPC handlers:
  `hermes:fs:reveal`, `hermes:fs:rename`, `hermes:fs:writeText`,
  `hermes:fs:trash`, `hermes:git:*`, and `hermes:terminal:start`. These are not
  inherently wrong, but they are local OS operations and should be treated as
  local-only capabilities from the renderer.
- `apps/desktop/src/store/projects.ts` exposes `revealPath()`, which directly
  calls `window.hermesDesktop?.revealPath?.(path)`. Project/worktree menus in
  `apps/desktop/src/app/chat/sidebar/projects/project-menu.tsx` and
  `workspace-header.tsx` still render Reveal actions for project or worktree
  paths without checking remote mode. In remote mode those paths describe the
  backend environment, not necessarily the local Mac filesystem.
- The interactive user terminal is local. `apps/desktop/src/app/right-sidebar/terminal/terminals.ts`
  `createTerminal()` captures `$currentCwd`; `persistent.tsx` calls
  `ensureTerminal()` when the terminal pane opens; `use-terminal-session.ts`
  calls `window.hermesDesktop?.terminal.start({ cwd })`; and
  `apps/desktop/electron/main.cjs` handles `hermes:terminal:start` by spawning
  `nodePty.spawn()`. If the backend cwd does not exist locally,
  `safeTerminalCwd()` falls back to the local home directory. That makes the
  terminal a local shell even while the visible workspace paths are remote.
- `apps/desktop/src/components/assistant-ui/thread.tsx` has a fallback in
  `uploadOsDropRefs()` that returns inline refs for OS drops if no gateway or
  session is present. That matches old behavior, but it could still leak a raw
  local path into edited text if the edit composer is somehow active without a
  usable gateway/session.
- In `apps/desktop/src/app/chat/hooks/use-composer-actions.ts`,
  `attachDroppedItems()` initially stores an OS drop's local path on the
  composer attachment. Submit-time remote upload rewrites it to a gateway-side
  path, so the data flow is mostly safe, but the chip can temporarily display a
  local path in a remote workspace. Image drops also try `attachImagePath()` on
  the local path first; final remote upload still happens at submit time.
- `pickImages()` uses `selectDesktopPaths()` for a file picker. In remote mode,
  `selectDesktopPaths()` only delegates directory selection to the remote
  picker and returns `[]` for file selection. That avoids incorrectly treating a
  local chosen file as a backend file, but it may also make an explicit local
  image import affordance silently unavailable.
- `apps/desktop/electron/main.cjs` has a `file://` open path that delegates to
  `shell.openPath()` and falls back to `shell.showItemInFolder()`. That is
  correct for local artifacts, but a backend-generated `file://` URL or remote
  absolute path would be misleading in remote mode.

## Explicit Local Import Or Upload Flows That Are Valid

- Finder/Explorer drag and drop into the composer is valid as a local import,
  because those drops carry a native `File` handle and are separated from
  in-app path-only drags by `partitionDroppedFiles()`.
- Clipboard image paste is valid as local import. `pasteClipboardImage()` calls
  `window.hermesDesktop?.saveClipboardImage()` and then attaches the saved local
  image for upload/staging.
- Saving a pasted/dropped image buffer through `saveImageBuffer()` is valid as
  local staging before upload. The important boundary is that the resulting
  file should be treated as a local upload source, not a backend path.
- Reading local file bytes inside `uploadComposerAttachment()` is valid in
  remote mode because it is part of an explicit user import/upload path.
- Local clipboard copy of backend paths is valid. Copying a remote path does
  not imply the local OS can open or mutate it.
- Revealing Reuben Desktop's own local logs is valid because those logs belong
  to the frontend app, not the remote backend workspace.

## Actions To Hide, Disable, Or Rename In Remote Mode

- Hide or disable project/worktree "Reveal" actions in
  `project-menu.tsx` and `workspace-header.tsx` when `$connection.mode ===
  "remote"`. The right-sidebar file tree already does this correctly through
  `FileEntryContextMenu`.
- Add a defensive remote-mode guard to `revealDesktopPath()`,
  `renameDesktopPath()`, and `trashDesktopPath()` in `desktop-fs.ts`, or ensure
  every caller is statically gated. A facade-level guard would make accidental
  future calls less dangerous.
- Hide the interactive user terminal in remote mode, or relabel it explicitly
  as a local terminal and prevent it from inheriting backend cwd values. A real
  remote terminal should use a backend/gateway PTY path rather than Electron
  `node-pty`.
- Treat `file://` open actions as local-only. In remote mode, backend files
  should open through the existing preview/fetch path or a gateway-served media
  URL, not Electron `shell.openPath()`.
- Consider renaming explicit local import actions to "Upload local file" or
  "Attach local image" where the current label could imply selecting a backend
  path. Backend file context selection and local upload selection are different
  product actions.

## Recommended Follow-Up Fixes/Tests

- Add component tests that project/worktree menus do not render or do not call
  local `revealPath()` when `$connection.mode === "remote"`.
- Add focused tests for `desktop-fs.ts` local-only helpers so
  `revealDesktopPath()`, `renameDesktopPath()`, and `trashDesktopPath()` are
  blocked or no-op in remote mode.
- Add a terminal remote-mode test proving that opening the terminal pane does
  not create a user `node-pty` terminal with a backend cwd. Keep agent
  background terminal mirrors available if they are backend-originated streams.
- Add a drag/drop regression test for the edit composer fallback in
  `thread.tsx` so OS drops cannot become raw inline local paths in remote mode.
- Add a `file://` preview/open test that distinguishes local artifacts from
  remote backend paths and avoids `shell.openPath()` for remote workspace files.
- Add a small UX test around `pickImages()` or the image attach button in remote
  mode. It should either clearly upload a local image or be disabled with a
  clear reason, rather than silently returning no selection.
