import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { translateNow } from '@/i18n'
import {
  captureDesktopFsOperationScope,
  copyTextToClipboard,
  createDesktopEntry,
  desktopFsCacheKey,
  type DesktopFsOperationScope,
  isDesktopFsRemoteMode,
  renameDesktopPath,
  revealDesktopPath,
  trashDesktopPath
} from '@/lib/desktop-fs'
import { downloadGatewayFileWithFeedback } from '@/lib/media'
import { notify, notifyError } from '@/store/notifications'
import {
  $connection,
  $cronSessions,
  $messagingSessions,
  $sessionOwnerRevision,
  $sessions,
  $unlistedSessionOwnerRows
} from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles, knownOwnerForSession } from '@/store/session-states'
import { notifyWorkspaceChanged } from '@/store/workspace-events'

// Shared file-row actions for BOTH trees (the file browser + the review/git
// tree): reveal, copy path, download (remote), rename, delete. Rename/delete
// route through a single dialog set (driven by this atom, rendered once by
// `FileActionDialogs`) instead of one dialog per row. After a successful
// mutation we bump the workspace tick so every git-/fs-mirroring surface
// refreshes.

export interface FileActionTarget {
  isDirectory: boolean
  /** Display name (basename) shown in dialogs. */
  name: string
  /** Absolute path on disk. */
  path: string
}

// Delete routes through a single confirm dialog (rendered once). Rename is
// INLINE (VS Code style — an input in the row), driven by `$renamingPath`.
export interface FileActionScope extends DesktopFsOperationScope {
  contextKey: string
}

export type FileActionDialog = { kind: 'delete'; scope: FileActionScope } & FileActionTarget

export const $fileActionDialog = atom<FileActionDialog | null>(null)

export function requestFileDelete(target: FileActionTarget): void {
  const scope = captureFileActionScope()

  if (scope) {$fileActionDialog.set({ kind: 'delete', scope, ...target })}
}

export function closeFileActionDialog(): void {
  $fileActionDialog.set(null)
}

// Absolute path of the row currently being renamed inline, or null. A row whose
// path matches renders an edit input in place of its label; F2 / Enter (on a
// focused row) and the context-menu "Rename" all set this.
export const $renamingPath = atom<null | string>(null)
export const $renamingScope = atom<FileActionScope | null>(null)

export function beginInlineRename(path: string): void {
  const scope = captureFileActionScope()

  if (scope) {
    $renamingPath.set(path)
    $renamingScope.set(scope)
  }
}

export function cancelInlineRename(): void {
  $renamingPath.set(null)
  $renamingScope.set(null)
}

// Pending new-file/new-folder creation, if any: `{ parentDir, directory }`.
// The folder row whose id matches `parentDir` renders an inline input for the
// new entry's name (empty seed, select-all on focus) — the same VS Code flow
// as rename, reusing the InlineRenameInput.
export interface CreatingEntry {
  directory: boolean
  parentDir: string
  scope?: FileActionScope
}

export const $creatingEntry = atom<CreatingEntry | null>(null)

export async function requestNewEntry(creating: Omit<CreatingEntry, 'scope'>): Promise<void> {
  // Creating inside a collapsed folder is invisible — expand it first via the
  // tree's open-state store, then let the row render the input.
  const scope = captureFileActionScope()

  if (scope) {$creatingEntry.set({ ...creating, scope })}
}

export function cancelNewEntry(): void {
  $creatingEntry.set(null)
}

// Trees list the window owner; previews currently follow the focused session.
// Do not mutate either host when those owners differ. A session id is deliberately
// NOT forwarded: tree entries are host/SSH-workspace paths, not Docker sandbox paths.
function focusedFileOwner(): { connectionId?: string; profile: string; targetProfile?: string } | undefined {
  const sessionId = $focusedStoredSessionId.get()
  const owner = knownOwnerForSession(sessionId)

  return typeof owner === 'string' ? { profile: owner } : owner ?? undefined
}

function fileActionContextKey(): string {
  const scope = captureDesktopFsOperationScope()
  const owner = focusedFileOwner()

  return JSON.stringify([
    scope.cacheKey,
    $focusedStoredSessionId.get(),
    owner?.connectionId || scope.connectionId || LOCAL_CONNECTION_ID,
    owner?.profile || scope.profile || 'default',
    owner?.targetProfile ?? null,
    Boolean(owner)
  ])
}

function assertFileActionOwner(scope?: FileActionScope): void {
  const tree = captureDesktopFsOperationScope()
  const owner = focusedFileOwner()
  const connectionId = tree.connectionId || LOCAL_CONNECTION_ID
  const profile = tree.profile || 'default'

  if (owner && ((owner.connectionId || connectionId) !== connectionId || owner.profile !== profile)) {
    cancelPendingFileActions()
    throw new Error(
      'The focused session uses a different connection or profile from the Files tree. Switch to a session owned by this window before changing files.'
    )
  }

  if (scope && scope.contextKey !== fileActionContextKey()) {
    cancelPendingFileActions()
    throw new Error('The file action owner changed before the operation could start')
  }
}

function captureFileActionScope(): FileActionScope | null {
  try {
    assertFileActionOwner()

    return { ...captureDesktopFsOperationScope(), contextKey: fileActionContextKey() }
  } catch (error) {
    notifyError(error, translateNow('errors.genericFailure'))

    return null
  }
}

function cancelPendingFileActions(): void {
  cancelNewEntry()
  cancelInlineRename()
  closeFileActionDialog()
}

// Read the production owner resolver, including late tile/row/hint/event ownership.
// A semantic key avoids canceling an editor on unrelated session-list refreshes.
const $fileActionContext = computed(
  [
    $connection,
    $focusedStoredSessionId,
    $sessionTiles,
    $sessions,
    $cronSessions,
    $messagingSessions,
    $unlistedSessionOwnerRows,
    $sessionOwnerRevision
  ],
  fileActionContextKey
)

let lastContextKey = $fileActionContext.get()
$fileActionContext.subscribe(key => {
  if (key !== lastContextKey) {cancelPendingFileActions()}
  lastContextKey = key
})

/** Create the entry after the inline input commits. Throws on failure so the
 *  CALLER (InlineRenameInput) owns the error notification — this function only
 *  cleans up state. Bumps the workspace tick on success. */
export async function executeEntryCreate(creating: CreatingEntry, name: string): Promise<string> {
  if ($creatingEntry.get() !== creating) {
    throw new Error('File creation was canceled')
  }

  try {
    assertFileActionOwner(creating.scope)
    const created = await createDesktopEntry(creating.parentDir, name, creating.directory, creating.scope)

    if (creating.scope?.cacheKey === desktopFsCacheKey()) {
      notifyWorkspaceChanged()
    }

    return created
  } finally {
    // A late result must not clear a newer editor opened after this one.
    if ($creatingEntry.get() === creating) {
      cancelNewEntry()
    }
  }
}

// ── Direct (no-dialog) actions ───────────────────────────────────────────────

export async function revealFile(path: string): Promise<void> {
  try {
    await revealDesktopPath(path)
  } catch (error) {
    notifyError(error, translateNow('errors.genericFailure'))
  }
}

export async function copyFilePath(path: string): Promise<void> {
  try {
    await copyTextToClipboard(path)
    notify({ durationMs: 1500, kind: 'info', message: translateNow('fileMenu.pathCopied') })
  } catch (error) {
    notifyError(error, translateNow('common.copyFailed'))
  }
}

/** Whether "Open containing folder" can work for a session's workspace: only
 *  when that session's backend is this computer. A row tagged with a Connections
 *  gateway other than `local` runs there; an untagged row runs on the window's
 *  primary, remote or not (the rule the sidebar menus already apply). */
export function shouldOfferLocalReveal(
  connectionId: null | string | undefined,
  primaryRemote = isDesktopFsRemoteMode()
): boolean {
  const tagged = String(connectionId || '').trim()

  return tagged ? tagged === LOCAL_CONNECTION_ID : !primaryRemote
}

/** Download is the local-copy affordance for a remote file; Reveal remains
 *  local-only. Folders stay out — `/api/fs/download`
 *  streams a single file. */
export function shouldOfferRemoteFileDownload(isDirectory: boolean, remote = isDesktopFsRemoteMode()): boolean {
  return remote && !isDirectory
}

export function downloadRemoteFile(path: string): Promise<void> {
  return downloadGatewayFileWithFeedback(path)
}

/** Strip a `relativeTo` prefix to produce a repo/cwd-relative path. */
export function toRelativePath(path: string, relativeTo: string): string {
  const base = relativeTo.replace(/[\\/]+$/, '')

  if (path === base) {
    return path
  }

  return path.startsWith(`${base}/`) || path.startsWith(`${base}\\`) ? path.slice(base.length + 1) : path
}

// ── Dialog-confirmed mutations (called by FileActionDialogs) ──────────────────

// Caller-owned error handling, one source of truth: the rejection propagates
// so the UI surface that invoked the action owns the error (InlineRenameInput
// toasts rename failures, ConfirmDialog shows delete failures inline). No
// notify here — the old stack toasted AND re-threw, doubling the message.
export async function executeFileRename(path: string, newName: string, scope?: FileActionScope): Promise<void> {
  if (scope && $renamingScope.get() !== scope) {throw new Error('File rename was canceled')}
  assertFileActionOwner(scope)
  await renameDesktopPath(path, newName, scope)

  if (!scope || scope.cacheKey === desktopFsCacheKey()) {
    notifyWorkspaceChanged()
  }
}

export async function executeFileDelete(path: string, scope?: FileActionScope): Promise<void> {
  if (scope && $fileActionDialog.get()?.scope !== scope) {throw new Error('File deletion was canceled')}
  assertFileActionOwner(scope)
  await trashDesktopPath(path, scope)

  if (!scope || scope.cacheKey === desktopFsCacheKey()) {
    notifyWorkspaceChanged()
  }
}
