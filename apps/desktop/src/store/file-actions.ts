import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import {
  copyTextToClipboard,
  isDesktopFsRemoteMode,
  renameDesktopPath,
  revealDesktopPath,
  trashDesktopPath
} from '@/lib/desktop-fs'
import { downloadGatewayFileWithFeedback } from '@/lib/media'
import { notify, notifyError } from '@/store/notifications'
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
export type FileActionDialog = { kind: 'delete' } & FileActionTarget

export const $fileActionDialog = atom<FileActionDialog | null>(null)

export function requestFileDelete(target: FileActionTarget): void {
  $fileActionDialog.set({ kind: 'delete', ...target })
}

export function closeFileActionDialog(): void {
  $fileActionDialog.set(null)
}

// Absolute path of the row currently being renamed inline, or null. A row whose
// path matches renders an edit input in place of its label; F2 / Enter (on a
// focused row) and the context-menu "Rename" all set this.
export const $renamingPath = atom<null | string>(null)

export function beginInlineRename(path: string): void {
  $renamingPath.set(path)
}

export function cancelInlineRename(): void {
  $renamingPath.set(null)
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

/** Remote Files panel can list gateway files but Reveal/Rename/Delete are local-only.
 *  Download is the local-copy affordance. Folders stay out — `/api/fs/download`
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

export async function executeFileRename(path: string, newName: string): Promise<void> {
  await renameDesktopPath(path, newName)
  notifyWorkspaceChanged()
}

export async function executeFileDelete(path: string): Promise<void> {
  await trashDesktopPath(path)
  notifyWorkspaceChanged()
}
