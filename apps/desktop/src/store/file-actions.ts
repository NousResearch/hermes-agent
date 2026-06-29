import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import { copyTextToClipboard, readDesktopFileText, renameDesktopPath, revealDesktopPath, trashDesktopPath } from '@/lib/desktop-fs'
import { notify, notifyError } from '@/store/notifications'
import { notifyWorkspaceChanged } from '@/store/workspace-events'

// Shared file-row actions for BOTH trees (the file browser + the review/git
// tree): reveal, copy path, rename, delete. Rename/delete route through a single
// dialog set (driven by this atom, rendered once by `FileActionDialogs`) instead
// of one dialog per row. After a successful mutation we bump the workspace tick
// so every git-/fs-mirroring surface refreshes.

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

/** Read a file's text content and copy it to the system clipboard. */
export async function copyFileContent(path: string): Promise<void> {
  try {
    const result = await readDesktopFileText(path)

    if (result.binary) {
      notifyError(new Error('Binary file'), translateNow('common.copyFailed'))

      return
    }

    const text = result.text

    if (!text) {
      notifyError(new Error('File has no readable text content'), translateNow('common.copyFailed'))

      return
    }

    await copyTextToClipboard(text)
    notify({ durationMs: 1500, kind: 'info', message: translateNow('fileMenu.contentCopied') })
  } catch (error) {
    notifyError(error, translateNow('common.copyFailed'))
  }
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
