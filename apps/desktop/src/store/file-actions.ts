import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import {
  copyTextToClipboard,
  createDesktopFolder,
  createDesktopTextFile,
  renameDesktopPath,
  revealDesktopPath,
  trashDesktopPath
} from '@/lib/desktop-fs'
import { notify, notifyError } from '@/store/notifications'
import { notifyWorkspaceChanged } from '@/store/workspace-events'

// Shared file-row actions for BOTH trees (the file browser + the review/git
// tree): reveal, copy path, create, rename, delete. Rename/delete/create route
// through shared UI state where appropriate. After a successful mutation we bump
// the workspace tick so every git-/fs-mirroring surface refreshes.

export interface FileActionTarget {
  isDirectory: boolean
  /** Display name (basename) shown in dialogs. */
  name: string
  /** Absolute path on disk. */
  path: string
}

// Delete/create route through a single dialog set (rendered once). Rename is
// INLINE (VS Code style — an input in the row), driven by `$renamingPath`.
export type FileActionDialog =
  | ({ kind: 'delete' } & FileActionTarget)
  | { kind: 'create-file' | 'create-folder'; parentPath: string }

export const $fileActionDialog = atom<FileActionDialog | null>(null)

export function requestFileDelete(target: FileActionTarget): void {
  $fileActionDialog.set({ kind: 'delete', ...target })
}

export function requestFileCreate(parentPath: string, kind: 'file' | 'folder'): void {
  $fileActionDialog.set({ kind: kind === 'file' ? 'create-file' : 'create-folder', parentPath })
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

/** Strip a `relativeTo` prefix to produce a repo/cwd-relative path. */
export function toRelativePath(path: string, relativeTo: string): string {
  const base = relativeTo.replace(/[\\/]+$/, '')

  if (path === base) {
    return path
  }

  return path.startsWith(`${base}/`) || path.startsWith(`${base}\\`) ? path.slice(base.length + 1) : path
}

// ── Dialog-confirmed mutations (called by FileActionDialogs) ──────────────────

function validateNewEntryName(name: string): string {
  const value = name.trim()

  if (!value || value === '.' || value === '..' || value.includes('/') || value.includes('\\') || value.includes('\0')) {
    throw new Error('Invalid name')
  }

  return value
}

export function childPath(parentPath: string, name: string): string {
  const cleanParent = parentPath.replace(/[\\/]+$/, '')
  const separator = cleanParent.includes('\\') && !cleanParent.includes('/') ? '\\' : '/'

  return `${cleanParent}${separator}${validateNewEntryName(name)}`
}

export async function executeFileCreate(parentPath: string, name: string, content = ''): Promise<string> {
  const result = await createDesktopTextFile(childPath(parentPath, name), content)
  notifyWorkspaceChanged()

  return result.path
}

export async function executeFolderCreate(parentPath: string, name: string): Promise<string> {
  const result = await createDesktopFolder(childPath(parentPath, name))
  notifyWorkspaceChanged()

  return result.path
}

export async function executeFileRename(path: string, newName: string): Promise<void> {
  await renameDesktopPath(path, newName)
  notifyWorkspaceChanged()
}

export async function executeFileDelete(path: string): Promise<void> {
  await trashDesktopPath(path)
  notifyWorkspaceChanged()
}
