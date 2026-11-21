import { connectionScopedAtom } from '@/lib/connection-scoped'
import { cleanPath, comparisonPath } from '@/lib/path-compare'
import { Codecs } from '@/lib/persisted'

import {
  type FileSortKey,
  fileSortKey,
  type FileSortPreference,
  isFileSortKey,
  resolveFolderSort
} from './sort'

// "Show gitignored files" is a per-project preference: a repo that keeps a
// second checkout, a build output or an env-specific folder out of git is
// exactly the repo whose owner wants to browse it, while every other project
// should stay clean. Storing one global flag makes that choice leak across
// projects, so the roots opted in are persisted as a list.
//
// The scope is CONNECTION + project root: the same absolute path can name two
// different repos on two backends, and the connection scope keeps those apart
// (see lib/connection-scoped). Presence in the list means "show"; absence is
// the default, so an untouched install persists nothing at all.
const SHOW_IGNORED_KEY = 'hermes.desktop.files.showIgnored'

export const $showIgnoredRoots = connectionScopedAtom<string[]>(SHOW_IGNORED_KEY, [], Codecs.stringArray)

// Per-folder Files sort (`name:asc` | `mtime:desc` | …). Keyed by folder path
// under the active connection so two backends with the same absolute path stay
// independent. Empty map = no overrides (media folders still default newest-first).
const FOLDER_SORT_KEY = 'hermes.desktop.files.folderSort'

function sanitizeFolderSortMap(value: unknown): Record<string, FileSortKey> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const next: Record<string, FileSortKey> = {}

  for (const [path, raw] of Object.entries(value as Record<string, unknown>)) {
    if (!path || !isFileSortKey(raw)) {
      continue
    }

    next[path] = raw
  }

  return next
}

export const $folderSortPrefs = connectionScopedAtom<Record<string, FileSortKey>>(
  FOLDER_SORT_KEY,
  {},
  Codecs.json(sanitizeFolderSortMap)
)

/** Storage/comparison spelling for a project root — backends, pickers and
 *  session rows disagree on separator and drive-letter case. */
function rootKey(root: string): string {
  return comparisonPath(cleanPath(root))
}

/** Folder path spelling used as the sort-pref map key. */
function folderKey(folderPath: string): string {
  return comparisonPath(cleanPath(folderPath))
}

/** The single resolver every reader goes through, so the tree, its lazy child
 *  reads and its revalidations always agree on one answer for a root. */
export function showsIgnoredFiles(root: string): boolean {
  if (!root) {
    return false
  }

  return $showIgnoredRoots.get().includes(rootKey(root))
}

/** Set the preference for one project root. Returns whether it actually
 *  changed, so the caller can skip an expensive reload on a no-op. */
export function setShowIgnoredFiles(root: string, show: boolean): boolean {
  if (!root) {
    return false
  }

  const key = rootKey(root)
  const current = $showIgnoredRoots.get()
  const has = current.includes(key)

  if (has === show) {
    return false
  }

  $showIgnoredRoots.set(show ? [...current, key] : current.filter(item => item !== key))

  return true
}

/** Effective sort for a folder (explicit pref → media heuristic → name asc). */
export function folderSortPreference(folderPath: string): FileSortPreference {
  if (!folderPath) {
    return resolveFolderSort('', $folderSortPrefs.get())
  }

  return resolveFolderSort(folderKey(folderPath), $folderSortPrefs.get())
}

/**
 * Remember a folder's sort. Returns whether the stored map changed so callers
 * can skip a tree rebuild on a no-op.
 */
export function setFolderSortPreference(folderPath: string, pref: FileSortPreference): boolean {
  if (!folderPath) {
    return false
  }

  const key = folderKey(folderPath)
  const nextKey = fileSortKey(pref)
  const current = $folderSortPrefs.get()

  if (current[key] === nextKey) {
    return false
  }

  $folderSortPrefs.set({ ...current, [key]: nextKey })

  return true
}
