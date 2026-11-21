/** Files panel sort: field + direction, per-folder defaults for generated media. */

export type FileSortField = 'mtime' | 'name' | 'size'
export type FileSortDirection = 'asc' | 'desc'

export interface FileSortPreference {
  direction: FileSortDirection
  field: FileSortField
}

/** Stable key form stored in prefs (`name:asc`, `mtime:desc`, …). */
export type FileSortKey = `${FileSortField}:${FileSortDirection}`

export const FILE_SORT_KEYS: readonly FileSortKey[] = [
  'name:asc',
  'name:desc',
  'mtime:desc',
  'mtime:asc',
  'size:desc',
  'size:asc'
] as const

export const DEFAULT_FILE_SORT: FileSortPreference = { direction: 'asc', field: 'name' }
/** Newest-first date sort — default for renders / screenshots / audio caches. */
export const MEDIA_FOLDER_FILE_SORT: FileSortPreference = { direction: 'desc', field: 'mtime' }

/** Basename tokens that usually hold generated media (case-insensitive). */
const MEDIA_FOLDER_BASENAMES = new Set([
  'audio',
  'audio_cache',
  'cache',
  'media',
  'output',
  'outputs',
  'recording',
  'recordings',
  'render',
  'renders',
  'screen',
  'screenshots',
  'shots',
  'video',
  'videos',
  'voice',
  'voice-memos',
  'voice_memos'
])

export function parseFileSortKey(raw: null | string | undefined): FileSortPreference | null {
  if (!raw || typeof raw !== 'string') {
    return null
  }

  const [field, direction] = raw.split(':') as [string, string]

  if (
    (field !== 'name' && field !== 'mtime' && field !== 'size') ||
    (direction !== 'asc' && direction !== 'desc')
  ) {
    return null
  }

  return { direction, field }
}

export function fileSortKey(pref: FileSortPreference): FileSortKey {
  return `${pref.field}:${pref.direction}`
}

export function isFileSortKey(value: unknown): value is FileSortKey {
  return typeof value === 'string' && (FILE_SORT_KEYS as readonly string[]).includes(value)
}

/** Last path segment, tolerant of `/` and `\\`. */
export function folderBasename(folderPath: string): string {
  const trimmed = folderPath.replace(/[\\/]+$/, '')
  const parts = trimmed.split(/[\\/]/).filter(Boolean)

  return parts.length > 0 ? parts[parts.length - 1]! : ''
}

export function isMediaFolderPath(folderPath: string): boolean {
  const base = folderBasename(folderPath).toLowerCase()

  if (!base) {
    return false
  }

  if (MEDIA_FOLDER_BASENAMES.has(base)) {
    return true
  }

  // Compound names: `work-renders`, `cache-audio`, `hf-renders`.
  return [...MEDIA_FOLDER_BASENAMES].some(token => base === token || base.endsWith(`-${token}`) || base.endsWith(`_${token}`))
}

/**
 * Resolve sort for a folder: explicit pref wins, else media heuristic → mtime desc,
 * else name ascending (today's default).
 */
export function resolveFolderSort(
  folderPath: string,
  prefs: Record<string, string> | null | undefined
): FileSortPreference {
  const explicit = parseFileSortKey(prefs?.[folderPath])

  if (explicit) {
    return explicit
  }

  return isMediaFolderPath(folderPath) ? MEDIA_FOLDER_FILE_SORT : DEFAULT_FILE_SORT
}

export interface SortableFileEntry {
  isDirectory: boolean
  mtimeMs?: number
  name: string
  size?: number
}

function numericField(value: number | undefined, missing: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : missing
}

/**
 * Directories stay above files (explorer convention). Within each group, apply
 * the requested field/direction; ties break on name ascending.
 */
export function compareFileEntries(a: SortableFileEntry, b: SortableFileEntry, pref: FileSortPreference): number {
  const dirDelta = Number(b.isDirectory) - Number(a.isDirectory)

  if (dirDelta !== 0) {
    return dirDelta
  }

  const dir = pref.direction === 'asc' ? 1 : -1
  let primary = 0

  if (pref.field === 'name') {
    primary = a.name.localeCompare(b.name)
  } else if (pref.field === 'mtime') {
    // Missing mtime sorts as oldest so known dates stay useful.
    primary = numericField(a.mtimeMs, 0) - numericField(b.mtimeMs, 0)
  } else {
    // Missing size sorts as 0.
    primary = numericField(a.size, 0) - numericField(b.size, 0)
  }

  if (primary !== 0) {
    return primary * dir
  }

  return a.name.localeCompare(b.name)
}

export function sortFileEntries<T extends SortableFileEntry>(entries: readonly T[], pref: FileSortPreference): T[] {
  return [...entries].sort((a, b) => compareFileEntries(a, b, pref))
}
