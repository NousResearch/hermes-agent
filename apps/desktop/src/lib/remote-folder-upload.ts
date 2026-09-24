import { zipSync } from 'fflate'

// ── Remote-mode local folder upload (#120449) ───────────────────────────────
// A desktop on a remote backend (SSH/URL) can't hand the agent a local folder:
// folder chips are `@folder:` refs to paths the backend can't see. Files cross
// as bytes through `file.attach`, so folders cross the same channel zipped —
// the gateway expands the archive server-side (`extract:true`) and answers an
// `@folder:` ref. Same skip-set and caps as the gateway enforces, so a folder
// the client accepts is one the backend will stage.

export interface RemoteFolderDirEntry {
  name: string
  path: string
  isDirectory: boolean
}

export interface RemoteFolderReader {
  readDir(path: string): Promise<{ entries: RemoteFolderDirEntry[]; error?: string }>
  /** Data-URL (or bare base64) file bytes. Null when unreadable. */
  readFileDataUrl(path: string): Promise<string | null>
}

export interface ZippedFolder {
  /** `<folder>.zip` — the gateway derives the staging dir from it. */
  filename: string
  dataUrl: string
  fileCount: number
}

/** Never cross (VCS/dependency trees, macOS resource forks) — mirrors the gateway. */
export const REMOTE_FOLDER_SKIP_DIRS = new Set(['.git', 'node_modules', '__MACOSX'])
// ponytail: fixed caps; raise if legitimate folder uploads hit them.
export const REMOTE_FOLDER_MAX_FILES = 1000
export const REMOTE_FOLDER_MAX_BYTES = 100 * 1024 * 1024

function baseName(folderPath: string): string {
  const trimmed = folderPath.replace(/[/\\]+$/, '')

  return trimmed.split(/[/\\]/).pop() || 'folder'
}

function dataUrlToBytes(dataUrl: string): Uint8Array {
  const base64 = dataUrl.includes(',') ? (dataUrl.split(',').pop() || '') : dataUrl
  const cleaned = base64.replace(/\s+/g, '')
  const binary = atob(cleaned)
  const bytes = new Uint8Array(binary.length)

  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }

  return bytes
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''

  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }

  return btoa(binary)
}

/**
 * Walk a LOCAL folder through the desktop bridge and zip it for `file.attach`
 * (`extract:true`). Throws a plain-Error message the caller surfaces; the
 * gateway re-enforces the same skips/caps server-side.
 */
export async function zipLocalFolderForRemoteUpload(
  folderPath: string,
  reader: RemoteFolderReader,
  limits: { maxFiles?: number; maxBytes?: number } = {}
): Promise<ZippedFolder> {
  const maxFiles = limits.maxFiles ?? REMOTE_FOLDER_MAX_FILES
  const maxBytes = limits.maxBytes ?? REMOTE_FOLDER_MAX_BYTES
  const base = folderPath.replace(/[/\\]+$/, '')
  const prefix = base + '/'
  const altPrefix = base.replace(/\//g, '\\') + '\\'
  const files: Record<string, Uint8Array> = {}
  let fileCount = 0
  let totalBytes = 0
  const stack: string[] = [folderPath]

  while (stack.length > 0) {
    const dir = stack.pop() as string
    let entries: RemoteFolderDirEntry[]

    try {
      ;({ entries } = await reader.readDir(dir))
    } catch {
      throw new Error(`Could not read folder ${baseName(folderPath)}.`)
    }

    for (const entry of entries ?? []) {
      if (entry.isDirectory) {
        if (!REMOTE_FOLDER_SKIP_DIRS.has(entry.name)) {
          stack.push(entry.path)
        }

        continue
      }

      if (entry.name === '.DS_Store') {
        continue
      }

      const rel = (entry.path.startsWith(prefix)
        ? entry.path.slice(prefix.length)
        : entry.path.startsWith(altPrefix)
          ? entry.path.slice(altPrefix.length)
          : entry.name
      ).replace(/\\/g, '/')

      if (!rel || rel.startsWith('/') || rel.split('/').includes('..')) {
        continue
      }

      if (rel.split('/').some(part => REMOTE_FOLDER_SKIP_DIRS.has(part))) {
        continue
      }

      const dataUrl = await reader.readFileDataUrl(entry.path)

      if (!dataUrl) {
        throw new Error(`Could not read ${entry.name}.`)
      }

      const bytes = dataUrlToBytes(dataUrl)
      totalBytes += bytes.length

      if (totalBytes > maxBytes) {
        throw new Error(
          `Folder is too large to upload to the remote gateway (max ${Math.floor(maxBytes / (1024 * 1024))} MB).`
        )
      }

      // First write wins: same relative path twice (case-juggling bridges) keeps one copy.
      if (!(rel in files)) {
        files[rel] = bytes
        fileCount += 1

        if (fileCount > maxFiles) {
          throw new Error(`Folder has too many files to upload to the remote gateway (max ${maxFiles}).`)
        }
      }
    }
  }

  if (fileCount === 0) {
    throw new Error(`Folder ${baseName(folderPath)} has no files to upload.`)
  }

  const zipped = zipSync(files)

  return {
    filename: `${baseName(folderPath)}.zip`,
    dataUrl: `data:application/zip;base64,${bytesToBase64(zipped)}`,
    fileCount
  }
}

/**
 * True when the folder root reads through the LOCAL bridge. A backend-side
 * path (in-app drag from a remote project tree) fails here — the gateway
 * resolves those directly, so callers pass them through instead of zipping.
 */
export async function canReadLocalFolder(reader: RemoteFolderReader, folderPath: string): Promise<boolean> {
  try {
    const result = await reader.readDir(folderPath)

    return Boolean(result) && !result.error
  } catch {
    return false
  }
}
