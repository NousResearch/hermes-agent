/**
 * desktop-storage.ts
 *
 * Read-only disk-usage scan of HERMES_HOME for Settings > Storage. Electron-free
 * so Vitest can drive the registered IPC handler. Never deletes anything, never
 * follows symlinks/junctions (a link out of HERMES_HOME must not be counted or
 * walked), and stops at a time/entry budget so a huge tree cannot hang the app.
 */

import type { Dir } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'

export interface StorageEntry {
  name: string
  bytes: number
  kind: 'dir' | 'file'
}

export interface StorageSummary {
  root: string
  totalBytes: number
  /** Largest first. */
  entries: StorageEntry[]
  /** True when the time or entry budget stopped the scan, so sizes are lower bounds. */
  truncated: boolean
}

export interface StorageScanOptions {
  maxMs?: number
  maxEntries?: number
}

const DEFAULT_MAX_MS = 20_000
const DEFAULT_MAX_ENTRIES = 5_000_000
// Folders read at once, and files stat'ed at once per folder. Together they cap
// the reads in flight, so a huge flat folder cannot queue millions of them.
const DIR_BATCH = 32
const FILE_CHUNK = 32

export async function scanStorage(root: string, options: StorageScanOptions = {}): Promise<StorageSummary> {
  const deadline: number = Date.now() + (options.maxMs ?? DEFAULT_MAX_MS)
  let remaining: number = options.maxEntries ?? DEFAULT_MAX_ENTRIES
  let truncated = false

  // Checked before every entry is admitted, so neither budget can be overshot by
  // more than one in-flight chunk, however flat or deep the tree is.
  const overBudget = (): boolean => {
    if (remaining <= 0 || Date.now() > deadline) {
      truncated = true

      return true
    }

    return false
  }

  const sizeOf = async (file: string): Promise<number> => fs.lstat(file).then((st): number => st.size, (): number => 0)

  const sumSizes = async (files: string[]): Promise<number> =>
    (await Promise.all(files.map(sizeOf))).reduce((sum: number, n: number): number => sum + n, 0)

  // Folders still to read, across every top-level entry: one shared queue, so
  // total concurrency is bounded by DIR_BATCH instead of growing per entry.
  const queue: Array<{ dir: string; owner: number }> = []
  const totals: number[] = []

  async function readDir(dir: string, owner: number): Promise<void> {
    let handle: Dir

    try {
      handle = await fs.opendir(dir)
    } catch {
      return
    }

    const files: string[] = []

    try {
      // Entries are streamed, so a folder with millions of files stops at the budget.
      for await (const child of handle) {
        if (overBudget()) {
          break
        }

        remaining -= 1

        // Dirent reports links as links (not their target), so this skips them.
        if (child.isSymbolicLink()) {
          continue
        }

        const full: string = path.join(dir, child.name)

        if (child.isDirectory()) {
          queue.push({ dir: full, owner })
        } else if (child.isFile()) {
          files.push(full)

          if (files.length >= FILE_CHUNK) {
            totals[owner] += await sumSizes(files.splice(0))
          }
        }
      }
    } finally {
      // Leaving the loop early already closes the handle; closing twice throws.
      await handle.close().catch((): void => undefined)
    }

    totals[owner] += await sumSizes(files)
  }

  // The scanned folder itself is streamed under the same budget.
  const entries: StorageEntry[] = []
  const rootFiles: Array<{ name: string; full: string }> = []
  const top: Dir = await fs.opendir(root)

  try {
    for await (const child of top) {
      if (overBudget()) {
        break
      }

      remaining -= 1

      if (child.isSymbolicLink()) {
        continue
      }

      const full: string = path.join(root, child.name)

      if (child.isDirectory()) {
        totals.push(0)
        entries.push({ name: child.name, bytes: 0, kind: 'dir' })
        queue.push({ dir: full, owner: totals.length - 1 })
      } else if (child.isFile()) {
        rootFiles.push({ name: child.name, full })
      }
    }
  } finally {
    await top.close().catch((): void => undefined)
  }

  // Folder entries are in `entries` in the same order as their `totals` slot.
  const dirEntries: StorageEntry[] = entries.filter((e: StorageEntry): boolean => e.kind === 'dir')

  for (let i = 0; i < rootFiles.length; i += FILE_CHUNK) {
    const chunk = rootFiles.slice(i, i + FILE_CHUNK)
    const sizes: number[] = await Promise.all(chunk.map((f): Promise<number> => sizeOf(f.full)))

    chunk.forEach((f, j): number => entries.push({ name: f.name, bytes: sizes[j], kind: 'file' }))
  }

  while (queue.length > 0) {
    if (overBudget()) {
      break
    }

    const batch = queue.splice(0, DIR_BATCH)

    await Promise.all(batch.map(({ dir, owner }): Promise<void> => readDir(dir, owner)))
  }

  dirEntries.forEach((e: StorageEntry, i: number): void => {
    e.bytes = totals[i]
  })

  entries.sort((a: StorageEntry, b: StorageEntry): number => b.bytes - a.bytes)

  return {
    root,
    totalBytes: entries.reduce((sum: number, e: StorageEntry): number => sum + e.bytes, 0),
    entries,
    truncated
  }
}

export interface StorageRevealResult {
  ok: boolean
}

export interface DesktopStorageIpcDeps {
  ipcMain: {
    handle: (channel: string, handler: (event: unknown, payload?: unknown) => Promise<unknown>) => void
  }
  hermesHome: () => string
  /** Show an item selected in its parent folder. */
  showItemInFolder: (target: string) => void
}

/**
 * Resolves renderer-supplied folder names to a path inside the Hermes home. The
 * renderer never sends a path: every segment must be a plain name (no
 * separators, no `..`) and must not be a link, so this cannot be pointed
 * outside HERMES_HOME. Returns null if any segment is refused or missing.
 */
async function resolveTrail(root: string, segments: unknown): Promise<string | null> {
  if (!Array.isArray(segments)) {
    return null
  }

  let current: string = root

  for (const name of segments) {
    if (typeof name !== 'string' || !name || name !== path.basename(name) || name === '.' || name === '..') {
      return null
    }

    current = path.join(current, name)

    try {
      if ((await fs.lstat(current)).isSymbolicLink()) {
        return null
      }
    } catch {
      return null
    }
  }

  return current
}

function trailOf(payload: unknown): unknown {
  return payload && typeof payload === 'object' && 'trail' in payload ? payload.trail : []
}

export function registerDesktopStorageIpc({ ipcMain, hermesHome, showItemInFolder }: DesktopStorageIpcDeps): void {
  // Scan the folder at `trail` (folder names below HERMES_HOME; empty = the home itself).
  ipcMain.handle('hermes:storage:summary', async (_event: unknown, payload?: unknown): Promise<StorageSummary> => {
    const target: string | null = await resolveTrail(hermesHome(), trailOf(payload))

    if (!target) {
      throw new Error('storage: folder not available')
    }

    return scanStorage(target)
  })

  // Show an entry of the folder at `trail` selected in the OS file manager.
  ipcMain.handle(
    'hermes:storage:reveal',
    async (_event: unknown, payload?: unknown): Promise<StorageRevealResult> => {
      const name: unknown = payload && typeof payload === 'object' && 'name' in payload ? payload.name : undefined
      const trail: unknown = trailOf(payload)
      const target: string | null = Array.isArray(trail) ? await resolveTrail(hermesHome(), [...trail, name]) : null

      if (!target) {
        return { ok: false }
      }

      showItemInFolder(target)

      return { ok: true }
    }
  )
}
