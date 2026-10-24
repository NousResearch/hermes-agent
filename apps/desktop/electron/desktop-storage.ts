/**
 * desktop-storage.ts
 *
 * Read-only disk-usage scan of HERMES_HOME for Settings > Storage. Electron-free
 * so Vitest can drive the registered IPC handler. Never deletes anything, never
 * follows symlinks/junctions (a link out of HERMES_HOME must not be counted or
 * walked), and stops at a time/entry budget so a huge tree cannot hang the app.
 */

import type { Dirent } from 'node:fs'
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
const DIR_BATCH = 32

export async function scanStorage(root: string, options: StorageScanOptions = {}): Promise<StorageSummary> {
  const deadline: number = Date.now() + (options.maxMs ?? DEFAULT_MAX_MS)
  let remaining: number = options.maxEntries ?? DEFAULT_MAX_ENTRIES
  let truncated = false

  // Directories are walked in parallel batches and file sizes are read
  // concurrently: serial lstat calls are far too slow on Windows.
  async function sizeOf(target: string): Promise<number> {
    let total = 0
    let frontier: string[] = [target]

    while (frontier.length > 0) {
      const batch: string[] = frontier.splice(0, DIR_BATCH)

      if (remaining <= 0 || Date.now() > deadline) {
        truncated = true

        return total
      }

      const sizes: number[] = await Promise.all(
        batch.map(async (dir: string): Promise<number> => {
          let children: Dirent[]

          try {
            children = await fs.readdir(dir, { withFileTypes: true })
          } catch {
            return 0
          }

          remaining -= children.length
          const files: string[] = []

          for (const child of children) {
            // Dirent reports links as links (not their target), so this skips them.
            if (child.isSymbolicLink()) {
              continue
            }

            const full: string = path.join(dir, child.name)

            if (child.isDirectory()) {
              frontier.push(full)
            } else if (child.isFile()) {
              files.push(full)
            }
          }

          const fileSizes: number[] = await Promise.all(
            files.map((file: string): Promise<number> => fs.lstat(file).then((st): number => st.size, (): number => 0))
          )

          return fileSizes.reduce((sum: number, n: number): number => sum + n, 0)
        })
      )

      total += sizes.reduce((sum: number, n: number): number => sum + n, 0)
    }

    return total
  }

  const top: Dirent[] = await fs.readdir(root, { withFileTypes: true })

  const settled: Array<StorageEntry | null> = await Promise.all(
    top.map(async (item: Dirent): Promise<StorageEntry | null> => {
      if (item.isSymbolicLink()) {
        return null
      }

      const full: string = path.join(root, item.name)

      if (item.isDirectory()) {
        return { name: item.name, bytes: await sizeOf(full), kind: 'dir' }
      }

      if (item.isFile()) {
        try {
          return { name: item.name, bytes: (await fs.lstat(full)).size, kind: 'file' }
        } catch {
          // Ignore files that disappear mid-scan.
        }
      }

      return null
    })
  )

  const entries: StorageEntry[] = settled.filter((e: StorageEntry | null): e is StorageEntry => e !== null)

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
