// Canvas library on disk + the hosted editor URL.
//
// Hermes owns `.pen` files under `$HERMES_HOME/pens`. The editor is
// app.pen.dev/new?embed; storage and tools go over pen/web-bridge.ts.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { ensurePenEmbedUrl } from './pen/embed-url'

const PEN_WEB_EDITOR_DEFAULT_URL = 'https://app.pen.dev/new?embed'

/** Hosted editor URL. `HERMES_PEN_WEB_URL` overrides for staging. */
export function penWebEditorUrl(): string {
  return ensurePenEmbedUrl(process.env.HERMES_PEN_WEB_URL || PEN_WEB_EDITOR_DEFAULT_URL)
}

export function penLibraryRoot(): string {
  const home = process.env.HERMES_HOME || path.join(os.homedir(), '.hermes')

  return path.join(home, 'pens')
}

export interface PenLibraryEntry {
  /** Absolute path to the .pen file. */
  path: string
  /** Folder holding the .pen (deleting a canvas removes this). */
  folder: string
  name: string
  modifiedAt: number
  size: number
  /** Rendered canvas preview (preview.png beside the .pen), when one exists. */
  previewPath: null | string
}

/** Every canvas in the library, newest first. One folder per canvas. */
export function listPenLibrary(): PenLibraryEntry[] {
  const root = penLibraryRoot()
  const entries: PenLibraryEntry[] = []

  let folders: string[]

  try {
    folders = fs.readdirSync(root)
  } catch {
    return entries
  }

  for (const folder of folders) {
    const dir = path.join(root, folder)

    let files: string[]

    try {
      if (!fs.statSync(dir).isDirectory()) {
        continue
      }

      files = fs.readdirSync(dir)
    } catch {
      continue
    }

    for (const file of files) {
      if (!file.endsWith('.pen')) {
        continue
      }

      const full = path.join(dir, file)

      try {
        const stat = fs.statSync(full)
        const preview = path.join(dir, 'preview.png')

        entries.push({
          path: full,
          folder: dir,
          name: file.replace(/\.pen$/, ''),
          modifiedAt: stat.mtimeMs,
          size: stat.size,
          previewPath: fs.existsSync(preview) ? preview : null
        })
      } catch {
        // A canvas being written right now — skip it this pass.
      }
    }
  }

  return entries.sort((a, b) => b.modifiedAt - a.modifiedAt)
}

/** Reserve a path in the library for a new canvas. Never overwrites. */
export function penLibraryPathFor(name: string): string {
  const safe =
    name
      .replace(/\.pen$/i, '')
      .replace(/[/\\:*?"<>|]/g, '-')
      .trim() || 'Untitled'

  const root = penLibraryRoot()

  for (let n = 0; n < 1000; n += 1) {
    const candidate = n === 0 ? safe : `${safe} ${n + 1}`
    const folder = path.join(root, candidate)

    if (!fs.existsSync(folder)) {
      return path.join(folder, `${candidate}.pen`)
    }
  }

  return path.join(root, `${safe} ${Date.now()}`, `${safe}.pen`)
}

/** Delete a canvas — the whole folder. Refuses anything outside the library. */
export function deletePenFromLibrary(target: string): boolean {
  const root = penLibraryRoot()
  const folder = target.endsWith('.pen') ? path.dirname(target) : target
  const resolved = path.resolve(folder)

  if (!resolved.startsWith(path.resolve(root) + path.sep)) {
    return false
  }

  try {
    fs.rmSync(resolved, { recursive: true, force: true })

    return true
  } catch {
    return false
  }
}

/** Rename a canvas: the .pen and its folder move together. Returns the new path. */
export function renamePenInLibrary(target: string, nextName: string): null | string {
  const root = path.resolve(penLibraryRoot())
  const current = path.resolve(target)

  if (!current.startsWith(root + path.sep) || !fs.existsSync(current)) {
    return null
  }

  const destination = penLibraryPathFor(nextName)

  try {
    fs.mkdirSync(path.dirname(destination), { recursive: true })
    fs.renameSync(current, destination)

    const oldFolder = path.dirname(current)

    try {
      if (fs.readdirSync(oldFolder).length === 0) {
        fs.rmdirSync(oldFolder)
      }
    } catch {
      // Leftover siblings — leave the folder alone.
    }

    return destination
  } catch {
    return null
  }
}
