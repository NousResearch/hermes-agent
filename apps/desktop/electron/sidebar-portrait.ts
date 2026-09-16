import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Mirrors gentle-pi's private ~/.pi/gentle-ai/sidebar-portrait.json contract: a
// normalized grayscale grid, no pixel colors. Read-only: the renderer never
// supplies a path, so there is nothing to traverse or sanitize.

export interface SidebarPortraitPayload {
  version: 1
  width: number
  height: number
  luminance: number[]
}

const SIDEBAR_PORTRAIT_MAX_PIXELS = 1_000_000
// Bound the read before parsing, like every other file-reading IPC handler:
// the portrait grid is tiny (120x120 => ~46 KB), so anything past a few MB is
// malformed or hostile and must not reach JSON.parse.
const SIDEBAR_PORTRAIT_MAX_BYTES = 4 * 1024 * 1024

export function sidebarPortraitPath(): string {
  return path.join(os.homedir(), '.pi', 'gentle-ai', 'sidebar-portrait.json')
}

export function isValidSidebarPortrait(value: unknown): value is SidebarPortraitPayload {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const candidate = value as Partial<SidebarPortraitPayload>
  if (candidate.version !== 1 || !Number.isInteger(candidate.width) || !Number.isInteger(candidate.height)) {
    return false
  }

  const width = candidate.width ?? 0
  const height = candidate.height ?? 0
  if (width <= 0 || height <= 0 || width * height > SIDEBAR_PORTRAIT_MAX_PIXELS) {
    return false
  }

  if (!Array.isArray(candidate.luminance) || candidate.luminance.length !== width * height) {
    return false
  }

  return candidate.luminance.every(entry => Number.isInteger(entry) && entry >= 0 && entry <= 255)
}

/** Load the user-owned portrait; a missing or malformed file is not fatal. */
export async function readSidebarPortrait(): Promise<SidebarPortraitPayload | null> {
  try {
    const portraitPath = sidebarPortraitPath()
    const stat = await fs.promises.stat(portraitPath)
    if (!stat.isFile() || stat.size > SIDEBAR_PORTRAIT_MAX_BYTES) {
      return null
    }

    const parsed: unknown = JSON.parse(await fs.promises.readFile(portraitPath, 'utf8'))
    return isValidSidebarPortrait(parsed) ? parsed : null
  } catch {
    return null
  }
}
