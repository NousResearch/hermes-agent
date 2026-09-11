import * as fs from 'node:fs'
import * as path from 'node:path'

/**
 * Last-used directory for the native open dialogs (#92925). Electron is
 * authoritative for machine-side facts, so this lives in the main process —
 * the renderer's `defaultPath` (composer cwd, #91074's Downloads default)
 * only applies when the user hasn't already shown us where their files live.
 */

/**
 * The defaultPath the next dialog should open at: the caller's explicit
 * choice when given (it expresses current intent — the composer's working
 * directory), else the remembered last-used directory when it still exists,
 * else undefined (OS default). Pure so the precedence policy is testable.
 */
export function nextPickerDefaultPath(
  explicitDefault: string | undefined,
  rememberedDir: string | undefined,
  directoryExists: (dir: string) => boolean
): string | undefined {
  if (explicitDefault) {
    return explicitDefault
  }

  if (rememberedDir && directoryExists(rememberedDir)) {
    return rememberedDir
  }

  return undefined
}

/**
 * Directory worth remembering after a successful pick: the parent of the
 * first selected path. Returns null for empty/canceled results.
 */
export function dirToRemember(filePaths: unknown): string | null {
  const first = Array.isArray(filePaths) ? filePaths[0] : null

  if (typeof first !== 'string' || !first.trim()) {
    return null
  }

  const dir = path.dirname(first)

  return dir && dir !== '.' ? dir : null
}

/** Read the persisted last-used directory. Corrupt/missing file → undefined. */
export function readLastPickerDir(statePath: string): string | undefined {
  try {
    const parsed = JSON.parse(fs.readFileSync(statePath, 'utf8'))
    const dir = typeof parsed?.lastDir === 'string' ? parsed.lastDir.trim() : ''

    return dir || undefined
  } catch {
    return undefined
  }
}

/** Persist the last-used directory. Best-effort: a failed write only costs
 *  the memory, never the dialog itself. */
export function writeLastPickerDir(statePath: string, dir: string): void {
  try {
    fs.mkdirSync(path.dirname(statePath), { recursive: true })

    const tmp = `${statePath}.${process.pid}.tmp`

    fs.writeFileSync(tmp, JSON.stringify({ lastDir: dir }, null, 2))
    fs.renameSync(tmp, statePath)
  } catch {
    /* remembering the directory is best-effort */
  }
}
