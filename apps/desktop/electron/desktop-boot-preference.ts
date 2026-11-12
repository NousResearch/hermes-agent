import { randomUUID } from 'node:crypto'
import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'

export interface DesktopBootPreference {
  profile: string | null
  _migrated?: boolean
}

function validProfile(profile: string | null): boolean {
  return profile === null || /^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile)
}

/** Missing is a fresh install; unreadable state must not be treated as permission to replace it. */
export function readDesktopBootPreference(file: string): DesktopBootPreference | null {
  let text: string

  try {
    text = readFileSync(file, 'utf8')
  } catch (error: unknown) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return null
    }

    throw error
  }

  return parseDesktopBootPreference(text)
}

// Only this disk-JSON boundary inspects primitive representations. Callers get
// a complete preference contract, including preserved fields owned by others.
/* oxlint-disable anti-slop/no-runtime-typeof */
function parseDesktopBootPreference(text: string): DesktopBootPreference {
  const parsed: unknown = JSON.parse(text)

  if (!parsed || typeof parsed !== 'object' || !('profile' in parsed)) {
    throw new Error('Invalid desktop boot preference')
  }

  const profile: unknown = parsed.profile

  if (profile !== null && (typeof profile !== 'string' || !validProfile(profile))) {
    throw new Error('Invalid profile in desktop boot preference')
  }

  const preference: DesktopBootPreference = { ...parsed, profile: typeof profile === 'string' ? profile : null }

  if ('_migrated' in parsed) {
    if (typeof parsed._migrated !== 'boolean') {
      throw new Error('Invalid migration marker in desktop boot preference')
    }

    preference._migrated = parsed._migrated
  }

  return preference
}
/* oxlint-enable anti-slop/no-runtime-typeof */

function updatePreference(
  file: string,
  update: (current: DesktopBootPreference | null) => DesktopBootPreference
): DesktopBootPreference {
  mkdirSync(path.dirname(file), { recursive: true })
  const temporary: string = `${file}.${randomUUID()}.tmp`

  // The destination's single-instance main process owns all preference writes.
  // Keep read, compare, and rename synchronous so its IPC handlers cannot interleave.
  try {
    let current: DesktopBootPreference | null

    try {
      current = readDesktopBootPreference(file)
    } catch (error: unknown) {
      throw new Error('Cannot update an unreadable desktop boot preference', { cause: error })
    }

    const next: DesktopBootPreference = update(current)

    writeFileSync(temporary, JSON.stringify(next, null, 2) + '\n', { encoding: 'utf8', mode: 0o600 })
    const handle: number = openSync(temporary, 'r+')

    try {
      fsyncSync(handle)
    } finally {
      closeSync(handle)
    }

    renameSync(temporary, file)

    if (process.platform !== 'win32') {
      const directory: number = openSync(path.dirname(file), 'r')

      try {
        fsyncSync(directory)
      } finally {
        closeSync(directory)
      }
    }

    return next
  } finally {
    rmSync(temporary, { force: true })
  }
}

export function writeDesktopProfile(file: string, profile: string | null): string | null {
  if (!validProfile(profile)) {
    throw new Error(`Invalid profile name: ${profile}`)
  }

  return updatePreference(file, (current: DesktopBootPreference | null): DesktopBootPreference => {
    const next: DesktopBootPreference = { ...current, profile }

    delete next._migrated

    return next
  }).profile
}
