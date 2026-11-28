// Core's host-only marker for a profile a plugin runs somewhere other than this
// host's backend (for example in a sandbox behind the plugin's launcher). One
// file per profile: `<hermes root>/profiles_run_elsewhere/<profile>.json`, whose
// presence is the marker and whose `{"plugin": "<id>"}` only names the owner
// (`hermes_cli/profiles_run_elsewhere.py`).
//
// The desktop reads it to fail closed: while a profile is marked, nothing on
// this host may start or serve it, so a profile whose route to its launcher is
// missing is refused instead of falling back to a local `gateway ensure`.
import fs from 'node:fs'
import path from 'node:path'

export const PROFILES_RUN_ELSEWHERE_DIR = 'profiles_run_elsewhere'

const PROFILE_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

export interface RunElsewhereMarker {
  /** The plugin that runs the profile, or null when the marker is unreadable. */
  plugin: null | string
}

function errorCode(error: unknown): string {
  return String((error as NodeJS.ErrnoException | null)?.code || '')
}

/** The profile ids the folder marks, or the error that marks every profile. */
function markedProfiles(dir: string): Error | Set<string> {
  try {
    return new Set(
      fs
        .readdirSync(dir)
        .filter(name => name.endsWith('.json'))
        .map(name => name.slice(0, -'.json'.length))
    )
  } catch (error) {
    // Missing, or not listable (the backend a plugin runs elsewhere is denied the folder): nothing is
    // marked. Any other error, a file in the folder's place included, marks every profile, as in core.
    return ['ENOENT', 'EACCES', 'EPERM'].includes(errorCode(error)) ? new Set() : (error as Error)
  }
}

/**
 * The marker for `profile`, or null when nothing marks it. Fails closed the way
 * core does: a marker that cannot be read or parsed still marks the profile
 * (its plugin is just unknown), and a marker folder that fails to list for any
 * reason other than missing or permission denied (a file in its place, an I/O
 * error) marks every profile.
 */
export function readRunElsewhereMarker(
  hermesRoot: string,
  profile: null | string | undefined
): null | RunElsewhereMarker {
  const id = String(profile ?? '').trim().toLowerCase() || 'default'

  if (id !== 'default' && !PROFILE_ID_RE.test(id)) {
    return null
  }

  const dir = path.join(hermesRoot, PROFILES_RUN_ELSEWHERE_DIR)
  const marked = markedProfiles(dir)

  if (marked instanceof Error) {
    return { plugin: null }
  }

  if (!marked.has(id)) {
    return null
  }

  try {
    const parsed = JSON.parse(fs.readFileSync(path.join(dir, `${id}.json`), 'utf8').replace(/^\uFEFF/, ''))
    const plugin = typeof parsed?.plugin === 'string' ? parsed.plugin.trim() : ''

    return { plugin: plugin || null }
  } catch {
    return { plugin: null }
  }
}

/** Refuse a local start of a profile core marks as run elsewhere: this app has no route to its backend. */
export function assertProfileMayRunLocally(hermesRoot: string, profile: null | string | undefined): void {
  const name = String(profile ?? '').trim() || 'default'
  const marker = readRunElsewhereMarker(hermesRoot, name)

  if (!marker) {
    return
  }

  const runner = marker.plugin ? `runs through the ${marker.plugin} plugin` : 'runs through a plugin'

  throw new Error(
    `Profile "${name}" ${runner}, but this app has no connection to it. Hermes won't start it on this computer.`
  )
}
