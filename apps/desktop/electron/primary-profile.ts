import fs from 'node:fs'
import path from 'node:path'

import { normalizeDesktopProfile } from './profile-name'

interface EffectivePrimaryProfileOptions {
  desktopProfile: unknown
  hermesHome: string
  readFile?: (file: string, encoding: BufferEncoding) => string
}

function resolveConfiguredProfile(value: unknown, source: string): null | string {
  if (value == null) {
    return null
  }

  if (typeof value !== 'string') {
    throw new Error(`Invalid profile name in ${source}: ${JSON.stringify(value)}`)
  }

  const raw = value.trim()

  if (!raw) {
    return null
  }

  const profile = normalizeDesktopProfile(raw)

  if (!profile) {
    throw new Error(`Invalid profile name in ${source}: ${JSON.stringify(raw)}`)
  }

  return profile
}

/** Parse the persisted Desktop preference without hiding malformed state. */
export function parseDesktopProfilePreference(raw: string): null | string {
  const parsed: unknown = JSON.parse(raw)

  if (!parsed || typeof parsed !== 'object' || !('profile' in parsed)) {
    return null
  }

  return resolveConfiguredProfile((parsed as { profile: unknown }).profile, 'Desktop preference')
}

/** Keep the resolved owner stable until the primary backend is torn down. */
export function createPrimaryProfileOwner(resolve: () => string) {
  let current: null | string = null

  return {
    get() {
      current ??= resolve()

      return current
    },
    reset() {
      current = null
    }
  }
}

/** Mirror the CLI's profile precedence for an unpinned Desktop backend launch. */
export function resolveEffectivePrimaryProfile({
  desktopProfile,
  hermesHome,
  readFile = (file, encoding) => fs.readFileSync(file, encoding)
}: EffectivePrimaryProfileOptions): string {
  const explicit = resolveConfiguredProfile(desktopProfile, 'Desktop preference')

  if (explicit) {
    return explicit
  }

  const resolvedHome = path.resolve(hermesHome)

  if (path.basename(path.dirname(resolvedHome)) === 'profiles') {
    return resolveConfiguredProfile(path.basename(resolvedHome), 'HERMES_HOME') ?? 'default'
  }

  try {
    return (
      resolveConfiguredProfile(readFile(path.join(resolvedHome, 'active_profile'), 'utf8'), 'active_profile') ??
      'default'
    )
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') {
      return 'default'
    }

    throw error
  }
}
