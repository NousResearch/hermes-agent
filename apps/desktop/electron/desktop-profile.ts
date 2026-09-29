import fs from 'node:fs'
import path from 'node:path'

import { profileNameFromDeleteRequest } from './profile-delete-routing'
import { profileRenameFromRequest, type ProfileRenameRequest } from './profile-rename-routing'
import type { WindowConnectionRoute } from './window-connection-route'

export interface DesktopProfileRoute {
  connectionId: null | string
  profile: string
}

export const DESKTOP_PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

/**
 * Normalize a caller-supplied profile name before it can select or spawn a
 * backend. An absent / blank value returns null so the caller resolves the
 * primary profile; anything else must match DESKTOP_PROFILE_NAME_RE.
 *
 * This is the one validation used by every renderer-reachable profile path
 * (hermes:connection, hermes:gateway:ws-url, hermes:api, registry dials), not
 * only the persisted profile setter. The name ends up as the argument after
 * `--profile` on a `hermes serve` spawn, and on Windows that spawn may run with
 * `shell: true` for a .cmd/.bat shim, so this boundary is what keeps shell
 * metacharacters, whitespace and flag-like values out of the child argv
 * (GHSA-84j8-xmx8-jghv).
 */
export function requireDesktopProfileName(value: unknown): null | string {
  if (value === null || value === undefined) {
    return null
  }

  if (typeof value !== 'string') {
    throw new Error('Invalid profile name.')
  }

  const profile = value.trim()

  if (!profile) {
    return null
  }

  if (!DESKTOP_PROFILE_NAME_RE.test(profile)) {
    throw new Error(`Invalid profile name: ${JSON.stringify(profile.slice(0, 80))}`)
  }

  return profile
}

/**
 * Spawn-boundary assertion: the exact string about to be placed in a backend
 * child's argv must already be a canonical profile name. Unlike
 * requireDesktopProfileName this never normalizes; a value that needed
 * trimming is a bug upstream, not something to repair here.
 */
export function assertSpawnableProfileName(profile: unknown): string {
  if (typeof profile !== 'string' || !DESKTOP_PROFILE_NAME_RE.test(profile)) {
    throw new Error(
      `Refusing to start a backend for an invalid profile name: ${JSON.stringify(String(profile).slice(0, 80))}`
    )
  }

  return profile
}

export function requireDesktopProfileRoute(value: unknown): DesktopProfileRoute {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('A connection and profile route is required.')
  }

  const { connectionId, profile } = value as Record<string, unknown>

  if (typeof profile !== 'string' || !DESKTOP_PROFILE_NAME_RE.test(profile)) {
    throw new Error('Invalid profile name.')
  }

  if (
    connectionId !== null &&
    (typeof connectionId !== 'string' || !connectionId || connectionId.trim() !== connectionId)
  ) {
    throw new Error('Invalid connection id.')
  }

  return { connectionId: connectionId as null | string, profile }
}

// Explicit peer intent wins; a generic New Window inherits its IPC sender,
// never whichever other window happened to gain OS focus while IPC was queued.
export function resolveDesktopWindowRoute(
  explicit: unknown,
  source: WindowConnectionRoute | null,
  fallback: DesktopProfileRoute
): DesktopProfileRoute {
  if (explicit !== undefined) {
    return requireDesktopProfileRoute(explicit)
  }

  return source?.profile
    ? requireDesktopProfileRoute({
        connectionId: source.registryScoped ? source.connectionId : null,
        profile: source.profile
      })
    : { ...fallback }
}

/** A peer window's launch route plus whether the renderer asked for that
 * route explicitly ("Open profile in new window"). Only an explicit route is
 * the window's New-session default; an inherited one seeds boot alone. */
export interface DesktopWindowLaunch extends DesktopProfileRoute {
  profileWindow: boolean
}

export function resolveDesktopWindowLaunch(
  explicit: unknown,
  source: WindowConnectionRoute | null,
  fallback: DesktopProfileRoute
): DesktopWindowLaunch {
  return { ...resolveDesktopWindowRoute(explicit, source, fallback), profileWindow: explicit !== undefined }
}

// A profile-less boot/reconnect belongs to its sender. An explicit profile
// keeps the legacy route, even if the sender serves the same name remotely.
export function resolveDesktopConnectionRequest(
  profile: unknown,
  source: WindowConnectionRoute | null,
  primaryProfile: string
): DesktopProfileRoute {
  // Reject rather than pass through: this is the renderer's direct route into
  // ensureBackend(), and the persisted setter's validation must not be the
  // only place an invalid name is stopped.
  const requested = requireDesktopProfileName(profile) ?? ''

  if (source?.profile && !requested) {
    return resolveDesktopWindowRoute(undefined, source, { connectionId: null, profile: primaryProfile })
  }

  return { connectionId: null, profile: requested || primaryProfile }
}

// The legacy profile field records last use. defaultRoute is an explicit,
// app-wide preference, never changed by switching workspaces in any window.
export function createDesktopProfilePreferences(
  configPath: string,
  options: {
    onDefaultChanged?: (route: DesktopProfileRoute | null) => void
    validateRoute?: (route: DesktopProfileRoute) => void
  } = {}
) {
  function read(strict = false): Record<string, unknown> {
    try {
      const value = JSON.parse(fs.readFileSync(configPath, 'utf8'))

      return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
    } catch (error) {
      if (strict && (error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw error
      }

      return {}
    }
  }

  function write(value: Record<string, unknown>) {
    fs.mkdirSync(path.dirname(configPath), { recursive: true })
    const temporaryPath = `${configPath}.tmp`
    fs.writeFileSync(temporaryPath, JSON.stringify(value, null, 2), 'utf8')
    fs.renameSync(temporaryPath, configPath)
  }

  function getDefault(): DesktopProfileRoute | null {
    try {
      return requireDesktopProfileRoute(read().defaultRoute)
    } catch {
      return null
    }
  }

  function readActive(): null | string {
    const value = read().profile
    const profile = typeof value === 'string' ? value.trim() : ''

    return DESKTOP_PROFILE_NAME_RE.test(profile) ? profile : null
  }

  function remember(name: unknown): null | string {
    const profile = typeof name === 'string' ? name.trim() : ''

    if (profile && !DESKTOP_PROFILE_NAME_RE.test(profile)) {
      throw new Error(`Invalid profile name: ${profile}`)
    }

    write({ ...read(true), profile: profile || null })

    return profile || null
  }

  function setDefault(value: unknown): DesktopProfileRoute {
    const route = requireDesktopProfileRoute(value)
    options.validateRoute?.(route)
    write({ ...read(true), defaultRoute: route })
    options.onDefaultChanged?.(route)

    return route
  }

  function clearDefault() {
    write({ ...read(true), defaultRoute: null })
    options.onDefaultChanged?.(null)
  }

  function profileChanged(connectionId: null | string, oldName: string, newName: null | string, backendMode: string) {
    if (backendMode === 'local' && readActive() === oldName) {
      remember(newName || 'default')
    }

    const route = getDefault()

    if (route?.connectionId !== connectionId || route?.profile !== oldName) {
      return
    }

    if (newName) {
      setDefault({ ...route, profile: newName })
    } else {
      clearDefault()
    }
  }

  function connectionRemoved(connectionId: string) {
    if (getDefault()?.connectionId === connectionId) {
      clearDefault()
    }
  }

  function afterProfileRequest(
    connectionId: null | string,
    request: ProfileRenameRequest,
    response: unknown,
    backendMode: string
  ) {
    if (response && typeof response === 'object') {
      const result = response as Record<string, unknown>

      if (result.ok === false || result.success === false || result.error) {
        return
      }
    }

    const renamed = profileRenameFromRequest(request)
    const deleted = profileNameFromDeleteRequest(request)

    if (renamed) {
      profileChanged(connectionId, renamed.oldName, renamed.newName, backendMode)
    } else if (deleted) {
      profileChanged(connectionId, deleted, null, backendMode)
    }
  }

  return { afterProfileRequest, connectionRemoved, getDefault, profileChanged, readActive, remember, setDefault }
}
