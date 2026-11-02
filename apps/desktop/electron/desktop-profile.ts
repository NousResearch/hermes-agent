import fs from 'node:fs'
import path from 'node:path'

import { LOCAL_CONNECTION_ID } from './connection-registry'
import { profileNameFromDeleteRequest } from './profile-delete-routing'
import { profileRenameFromRequest, type ProfileRenameRequest } from './profile-rename-routing'
import type { WindowConnectionRoute } from './window-connection-route'

export interface DesktopProfileRoute {
  connectionId: null | string
  profile: string
}

/** What decides whether a remembered route names a profile on this machine. */
export interface MissingLocalProfileCheck {
  /** App-wide remote mode: one remote gateway serves every legacy (null-connection) route. */
  globalRemote: boolean
  /** The profile has its own remote, SSH or cloud route. */
  hasRemoteOverride: (profile: string) => boolean
  /** The profile this launch named with --profile; like `hermes -p`, it still fails loudly. */
  launchProfile: null | string
  log: (line: string) => void
  /** The profile's directory exists under this machine's profiles root. */
  profileExists: (profile: string) => boolean
}

export const DESKTOP_PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

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
  const requested = typeof profile === 'string' ? profile.trim() : ''

  if (source?.profile && !requested) {
    return resolveDesktopWindowRoute(undefined, source, { connectionId: null, profile: primaryProfile })
  }

  return { connectionId: null, profile: requested || primaryProfile }
}

/** True when resolveDesktopConnectionRequest filled in the primary's profile:
 *  the caller named none and its window has no route of its own. */
export function desktopConnectionRequestInheritsPrimary(
  profile: unknown,
  source: WindowConnectionRoute | null
): boolean {
  return !(typeof profile === 'string' && profile.trim()) && !source?.profile
}

/** The preference IPC the renderer reaches. Main owns each connection's
 *  last-used profile: windows send one change at a time and import their
 *  pre-mirror record once, never their whole copy. */
export function registerDesktopProfilePreferenceIpc(
  ipc: { handle: (channel: string, listener: (event: unknown, ...args: any[]) => unknown) => void },
  preferences: ReturnType<typeof createDesktopProfilePreferences>,
  isKnownConnection: (connectionId: string) => boolean
) {
  ipc.handle('hermes:profile:default:get', async () => preferences.getDefault())
  ipc.handle('hermes:profile:default:set', async (_event, route) => preferences.setDefault(route))
  ipc.handle('hermes:profile:connections:remember', async (_event, connectionId, profile) =>
    preferences.rememberConnectionProfile(connectionId, profile, isKnownConnection)
  )
  ipc.handle('hermes:profile:connections:import', async (_event, record) =>
    preferences.importConnectionProfiles(record, isKnownConnection)
  )
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

  // Each connection's last-used profile, mirrored from the renderer so main can
  // boot a remote primary on it before any window exists. Machine-local
  // `profile` above is never a remote's memory.
  function readConnectionProfiles(): Record<string, string> {
    const value = read().connectionProfiles
    const entries = value && typeof value === 'object' && !Array.isArray(value) ? Object.entries(value) : []

    return Object.fromEntries(
      entries.filter(
        (entry): entry is [string, string] => typeof entry[1] === 'string' && DESKTOP_PROFILE_NAME_RE.test(entry[1])
      )
    )
  }

  function connectionProfile(connectionId: null | string | undefined): null | string {
    return connectionId ? (readConnectionProfiles()[connectionId] ?? null) : null
  }

  // Main owns this map. Each window reports one (connection, profile) change,
  // never its whole copy: windows do not see each other's changes, so a
  // wholesale push from an older window would erase newer picks or resurrect
  // an entry connectionRemoved just pruned.
  function rememberConnectionProfile(
    connectionId: unknown,
    profile: unknown,
    isKnownConnection: (connectionId: string) => boolean = () => true
  ) {
    if (typeof connectionId !== 'string' || !connectionId.trim() || typeof profile !== 'string') {
      return
    }

    const existing = readConnectionProfiles()

    if (!DESKTOP_PROFILE_NAME_RE.test(profile) || existing[connectionId] === profile || !isKnownConnection(connectionId)) {
      return
    }

    write({ ...read(true), connectionProfiles: { ...existing, [connectionId]: profile } })
  }

  // One-time import of the record windows kept before main owned the map:
  // the first import fills only connections main has no entry for, later ones
  // are ignored, so a stale window can never overwrite main's map.
  function importConnectionProfiles(value: unknown, isKnownConnection: (connectionId: string) => boolean = () => true) {
    const current = read(true)

    if (current.connectionProfilesImported === true) {
      return
    }

    const entries = value && typeof value === 'object' && !Array.isArray(value) ? Object.entries(value) : []
    const existing = readConnectionProfiles()

    const imported = Object.fromEntries(
      entries.filter(
        ([id, profile]) =>
          id.trim() &&
          !(id in existing) &&
          isKnownConnection(id) &&
          typeof profile === 'string' &&
          DESKTOP_PROFILE_NAME_RE.test(profile)
      )
    )

    write({ ...current, connectionProfiles: { ...existing, ...imported }, connectionProfilesImported: true })
  }

  function connectionRemoved(connectionId: string) {
    if (getDefault()?.connectionId === connectionId) {
      clearDefault()
    }

    const { [connectionId]: removed, ...rest } = readConnectionProfiles()

    if (removed) {
      write({ ...read(true), connectionProfiles: rest })
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

  // userData outlives HERMES_HOME: a reinstall that wiped the Hermes home, or a
  // `hermes profile delete` while the app was closed, never reaches
  // afterProfileRequest. A remembered local profile that is gone would launch
  // `--profile <gone>` on every boot, which the backend refuses, so apply the
  // in-app delete's outcome: last-used falls back to default and a matching
  // explicit default is cleared. Remote routes are never judged by the local
  // disk. Runs on the startup path, so a failed write is logged, not thrown.
  function forgetMissingLocalProfiles(check: MissingLocalProfileCheck): string[] {
    const isGone = ({ connectionId, profile }: DesktopProfileRoute) =>
      profile !== 'default' &&
      profile !== check.launchProfile &&
      (connectionId === null ? !check.globalRemote : connectionId === LOCAL_CONNECTION_ID) &&
      !check.hasRemoteOverride(profile) &&
      !check.profileExists(profile)

    const forgotten: string[] = []

    try {
      const active = readActive()

      if (active && isGone({ connectionId: null, profile: active })) {
        remember('default')
        forgotten.push(active)
        check.log(`[profile] last-used profile "${active}" no longer exists on this machine; starting as default`)
      }

      const route = getDefault()

      if (route && isGone(route)) {
        clearDefault()
        forgotten.push(route.profile)
        check.log(`[profile] cleared default profile "${route.profile}": it no longer exists on this machine`)
      }
    } catch (error) {
      check.log(`[profile] could not forget missing local profiles: ${(error as Error).message}`)
    }

    return forgotten
  }

  return {
    afterProfileRequest,
    connectionProfile,
    connectionRemoved,
    forgetMissingLocalProfiles,
    getDefault,
    profileChanged,
    readActive,
    importConnectionProfiles,
    remember,
    rememberConnectionProfile,
    setDefault
  }
}
