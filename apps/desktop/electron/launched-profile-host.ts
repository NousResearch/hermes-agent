import fs from 'node:fs'
import path from 'node:path'

import {
  connectionScopeKey,
  profileHasRemoteConnection,
  profileIsLaunched,
  type ProfileRouteOptions
} from './connection-config'
import { DEFAULT_FETCH_TIMEOUT_MS, resolveTimeoutMs } from './hardening'
import { assertProfileMayRunLocally, readRunElsewhereMarker } from './profile-runs-elsewhere'

// What this host still does for a profile a plugin runs elsewhere (a launched
// profile: its per-profile entry carries `launcher`, see connection-config).
// The launched backend serves only that profile's home, so the host answers
// the profile listing for it, and nothing on the host may start or serve a
// profile core marks as run elsewhere (profile-runs-elsewhere).

/**
 * The REST calls the host answers for a launched profile instead of its
 * backend: `GET /api/profiles` (the launched backend cannot list the host's
 * profiles) and `GET /api/profiles/active` (it is scoped to its own profile by
 * construction). Null for anything else.
 */
export function launchedProfileHostRequest(method: unknown, requestPath: unknown): null | 'active' | 'list' {
  if (String(method || 'GET').toUpperCase() !== 'GET') {
    return null
  }

  let pathname: string

  try {
    pathname = new URL(String(requestPath || ''), 'https://example.invalid').pathname
  } catch {
    return null
  }

  if (pathname === '/api/profiles') {
    return 'list'
  }

  return pathname === '/api/profiles/active' ? 'active' : null
}

export interface LaunchedProfileHostDeps<Backend> {
  /** The backend `resolveProfileBackendRoute` gives a local profile. */
  ensureProfileBackend: (profile: string) => Promise<Backend>
  fetchJson: (backend: Backend, path: string, opts: { method: 'GET'; timeoutMs: number }) => Promise<unknown>
  hermesHome: string
  isProfileName: (name: string) => boolean
  primaryProfile: () => string
  readConfig: () => Record<string, any>
  startPrimary: () => Promise<Backend>
}

/** The registered connection a request is pinned to: its kind, and the dial of that connection's own backend. */
export interface PinnedConnection<Backend> {
  dial: () => Promise<Backend>
  kind: () => unknown
}

export function createLaunchedProfileHost<Backend>(deps: LaunchedProfileHostDeps<Backend>) {
  const isLaunched = (profile: null | string | undefined) => profileIsLaunched(deps.readConfig(), profile)
  const isMarked = (profile: string) => Boolean(readRunElsewhereMarker(deps.hermesHome, profile))

  // Hermes's own backend on this computer serves the profile: no plugin runs
  // it (a launcher entry is a remote connection too), core does not mark it,
  // and the user has not connected it to a remote host, whose backend would
  // list that computer's profiles.
  const runsHere = (profile: string) => !isMarked(profile) && !profileHasRemoteConnection(deps.readConfig(), profile)

  // Local profile names (folders under <HERMES_HOME>/profiles), default first.
  const localProfileNames = (): string[] => {
    try {
      const names = fs
        .readdirSync(path.join(deps.hermesHome, 'profiles'), { withFileTypes: true })
        .filter(entry => entry.isDirectory() && entry.name !== 'default' && deps.isProfileName(entry.name))
        .map(entry => entry.name)

      return ['default', ...names.sort()]
    } catch {
      return ['default']
    }
  }

  // The backend that lists this host's profiles: the primary's, or when a
  // plugin runs the primary profile, the backend of the first local profile
  // that runs here (resolveProfileBackendRoute pools it on this host).
  const ensureHostBackend = async (): Promise<Backend> => {
    if (!isLaunched(deps.primaryProfile())) {
      return deps.startPrimary()
    }

    const hostProfile = localProfileNames().find(runsHere)

    if (!hostProfile) {
      throw new Error(
        'Every profile on this computer runs through a plugin or connects to a remote host, so none can list profiles.'
      )
    }

    return deps.ensureProfileBackend(hostProfile)
  }

  // The backend a roster source lists from: for this computer, the host's while a plugin runs the primary.
  const rosterBackend = (kind: unknown, dial: () => Promise<Backend>): Promise<Backend> =>
    kind === 'local' && isLaunched(deps.primaryProfile()) ? ensureHostBackend() : dial()

  // The sticky CLI profile (`hermes profile use`), as /api/profiles/active reports it.
  const stickyActiveProfile = (): string => {
    try {
      return fs.readFileSync(path.join(deps.hermesHome, 'active_profile'), 'utf8').trim() || 'default'
    } catch {
      return 'default'
    }
  }

  return {
    /**
     * The renderer's REST call when the host answers it for a launched profile; undefined otherwise.
     * `pinned`: the request names a registered connection. "This device" is this computer too, so the
     * host answers there as well, from that connection's own backend; a gateway on another computer
     * lists its own profiles.
     */
    async answer(
      request: { method?: unknown; path?: string; profile?: unknown; timeoutMs?: unknown },
      pinned?: PinnedConnection<Backend>
    ) {
      const kind = launchedProfileHostRequest(request?.method, request?.path)
      const profile = connectionScopeKey(request?.profile) || deps.primaryProfile()

      if (!kind || !isLaunched(profile) || (pinned && pinned.kind() !== 'local')) {
        return undefined
      }

      if (kind === 'active') {
        return { active: stickyActiveProfile(), current: profile }
      }

      const backend = pinned ? rosterBackend('local', pinned.dial) : ensureHostBackend()

      return deps.fetchJson(await backend, String(request.path), {
        method: 'GET',
        timeoutMs: resolveTimeoutMs(request?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)
      })
    },

    /** Refuse a local start of a profile core marks as run elsewhere. */
    assertMayRunLocally: (profile: null | string | undefined) => assertProfileMayRunLocally(deps.hermesHome, profile),

    /** The same refusal for a profile the local primary would serve by scope. */
    assertScopeMayRunLocally(profile: string, route: ProfileRouteOptions) {
      const servedElsewhere = route.profileRemoteOverride || route.globalRemote || route.primaryRemoteActive

      if (profile !== deps.primaryProfile() && !servedElsewhere) {
        assertProfileMayRunLocally(deps.hermesHome, profile)
      }
    },

    ensureHostBackend,

    rosterBackend,

    /** A home a sandbox can write: launched now, or marked by core whether or not this app reaches it. */
    homeIsSandboxed: (profile: string) => isLaunched(profile) || isMarked(profile),

    isLaunched
  }
}
