import { computeWindowOptions, sanitizeWindowState } from './window-state'

// Secondary session-window ("pop-out chat") restore. Main-window tabs/active
// route are already covered by the renderer's own sessionTiles.v2 +
// layoutTree.v2 persistence and display.resume_last_session — this file
// extends neither; it only remembers which EXTRA windows existed so they can
// be reopened through the same spawnSecondaryWindow/createSessionWindow path.
// Session ids, profile names and registry connection ids only — never a
// backend URL, token, or title.
export interface SavedSessionWindow {
  profile: string | null
  connectionId: string | null
  sessionId: string
  watch: boolean
  geometry: NonNullable<ReturnType<typeof sanitizeWindowState>>
}

// "Instance" windows (File > New Window / hermes:window:openInstance) are a
// different kind of secondary window from the session pop-outs above: a full
// peer app window routed to a profile/connectionId, not a single chat. They
// have no sessionId and nothing to existence-probe — a profile route is
// always safe to reopen — so they get their own minimal saved shape and
// their own array rather than overloading SavedSessionWindow.
// Stable per-window identity, minted once by createInstanceWindow
// (crypto.randomUUID()) and carried through buildInstanceWindowUrl's
// `windowId` query param. The renderer's storage choke point
// (src/store/windows.ts windowScopedKey) namespaces its own tab/layout
// persistence keys by this id — reusing the SAME id across a quit/relaunch
// is what lets a reopened peer recover its own scoped sessionTiles.v2 /
// layoutTree.v2 instead of booting into empty storage (BUILD-STABLE-WINDOW-ID).
// `null` covers two legitimate cases: a pre-fix saved record that predates
// this field, and a value that failed WINDOW_ID_PATTERN — both degrade to a
// freshly minted id at reopen, same as createInstanceWindow's own default.
export interface SavedInstanceWindow {
  profile: string | null
  connectionId: string | null
  geometry: NonNullable<ReturnType<typeof sanitizeWindowState>>
  windowId: string | null
  // Missing legacy metadata cannot distinguish inherited from explicit routes.
  // Default to unpinned; retain the boot route without inventing a session default.
  profileWindow: boolean
}

export interface WindowSessionState {
  version: 1
  windows: SavedSessionWindow[]
  instanceWindows: SavedInstanceWindow[]
}

// Session/profile/connection ids are opaque identifiers Hermes itself mints
// (uuids, profile slugs); loose but bounded so a corrupt file can't inject
// something pathological into a later file path or URL.
const identifier = (value: unknown): value is string =>
  typeof value === 'string' && value.length > 0 && value.length <= 256

// Mirrors WINDOW_INSTANCE_ID_PATTERN in apps/desktop/src/store/windows.ts —
// the renderer only recognizes a windowId query value that matches this
// exact shape, so a saved id that WOULD be rejected there must also be
// rejected here (degrade to null) rather than round-tripping into the next
// reopen's URL unexamined.
export const WINDOW_INSTANCE_ID_PATTERN = /^[\w-]{1,64}$/

const sanitizedWindowId = (value: unknown): string | null =>
  typeof value === 'string' && WINDOW_INSTANCE_ID_PATTERN.test(value) ? value : null

export function sanitizeWindowSessionState(raw: unknown): WindowSessionState | null {
  if (!raw || typeof raw !== 'object' || (raw as any).version !== 1 || !Array.isArray((raw as any).windows)) {
    return null
  }

  const windows: SavedSessionWindow[] = []
  const seen = new Set<string>()

  for (const row of (raw as any).windows as any[]) {
    if (!row || !identifier(row.sessionId)) {
      continue
    }

    const profile = row.profile === null || identifier(row.profile) ? (row.profile ?? null) : undefined
    const connectionId = row.connectionId === null || identifier(row.connectionId) ? (row.connectionId ?? null) : undefined

    if (profile === undefined || connectionId === undefined) {
      continue
    }

    const geometry = sanitizeWindowState(row.geometry)
    const key = JSON.stringify([connectionId, profile, row.sessionId])

    if (!geometry || seen.has(key)) {
      continue
    }

    seen.add(key)
    windows.push({ profile, connectionId, sessionId: row.sessionId, watch: row.watch === true, geometry })
  }

  const instanceWindows: SavedInstanceWindow[] = []
  const seenInstanceIds = new Set<string>()
  const rawInstances = Array.isArray((raw as any).instanceWindows) ? ((raw as any).instanceWindows as any[]) : []

  for (const row of rawInstances) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const profile = row.profile === null || identifier(row.profile) ? (row.profile ?? null) : undefined
    const connectionId = row.connectionId === null || identifier(row.connectionId) ? (row.connectionId ?? null) : undefined

    if (profile === undefined || connectionId === undefined) {
      continue
    }

    const geometry = sanitizeWindowState(row.geometry)

    if (!geometry) {
      continue
    }

    // Dedup key is the windowId ALONE, never (connectionId, profile): a user
    // can legitimately have several peer windows open on the exact same
    // profile/connection route (multiple "File > New Window" instances),
    // and collapsing them by route was the t_7690aaea regression this file
    // exists to fix. A legacy row (no/invalid windowId) carries no identity
    // to collide on and is never deduped against another row — only an
    // EXACT windowId match (a corrupt file listing the same id twice) is
    // rejected, since two live windows can never share one storage scope.
    const windowId = sanitizedWindowId(row.windowId)

    if (windowId !== null) {
      if (seenInstanceIds.has(windowId)) {
        continue
      }

      seenInstanceIds.add(windowId)
    }

    instanceWindows.push({ profile, connectionId, geometry, windowId, profileWindow: row.profileWindow === true })
  }

  return { version: 1, windows, instanceWindows }
}

export function parseWindowSessionState(text: string): WindowSessionState | null {
  try {
    return sanitizeWindowSessionState(JSON.parse(text))
  } catch {
    return null
  }
}

export function serializeWindowSessionState(windows: SavedSessionWindow[], instanceWindows: SavedInstanceWindow[] = []): string {
  return JSON.stringify({ version: 1, windows, instanceWindows }, null, 2)
}

/**
 * Reopen every saved window via the injected `open` callback (the real
 * spawnSecondaryWindow/createSessionWindow path — never a parallel window
 * constructor). A session deleted since the last quit is skipped and logged,
 * never thrown: one missing session must not block the rest of startup.
 * `exists` is a caller-provided probe (a scoped backend GET) so this module
 * stays Electron/network-free and unit-testable.
 */
export async function restoreSessionWindows(
  state: WindowSessionState | null,
  deps: {
    exists: (window: SavedSessionWindow) => Promise<boolean>
    open: (window: SavedSessionWindow) => void
    quitting: () => boolean
    log: (message: string) => void
  }
): Promise<void> {
  for (const window of state?.windows ?? []) {
    if (deps.quitting()) {
      return
    }

    let found = false

    try {
      found = await deps.exists(window)
    } catch (error: any) {
      deps.log(`[window-sessions] restore probe failed, skipping one window: ${error?.message || error}`)

      continue
    }

    if (!found) {
      deps.log('[window-sessions] skipped a saved window whose session no longer exists')

      continue
    }

    if (!deps.quitting()) {
      deps.open(window)
    }
  }
}

export function restoredSessionWindowOptions(
  window: SavedSessionWindow,
  displays: unknown[]
): ReturnType<typeof computeWindowOptions> {
  return computeWindowOptions(window.geometry, displays)
}

/**
 * Reopen every saved instance window via the injected `open` callback (the
 * real createInstanceWindow path). Unlike session windows, an instance
 * window has nothing to existence-probe — a profile/connectionId route is
 * always safe to reopen — so this is just a straight loop, still aborting if
 * a quit arrives mid-restore.
 */
export function restoreInstanceWindows(
  state: WindowSessionState | null,
  deps: {
    open: (window: SavedInstanceWindow) => void
    quitting: () => boolean
  }
): void {
  for (const window of state?.instanceWindows ?? []) {
    if (deps.quitting()) {
      return
    }

    deps.open(window)
  }
}

export function restoredInstanceWindowOptions(
  window: SavedInstanceWindow,
  displays: unknown[]
): ReturnType<typeof computeWindowOptions> {
  return computeWindowOptions(window.geometry, displays)
}
