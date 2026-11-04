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

export interface WindowSessionState {
  version: 1
  windows: SavedSessionWindow[]
}

// Session/profile/connection ids are opaque identifiers Hermes itself mints
// (uuids, profile slugs); loose but bounded so a corrupt file can't inject
// something pathological into a later file path or URL.
const identifier = (value: unknown): value is string =>
  typeof value === 'string' && value.length > 0 && value.length <= 256

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

  return { version: 1, windows }
}

export function parseWindowSessionState(text: string): WindowSessionState | null {
  try {
    return sanitizeWindowSessionState(JSON.parse(text))
  } catch {
    return null
  }
}

export function serializeWindowSessionState(windows: SavedSessionWindow[]): string {
  return JSON.stringify({ version: 1, windows }, null, 2)
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
