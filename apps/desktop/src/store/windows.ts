import { notifyError } from './notifications'

// Window flag set by the Electron main process when it opens a standalone
// session window (see electron/main.ts buildSessionWindowUrl). It rides in the
// query string BEFORE the HashRouter '#', so we read it from location.search,
// never from the router. A "secondary" window renders a single chat without the
// global session sidebar or the install / onboarding overlays.
const SECONDARY_WINDOW_FLAG = 'secondary'
const NEW_SESSION_WINDOW_FLAG = '1'

let secondaryWindowCache: boolean | null = null

export function isSecondaryWindow(): boolean {
  if (secondaryWindowCache !== null) {
    return secondaryWindowCache
  }

  let result = false

  try {
    result = new URLSearchParams(window.location.search).get('win') === SECONDARY_WINDOW_FLAG
  } catch {
    result = false
  }

  secondaryWindowCache = result

  return result
}

let newSessionWindowCache: boolean | null = null

export function isNewSessionWindow(): boolean {
  if (newSessionWindowCache !== null) {
    return newSessionWindowCache
  }

  let result = false

  try {
    result = new URLSearchParams(window.location.search).get('new') === NEW_SESSION_WINDOW_FLAG
  } catch {
    result = false
  }

  newSessionWindowCache = result

  return result
}

let watchWindowCache: boolean | null = null

// A "watch" window spectates a session that is being driven elsewhere (a
// running subagent). It resumes lazily — the gateway registers history + a
// transport for the live mirror without building an agent, so opening it is
// cheap even while the backend is busy running the delegation.
export function isWatchWindow(): boolean {
  if (watchWindowCache !== null) {
    return watchWindowCache
  }

  let result = false

  try {
    result = new URLSearchParams(window.location.search).get('watch') === '1'
  } catch {
    result = false
  }

  watchWindowCache = result

  return result
}

let secondaryWindowProfileCache: string | null | undefined

export function secondaryWindowProfile(): string | null {
  if (secondaryWindowProfileCache !== undefined) {
    return secondaryWindowProfileCache
  }

  let result: string | null = null

  try {
    const profile = new URLSearchParams(window.location.search).get('profile')?.trim()

    result = profile || null
  } catch {
    result = null
  }

  secondaryWindowProfileCache = result

  return result
}

// True when running inside the Electron desktop shell (the preload bridge is
// present). The "open in new window" affordance is desktop-only.
export function canOpenSessionWindow(): boolean {
  return typeof window !== 'undefined' && typeof window.hermesDesktop?.openSessionWindow === 'function'
}

type WindowOpenResult = { ok: boolean; error?: string } | undefined

// Run a window-open bridge call, surfacing any failure as a toast. Shared by the
// session pop-out and the new-session pop-out.
async function openWindow(call: () => Promise<WindowOpenResult>, failMessage: string): Promise<void> {
  try {
    const result = await call()

    if (!result?.ok) {
      notifyError(new Error(result?.error || 'unknown error'), failMessage)
    }
  } catch (err) {
    notifyError(err, failMessage)
  }
}

// Open (or focus) a standalone OS window for a single chat session. No-ops
// gracefully outside Electron so callers can wire it unconditionally.
// `watch: true` opens a spectator window (lazy resume, live-mirror stream).
export async function openSessionInNewWindow(
  sessionId: string,
  opts?: { profile?: string | null; watch?: boolean }
): Promise<void> {
  if (!sessionId || !canOpenSessionWindow()) {
    return
  }

  await openWindow(() => window.hermesDesktop.openSessionWindow(sessionId, opts), 'Could not open chat in a new window')
}

// Open a fresh compact window on the new-session draft.
export async function openNewSessionInNewWindow(): Promise<void> {
  if (!canOpenSessionWindow() || typeof window.hermesDesktop.openNewSessionWindow !== 'function') {
    return
  }

  await openWindow(() => window.hermesDesktop.openNewSessionWindow(), 'Could not open new session window')
}
