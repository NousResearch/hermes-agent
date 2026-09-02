import { notifyError } from './notifications'

// Window flag set by the Electron main process when it opens a standalone
// session window (see electron/main.ts buildSessionWindowUrl). It rides in the
// query string BEFORE the HashRouter '#', so we read it from location.search,
// never from the router. A "secondary" window renders a single chat without the
// global session sidebar or the install / onboarding overlays.
const SECONDARY_WINDOW_FLAG = 'secondary'

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

let watchWindowCache: boolean | null = null

// A "hud" window is HUD mode: the chrome-free floating chat. Unlike the pet
// overlay / quick entry it is a FULL app renderer with its own gateway — the
// flag only tells the shell to render the slim floating layout (composer +
// scrollback) instead of the pane tree, so the composer it shows is the real
// one. Read from location.search for the same reason as the flag above.
let hudWindowCache: boolean | null = null

export function isHudWindow(): boolean {
  if (hudWindowCache !== null) {
    return hudWindowCache
  }

  let result = false

  try {
    result = new URLSearchParams(window.location.search).get('win') === 'hud'
  } catch {
    result = false
  }

  hudWindowCache = result

  return result
}

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

// A "browser" window is a popped-out in-app Browser: the same webview +
// address bar as the docked tab, in its own OS window. The tab id rides
// `?tab=` next to `win=browser` (before the hash, same contract as secondary).
let browserWindowCache: boolean | null = null

export function isBrowserWindow(): boolean {
  if (browserWindowCache !== null) {
    return browserWindowCache
  }

  let result = false

  try {
    result = new URLSearchParams(window.location.search).get('win') === 'browser'
  } catch {
    result = false
  }

  browserWindowCache = result

  return result
}

export function windowBrowserTabId(): null | string {
  try {
    return new URLSearchParams(window.location.search).get('tab')?.trim() || null
  } catch {
    return null
  }
}

// True for any window that is NOT the primary app instance — a secondary
// session window, the HUD, or a popped-out Browser. Single-claim channels
// (the quick-entry capture bridge, the pet overlay control bridge) and the
// install/onboarding overlays belong to the primary alone.
export const isAuxiliaryWindow = (): boolean => isSecondaryWindow() || isHudWindow() || isBrowserWindow()

// A full peer window renders the ordinary app shell against the backend that
// Electron already has running. It is not an auxiliary/specialized renderer,
// but it must not replay the primary window's app-launch source restoration
// after boot and silently re-home itself to another registered gateway.
export function isPeerInstanceWindow(search = typeof window === 'undefined' ? '' : window.location.search): boolean {
  try {
    return new URLSearchParams(search).get('peer') === '1'
  } catch {
    return false
  }
}

// The profile a helper window (the HUD) was asked to boot against, carried in
// the query string by the main process (see hudUrl). The HUD is a full app
// renderer that otherwise adopts the PRIMARY backend's profile — wrong the
// moment the conversation it was opened on belongs to another profile
// (#82285). Empty/absent means "no override": boot adopts the primary as
// before, so ordinary windows and single-profile users are untouched.
// Not cached: it is read a handful of times per boot and staying cache-free
// keeps it honest under test.
export function windowProfileOverride(): null | string {
  try {
    return new URLSearchParams(window.location.search).get('profile')?.trim() || null
  } catch {
    return null
  }
}

// True when running inside the Electron desktop shell (the preload bridge is
// present). The "open in new window" affordance is desktop-only.
export function canOpenSessionWindow(): boolean {
  return typeof window !== 'undefined' && typeof window.hermesDesktop?.openSessionWindow === 'function'
}

// True when the shell can open a full peer app window (⌘⇧N / "New Window").
export function canOpenNewWindow(): boolean {
  return typeof window !== 'undefined' && typeof window.hermesDesktop?.openWindow === 'function'
}

// True when the shell can pop the in-app Browser into its own OS window.
export function canOpenBrowserWindow(): boolean {
  return typeof window !== 'undefined' && typeof window.hermesDesktop?.openBrowserWindow === 'function'
}

// True when the shell can hand a session to the user's own terminal emulator.
// Desktop-only, and a REMOTE connection is excluded by the caller: the terminal
// we'd open is on this machine, but the session lives on the remote host.
export function canOpenSessionInTerminal(): boolean {
  return typeof window !== 'undefined' && typeof window.hermesDesktop?.openSessionInTerminal === 'function'
}

type WindowOpenResult = { ok: boolean; error?: string } | undefined

// Run a window-open bridge call, surfacing any failure as a toast. Shared by the
// session pop-out and the new-window opener.
async function runWindowOpen(call: () => Promise<WindowOpenResult>, failMessage: string): Promise<boolean> {
  try {
    const result = await call()

    if (!result?.ok) {
      notifyError(new Error(result?.error || 'unknown error'), failMessage)

      return false
    }

    return true
  } catch (err) {
    notifyError(err, failMessage)

    return false
  }
}

// Open (or focus) a standalone OS window for a single chat session. No-ops
// gracefully outside Electron so callers can wire it unconditionally.
// `watch: true` opens a spectator window (lazy resume, live-mirror stream).
// The window is a full renderer that adopts the PRIMARY profile unless told
// otherwise, so the owning profile rides along (same ladder as openHud,
// #82285): the session's stamped owner wins, and an unstamped/uncached id —
// a brand-new subagent child — inherits the profile the user is looking at
// (#82768, #61286).
export async function openSessionInNewWindow(sessionId: string, opts?: { watch?: boolean }): Promise<void> {
  if (!sessionId || !canOpenSessionWindow()) {
    return
  }

  // Lazy imports: `./profile` subscribes to the API client on load, so a
  // static import here would drag it into every page that opens windows.
  const [{ $activeGatewayProfile, normalizeProfileKey }, { $sessions, rememberedSessionProfile }] = await Promise.all([
    import('./profile'),
    import('./session')
  ])
  const profile = normalizeProfileKey(rememberedSessionProfile($sessions.get(), sessionId, $activeGatewayProfile.get()))

  await runWindowOpen(
    () => window.hermesDesktop.openSessionWindow(sessionId, { ...opts, profile }),
    'Could not open chat in a new window'
  )
}

// Open a new full-chrome app window — a peer instance of the primary that
// renders the complete app against the shared backend. No-ops outside Electron.
export async function openNewWindow(): Promise<void> {
  if (!canOpenNewWindow()) {
    return
  }

  await runWindowOpen(() => window.hermesDesktop.openWindow(), 'Could not open a new window')
}

/** Pop the in-app Browser into its own OS window. Returns whether the
 *  window opened so the caller can dock the tab again on failure. */
export async function openBrowserInNewWindow(tabId: string): Promise<boolean> {
  if (!tabId || !canOpenBrowserWindow()) {
    return false
  }

  return runWindowOpen(() => window.hermesDesktop.openBrowserWindow(tabId), 'Could not pop out browser')
}

// Resume a session in the user's own terminal emulator, running the TUI there.
// `cwd` starts the shell in the session's workspace; `profile` pins the runtime
// to the profile that owns the session. No-ops gracefully outside Electron.
export async function openSessionInTerminal(
  sessionId: string,
  opts?: { cwd?: string; profile?: string }
): Promise<void> {
  if (!sessionId || !canOpenSessionInTerminal()) {
    return
  }

  await runWindowOpen(
    () => window.hermesDesktop.openSessionInTerminal(sessionId, opts),
    'Could not open chat in a terminal'
  )
}
