// Secondary "session windows" — one extra OS window per chat so a user can
// work with multiple chats side by side. The pure, Electron-free pieces live
// here so they can be unit-tested with node --test (mirroring how the rest of
// electron/*.ts splits testable logic out of the main.ts monolith).

import { pathToFileURL } from 'node:url'

// Secondary windows open at the minimum usable size — a compact side panel for
// subagent watch / cmd-click session pop-out, not a second full desktop.
const SESSION_WINDOW_MIN_WIDTH = 420
const SESSION_WINDOW_MIN_HEIGHT = 620

// Shared webPreferences for every window that renders the chat transcript — the
// primary window AND the secondary session windows. Keeping it in one place is
// the whole point: the two BrowserWindow definitions in main.ts used to be
// hand-copied, and the secondary windows silently drifted apart (a streamed
// answer stalled until the window regained focus because one of them lost the
// throttling opt-out).
//
// Background throttling is deliberately NOT set here. It is managed at runtime
// by main.ts (`setBackgroundThrottling` driven by the merged `hermes:active-work`
// reports): while any turn is in flight every chat window is unthrottled so the
// transcript's bounded timer flush keeps painting while blurred, occluded, or
// minimized — and once all turns finish, Chromium's default throttling returns
// so an idle hidden window costs ~nothing. A static `backgroundThrottling:
// false` here would pin `document.visibilityState` to 'visible' forever,
// turning every visibility-gated poll in the renderer into an always-on timer
// (the "Hermes idles at 20% CPU while minimized" bug). The preload path is
// injected because it depends on the Electron entry's __dirname.
//
// `autoplayPolicy: 'no-user-gesture-required'` is load-bearing for voice:
// Chromium's default autoplay policy suspends audio (HTMLAudioElement.play()
// and AudioContext) until the user has interacted with the frame. A voice
// conversation started by the "Hey Hermes" wake word has NO preceding click,
// so the FIRST reply's audio playback was rejected (NotAllowedError, silently
// swallowed) and only turn 2+ spoke — the very "first message in a new voice
// session is silent" bug. Manual voice-start worked only because the button
// click counted as the gesture. This is a native app the user deliberately
// launched; there is no drive-by-autoplay concern to protect against.
//
// `focusOnNavigation: false` keeps renderer-driven work passive. Electron's
// default is true, so an in-page/SPA navigation can activate a blurred chat
// window while its transcript is streaming. Explicit user actions still call
// the main-process window focus paths (session re-open, notification/deep-link,
// app activation), preserving intentional raises without background focus theft.
function chatWindowWebPreferences(preloadPath: string) {
  return {
    preload: preloadPath,
    contextIsolation: true,
    webviewTag: true,
    sandbox: true,
    nodeIntegration: false,
    devTools: true,
    autoplayPolicy: 'no-user-gesture-required' as const,
    focusOnNavigation: false
  }
}

// Build the renderer URL for a secondary window. The renderer uses a
// HashRouter, so the session route lives after the '#'. The `?win=secondary`
// flag MUST sit in the query string BEFORE the '#': anything after the '#' is
// treated as the route by HashRouter and would break routeSessionId(). The
// renderer reads the flag from window.location.search to suppress the install /
// onboarding overlays and the global session sidebar. `watch=1` marks a
// spectator window (e.g. a running subagent's session): the renderer resumes it
// lazily so the gateway never builds an agent just to stream into it.
// `profile` names the backend the window must boot against (same carry as the
// HUD's buildHudWindowUrl): without it a pop-out/watch window adopts the
// PRIMARY profile and resolves the session id against the wrong backend
// (#82768, #61286). Absent → unchanged primary adoption.
function buildSessionWindowUrl(sessionId: string, { devServer, profile, rendererIndexPath, watch }: any = {}) {
  const profileKey = typeof profile === 'string' ? profile.trim() : ''
  const query = `?win=secondary${watch ? '&watch=1' : ''}${profileKey ? `&profile=${encodeURIComponent(profileKey)}` : ''}`
  const route = `#/${encodeURIComponent(sessionId)}`

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}${route}`
  }

  return `${pathToFileURL(rendererIndexPath).toString()}${query}${route}`
}

// Full peer windows render the ordinary app shell, so they deliberately do
// not use the `win` query parameter that selects a specialized renderer. The
// separate marker lets the renderer distinguish a peer from the one primary
// app window: app-launch source restoration belongs to the primary only, while
// a peer keeps the already-running backend it joined during boot.
function buildInstanceWindowUrl({ devServer, rendererIndexPath }: any = {}) {
  const query = '?peer=1'

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}`
  }

  return `${pathToFileURL(rendererIndexPath).toString()}${query}`
}

// Full "instance" windows (⌘⇧N / the "New Window" command) open a complete app
// peer, not a compact chat. Cascade each one off its source window's bounds so a
// new window doesn't land exactly on top of the one it was spawned from. Pure so
// it's unit-testable; the Electron glue (reading the focused window's bounds,
// constructing the BrowserWindow) stays in main.ts. `base` is the source
// window's current bounds, or null when there's no live source window — then the
// persisted primary geometry (`fallback`) is used as-is.
const INSTANCE_CASCADE_OFFSET = 32

function instanceWindowBounds(base: { x: number; y: number; width: number; height: number } | null, fallback: any) {
  if (!base) {
    return fallback
  }

  return {
    width: base.width,
    height: base.height,
    x: base.x + INSTANCE_CASCADE_OFFSET,
    y: base.y + INSTANCE_CASCADE_OFFSET
  }
}

// A small registry keyed by sessionId that guarantees one window per chat:
// opening a session that already has a live window focuses it instead of
// spawning a duplicate, and a window removes itself from the registry when it
// closes. The actual BrowserWindow construction is injected (the `factory`) so
// this module stays free of Electron and is unit-testable.
function createSessionWindowRegistry() {
  const windows = new Map()

  function openOrFocus(sessionId, factory) {
    const key = typeof sessionId === 'string' ? sessionId.trim() : ''

    if (!key) {
      return null
    }

    const existing = windows.get(key)

    if (existing && !existing.isDestroyed()) {
      // Focus-or-create: never duplicate a window for the same chat.
      if (typeof existing.isMinimized === 'function' && existing.isMinimized()) {
        existing.restore?.()
      }

      if (typeof existing.isVisible === 'function' && !existing.isVisible()) {
        existing.show?.()
      }

      existing.focus?.()

      return existing
    }

    const win = factory(key)

    if (!win) {
      return null
    }

    windows.set(key, win)

    // Self-cleanup on close so the registry never holds a destroyed window.
    win.on?.('closed', () => {
      if (windows.get(key) === win) {
        windows.delete(key)
      }
    })

    return win
  }

  return {
    openOrFocus,
    get: key => windows.get(key),
    has: key => windows.has(key),
    get size() {
      return windows.size
    }
  }
}

export {
  buildInstanceWindowUrl,
  buildSessionWindowUrl,
  chatWindowWebPreferences,
  createSessionWindowRegistry,
  instanceWindowBounds,
  SESSION_WINDOW_MIN_HEIGHT,
  SESSION_WINDOW_MIN_WIDTH
}
