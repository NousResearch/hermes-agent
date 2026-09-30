import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

import { BrowserWindow, screen } from 'electron'

import { attachRendererConsoleCapture } from './renderer-log'
import {
  normalizeScanlineScope,
  normalizeScanlineState,
  SCANLINE_FADE_MS,
  SCANLINE_SCOPE_MAX_AGE_MS,
  scanlineBoundsForScope,
  type ScanlineScope,
  type ScanlineState,
  scanlineWindowBounds
} from './scanline'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'

interface ScanlineWindowOptions {
  devServer?: string
  loadWindowUrl: (window: BrowserWindow, url: string, label: string) => void
  log: (message: string) => void
  preloadPath: string
  rendererIndex: () => string
  wireWindow: (window: BrowserWindow) => void
  // Path to the scope file the capture helper writes (%LOCALAPPDATA%\hermes on
  // Windows). When present + fresh, it scopes the sweep to the screen being
  // captured. Omitted → the overlay always sweeps the whole desktop.
  scopePath?: string
  scopePollMs?: number
}

// A full-screen, transparent, click-through, always-on-top overlay that plays
// the "reading the display" scanline sweep while a screen capture is in
// flight. Two independent sources drive it:
//   1. Renderer IPC — a `computer_use` capture tool call (built-in, primary
//      display only). Sweeps the WHOLE desktop (scope `both`), the original
//      behaviour.
//   2. Scope file — the capture helper writes `{phase, scope, ts}` so the sweep
//      can scope to exactly the screen that was captured (primary / secondary /
//      both). The app polls it: `start` lights the sweep on that display,
//      `end` fades it out and resets to `both`.
// The window is visible while EITHER source is active; the bounds follow the
// helper scope when it's active, otherwise the full-desktop union.
export function createScanlineWindowController({
  devServer,
  loadWindowUrl,
  log,
  preloadPath,
  rendererIndex,
  wireWindow,
  scopePath,
  scopePollMs = 300
}: ScanlineWindowOptions) {
  let hideTimer: NodeJS.Timeout | null = null
  let window: BrowserWindow | null = null

  // Two trigger sources, reconciled into one visible state.
  let rendererActive = false
  let helperActive = false
  let scope: ScanlineScope = 'both'
  // Last scope file we acted on, so the poll doesn't re-fire a stale `start`.
  let lastScopeTs = 0

  const url = () => {
    if (devServer) {
      return `${devServer.endsWith('/') ? devServer.slice(0, -1) : devServer}/?win=scanline#/`
    }

    return `${pathToFileURL(rendererIndex()).toString()}?win=scanline#/`
  }

  const effectiveState = (): ScanlineState => (rendererActive || helperActive ? 'active' : 'hidden')

  // Bounds for the current trigger: scope to the helper's display when it's the
  // active source, otherwise span every display.
  const currentBounds = () =>
    helperActive
      ? scanlineBoundsForScope(screen.getAllDisplays(), scope, screen.getPrimaryDisplay().id)
      : scanlineWindowBounds(screen.getAllDisplays())

  const reposition = () => {
    if (!window || window.isDestroyed()) {
      return
    }

    window.setBounds(currentBounds())
  }

  const sendState = () => {
    if (!window || window.isDestroyed()) {
      return
    }

    window.webContents.send('hermes:scanline:state', effectiveState())
  }

  const spawn = () => {
    const requested = currentBounds()

    const next = new BrowserWindow({
      ...requested,
      alwaysOnTop: true,
      backgroundColor: '#00000000',
      focusable: false,
      frame: false,
      fullscreenable: false,
      hasShadow: false,
      maximizable: false,
      minimizable: false,
      movable: false,
      resizable: false,
      show: false,
      skipTaskbar: true,
      transparent: true,
      webPreferences: {
        backgroundThrottling: false,
        contextIsolation: true,
        devTools: true,
        nodeIntegration: false,
        preload: preloadPath,
        sandbox: true
      }
    })

    next.setAlwaysOnTop(true, 'screen-saver')
    // Click-through: the sweep is ambient and must never intercept input.
    // `forward: true` keeps Chromium hit-testing for the (unused) hover path.
    next.setIgnoreMouseEvents(true, { forward: true })

    wireWindow(next)

    // Log-only renderer lifecycle: the overlay is ambient; its loss belongs in
    // desktop.log, never resurrected (same policy as the wake cue).
    installWindowRendererLifecycle(next, { kind: 'scanline', callbacks: { log } })
    // Console errors go through the shared capture (renderer-log.ts owns
    // console-message; the lifecycle helper deliberately does not).
    attachRendererConsoleCapture(next, 'scanline', log)

    next.webContents.on('did-finish-load', () => {
      sendState()
    })
    next.once('ready-to-show', () => {
      if (!next.isDestroyed() && effectiveState() !== 'hidden') {
        // Re-assert the (possibly scoped) bounds at show time: Windows clamps a
        // freshly-created frameless window to the monitor's WORK AREA (it stops
        // at the taskbar) even when the constructor asked for the full display.
        // Re-applying setBounds on a now-visible window overrides that clamp so
        // the overlay truly covers the whole screen, taskbar included.
        next.setBounds(currentBounds())
        next.showInactive()
      }
    })
    next.on('closed', () => {
      if (window === next) {
        window = null
      }
    })

    loadWindowUrl(next, url(), 'Screen-analysis scanline')

    return next
  }

  // Reconcile the window with the desired visible state + bounds.
  const reconcile = () => {
    const state = effectiveState()

    if (hideTimer) {
      clearTimeout(hideTimer)
      hideTimer = null
    }

    if (state === 'hidden') {
      sendState()
      // Let the fade-out animation land before the window actually hides, so
      // the sweep dissolves rather than vanishing mid-scan.
      hideTimer = setTimeout(() => {
        hideTimer = null

        if (effectiveState() === 'hidden' && window && !window.isDestroyed()) {
          window.hide()
        }
      }, SCANLINE_FADE_MS)

      return
    }

    if (!window || window.isDestroyed()) {
      window = spawn()
    } else {
      reposition()
      sendState()
      window.showInactive()
    }
  }

  // Renderer IPC path: a `computer_use` capture tool call (built-in, primary).
  const setState = (value: unknown) => {
    rendererActive = normalizeScanlineState(value) === 'active'
    reconcile()
  }

  // Scope-file path: the capture helper bracketed an analysis of one screen.
  const applyScopeFile = (phase: string, scopeValue: unknown, ts: number) => {
    if (ts <= lastScopeTs) {
      return // already handled this (or an older) event
    }

    lastScopeTs = ts

    if (phase === 'start') {
      scope = normalizeScanlineScope(scopeValue)
      helperActive = true
    } else if (phase === 'end') {
      helperActive = false
      scope = 'both'
    } else {
      return
    }

    reconcile()
  }

  const pollScopeFile = () => {
    if (!scopePath) {
      return
    }

    let raw: string

    try {
      raw = readFileSync(scopePath, 'utf8')
    } catch {
      return // no file → nothing in flight from the helper
    }

    try {
      const parsed = JSON.parse(raw) as { phase?: unknown; scope?: unknown; ts?: unknown }
      const ts = typeof parsed.ts === 'number' ? parsed.ts : 0

      // A `start` older than the max age is a stale leftover (the helper or the
      // agent crashed before writing `end`) — drop it so the sweep can't stick.
      if (parsed.phase === 'start' && ts > 0 && Date.now() - ts > SCANLINE_SCOPE_MAX_AGE_MS) {
        if (helperActive) {
          helperActive = false
          scope = 'both'
          lastScopeTs = ts
          reconcile()
        }

        return
      }

      applyScopeFile(typeof parsed.phase === 'string' ? parsed.phase : '', parsed.scope, ts)
    } catch {
      // undecodable file — ignore
    }
  }

  let scopePoll: NodeJS.Timeout | null = null

  if (scopePath) {
    scopePoll = setInterval(pollScopeFile, scopePollMs)
    scopePoll.unref?.()
  }

  const close = () => {
    if (scopePoll) {
      clearInterval(scopePoll)
      scopePoll = null
    }

    if (hideTimer) {
      clearTimeout(hideTimer)
      hideTimer = null
    }

    if (window && !window.isDestroyed()) {
      window.close()
    }

    window = null
    rendererActive = false
    helperActive = false
    scope = 'both'
    lastScopeTs = 0
  }

  return {
    close,
    getState: () => effectiveState(),
    reposition,
    setState
  }
}
