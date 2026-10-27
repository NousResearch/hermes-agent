// hud-game-overlay.ts — is the HUD floating over a fullscreen app (a game)?
//
// Discord's in-game overlay behavior: while a fullscreen app owns the screen
// the HUD steps back to a low-opacity, glanceable state, and steps forward
// again when the user engages it or a reply lands. The DECISION lives here as
// pure functions over the same front-to-back enumeration `window-below.ts`
// uses; main polls while the HUD is open and pushes changes to the HUD
// renderer, which owns the visual treatment (see `data-hud-game` in
// styles.css).
//
// "Fullscreen app" means: the frontmost other-process window on the HUD's
// display covers that display edge-to-edge. Walking the z-order front-to-back
// mirrors how the screen actually reads — if something windowed sits on top of
// the game, the game is not what the HUD is floating over.
//
// A note on what this can and cannot float over, so nobody debugs the wrong
// layer: an always-on-top window covers BORDERLESS fullscreen (the default in
// most modern games) and macOS fullscreen Spaces. True exclusive-fullscreen
// bypasses the compositor entirely — nothing short of injecting into the
// game's render pipeline (what Discord's native overlay does) draws over it.
// Detection still works there; the HUD is simply behind until the user
// alt-tabs, which Windows answers by flipping the game to composited output.

import type { EnumeratedWindow } from './window-below'

export interface GameOverlayState {
  active: boolean
  /** The fullscreen app's name while active, '' otherwise — the renderer may
   *  surface it ("over Balatro") and the diff key needs it either way. */
  app: string
}

export const INACTIVE_GAME_OVERLAY: GameOverlayState = { active: false, app: '' }

interface Bounds {
  x: number
  y: number
  width: number
  height: number
}

/** Allowance for DPI rounding and the 1px oversize some engines use to dodge
 *  the OS's own "looks fullscreen" heuristics. */
const COVER_EPSILON_PX = 2

/**
 * Desktop-shell windows that legitimately report display-sized bounds and must
 * never read as a game: the Windows desktop (Progman/WorkerW both belong to
 * Explorer), the macOS Dock/desktop layers. Matched on the OWNER name — titles
 * are localized, unavailable without permissions on macOS, and empty for most
 * of these anyway.
 */
const SHELL_APPS = [
  /^windows explorer$/i,
  /^explorer(\.exe)?$/i,
  /^program manager$/i,
  /^finder$/i,
  /^dock$/i,
  /^window ?server$/i,
  /^windowmanager$/i,
  /^gnome-shell$/i,
  /^plasmashell$/i
]

export const isShellWindow = (app: string): boolean => SHELL_APPS.some(pattern => pattern.test(app.trim()))

/** Whether `bounds` covers `display` edge-to-edge (within the DPI epsilon).
 *  Work-area coverage is deliberately not enough: a maximized window stops at
 *  the taskbar/menu bar, a fullscreen one does not — that IS the distinction. */
export const coversDisplay = (bounds: Bounds, display: Bounds, epsilon: number = COVER_EPSILON_PX): boolean =>
  bounds.x <= display.x + epsilon &&
  bounds.y <= display.y + epsilon &&
  bounds.x + bounds.width >= display.x + display.width - epsilon &&
  bounds.y + bounds.height >= display.y + display.height - epsilon

const intersects = (a: Bounds, b: Bounds): boolean =>
  a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height

/**
 * The fullscreen app the HUD is floating over on `display`, or null.
 *
 * Front-to-back: skip every window of our own process (all Hermes windows
 * share main's pid) and the desktop shell's display-sized layers, then let the
 * FIRST window that intersects the display decide — covering it means a
 * fullscreen app, anything less means ordinary windows are on top and the
 * overlay treatment would just make the HUD illegible over a busy desktop.
 * Zero-area rows (minimized windows report those on some platforms) never
 * decide either way.
 */
export function detectFullscreenApp(
  windows: EnumeratedWindow[],
  selfPid: number,
  display: Bounds
): EnumeratedWindow | null {
  for (const win of windows) {
    if (win.pid === selfPid || isShellWindow(win.app)) {
      continue
    }

    if (win.bounds.width <= 0 || win.bounds.height <= 0 || !intersects(win.bounds, display)) {
      continue
    }

    return coversDisplay(win.bounds, display) ? win : null
  }

  return null
}

/**
 * The display-covering app anywhere in the stack, ignoring z-order.
 *
 * The STAY half of the hysteresis below. Front-to-back order answers "is a
 * game what I'm looking at" for ENTERING overlay mode, but it cannot answer
 * "am I still over the game" once the user clicks the HUD to type: the game
 * stops being foreground, and whatever else they had open (a terminal, an
 * editor, a browser) is suddenly above it and vetoes. The game did not go
 * anywhere, so neither should the treatment.
 */
export function findFullscreenAppAnywhere(
  windows: EnumeratedWindow[],
  selfPid: number,
  display: Bounds
): EnumeratedWindow | null {
  return (
    windows.find(
      win =>
        win.pid !== selfPid &&
        !isShellWindow(win.app) &&
        win.bounds.width > 0 &&
        win.bounds.height > 0 &&
        coversDisplay(win.bounds, display)
    ) ?? null
  )
}

export const gameOverlayStateFor = (
  windows: EnumeratedWindow[],
  selfPid: number,
  display: Bounds,
  wasActive = false
): GameOverlayState => {
  // Hysteresis. ENTERING needs the game to be what the user is actually
  // looking at (front-to-back, windowed apps on top veto it). STAYING only
  // needs the game to still be there: clicking the HUD to type pushes the game
  // out of foreground and floats every other open window above it, which would
  // otherwise drop the treatment at exactly the moment the user is reading it.
  const fullscreen = wasActive
    ? findFullscreenAppAnywhere(windows, selfPid, display)
    : detectFullscreenApp(windows, selfPid, display)

  return fullscreen ? { active: true, app: fullscreen.app } : INACTIVE_GAME_OVERLAY
}

export interface HudGameOverlayWatchDeps {
  /** Front-to-back window enumeration; null when the platform cannot answer
   *  (Wayland, missing native module). Same contract as window-below's. */
  enumerate: () => Promise<EnumeratedWindow[] | null>
  /** Bounds of the display the HUD currently sits on. */
  displayBounds: () => Bounds
  selfPid: number
  /** Push a CHANGED state to the HUD renderer. */
  send: (state: GameOverlayState) => void
  intervalMs?: number
  /** Injectable timers so tests never wait on a real clock. */
  setIntervalFn?: typeof setInterval
  clearIntervalFn?: typeof clearInterval
}

/** Consecutive failed enumerations before the watch concludes the platform
 *  cannot answer and stops burning a subprocess/native call per tick. Two, not
 *  one: a single null can be a transient failure mid-session. */
const FAILURES_BEFORE_GIVING_UP = 2

/**
 * Poll for fullscreen-app changes while the HUD is open. Returns the disposer;
 * idempotent, and the caller must invoke it when the HUD closes.
 *
 * Polling, not events: no OS surfaces a cross-process "a fullscreen app
 * appeared" signal to an unprivileged window, and every consumer of this class
 * of information (Discord, Steam, GeForce overlays) watches for it. The
 * interval is slow enough to be free next to the HUD's own cursor feed.
 */
export function startHudGameOverlayWatch({
  enumerate,
  displayBounds,
  selfPid,
  send,
  intervalMs = 1500,
  setIntervalFn = setInterval,
  clearIntervalFn = clearInterval
}: HudGameOverlayWatchDeps): () => void {
  let last: GameOverlayState | null = null
  let failures = 0
  let inFlight = false
  let disposed = false

  const publish = (state: GameOverlayState) => {
    if (last === null || last.active !== state.active || last.app !== state.app) {
      last = state
      send(state)
    }
  }

  const tick = async () => {
    // Enumeration is async and slower than the interval on a bad day (X11
    // shells out); overlapping ticks would answer out of order.
    if (inFlight || disposed) {
      return
    }

    inFlight = true

    try {
      const windows = await enumerate()

      if (disposed) {
        return
      }

      if (windows === null) {
        failures += 1

        // The platform cannot answer (and said so twice): settle on inactive
        // and stop asking.
        if (failures >= FAILURES_BEFORE_GIVING_UP) {
          publish(INACTIVE_GAME_OVERLAY)
          dispose()
        }

        return
      }

      failures = 0
      publish(gameOverlayStateFor(windows, selfPid, displayBounds(), last?.active ?? false))
    } catch {
      // A throwing enumerator counts the same as a null answer.
      failures += 1

      if (failures >= FAILURES_BEFORE_GIVING_UP && !disposed) {
        publish(INACTIVE_GAME_OVERLAY)
        dispose()
      }
    } finally {
      inFlight = false
    }
  }

  const timer = setIntervalFn(() => void tick(), intervalMs)

  function dispose() {
    if (!disposed) {
      disposed = true
      clearIntervalFn(timer)
    }
  }

  void tick()

  return dispose
}
