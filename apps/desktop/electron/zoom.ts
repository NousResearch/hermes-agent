/**
 * Pure helpers for window zoom. The main process owns webContents.setZoomLevel,
 * so the menu items, the Ctrl/Cmd shortcuts, and the settings UI all funnel
 * through this one clamped scale. Percent is the user-facing unit (100 = the
 * Chromium actual-size baseline); Chromium's internal unit is the zoom level,
 * where factor = 1.2 ^ level.
 *
 * Our shipped default is the Appearance 90% preset — tight enough to feel
 * denser than Chromium 100%, and selected in the UI Scale control on first run.
 */

export const ZOOM_STORAGE_KEY = 'hermes:desktop:zoomLevel'

const ZOOM_FACTOR_BASE = 1.2
const MIN_ZOOM_LEVEL = -9
const MAX_ZOOM_LEVEL = 9

/** Half Chromium's default step; matching the shortcuts and View menu. */
export const ZOOM_STEP = 0.1

/** Appearance 90% preset. Fresh installs + Actual Size / Ctrl+0. */
export const DEFAULT_ZOOM_LEVEL = Math.log(0.9) / Math.log(ZOOM_FACTOR_BASE)

export function clampZoomLevel(value) {
  if (!Number.isFinite(value)) {
    return DEFAULT_ZOOM_LEVEL
  }

  return Math.min(Math.max(value, MIN_ZOOM_LEVEL), MAX_ZOOM_LEVEL)
}

export function zoomLevelToPercent(level) {
  return Math.round(Math.pow(ZOOM_FACTOR_BASE, clampZoomLevel(level)) * 100)
}

export function percentToZoomLevel(percent) {
  if (!Number.isFinite(percent) || percent <= 0) {
    return DEFAULT_ZOOM_LEVEL
  }

  return clampZoomLevel(Math.log(percent / 100) / Math.log(ZOOM_FACTOR_BASE))
}

/**
 * Apply a clamped zoom level to a webContents AND notify the renderer, in that
 * order. Every path that changes zoom (user action, restore-on-load, lifecycle
 * re-assert) funnels through here so the settings UI Scale control can never
 * drift from the actually-applied level — the bug where restore set the level
 * but forgot to emit 'hermes:zoom:changed', leaving the control stuck at 100%.
 * Returns the clamped level so callers can persist it.
 */
export function applyZoomLevel(webContents, level) {
  const clamped = clampZoomLevel(level)
  webContents.setZoomLevel(clamped)
  webContents.send('hermes:zoom:changed', { level: clamped, percent: zoomLevelToPercent(clamped) })

  return clamped
}

// Chromium can drop webContents zoom when a BrowserWindow is resized, minimized
// and restored, crosses onto a monitor with different display scaling, or loses
// and regains focus (alt-tab on Windows high-DPI displays triggers a DPI
// re-evaluation). macOS and Windows provide trailing `resized`/`moved` events;
// Linux only provides the noisy `resize`/`move` pair, so debounce those
// fallbacks before re-applying the persisted level.
export const ZOOM_RESIZE_REASSERT_DELAY_MS = 100

// Linux settle-verify: a re-assert can land while the compositor is still
// reconfiguring the window's surface (Cosmic tiled mode fires a resize storm
// the moment a new session window opens, and the final event can arrive before
// the new scale is committed), in which case Chromium drops the just-applied
// zoom and the window stays stuck at the wrong scale until the next
// transition. After the debounced re-assert we therefore re-check a few times
// at a settle delay; each re-assert is drift-guarded (see
// restorePersistedZoomLevel), so a window that already matches the persisted
// level costs nothing and the chain is bounded.
export const ZOOM_REASSERT_SETTLE_DELAY_MS = 300
export const ZOOM_REASSERT_MAX_SETTLE_CHECKS = 3

export function zoomReassertWindowEvents(platform = process.platform) {
  return platform === 'linux'
    ? ['show', 'restore', 'focus', 'resize', 'move']
    : ['show', 'restore', 'focus', 'resized', 'moved']
}

// Linux/Wayland fires `focus` on intra-app focus shifts (sidebar clicks,
// Ctrl+Tab session switching, tile activation) — not just the cross-app
// alt-tab the Windows high-DPI immediate-reassert guard (#50837) targets.
// An undebounced reassert on every such focus event re-applies the persisted
// zoom level mid-interaction, producing a visible zoom/DPI jump. Debounce
// `focus` alongside `resize`/`move` on Linux; Windows alt-tab keeps its
// immediate path because `platform` is `win32` there.
const DEBOUNCED_REASSERT_EVENTS = new Set(['resize', 'move'])

export function isDebouncedReassertEvent(event, platform = process.platform) {
  return DEBOUNCED_REASSERT_EVENTS.has(event) || (platform === 'linux' && event === 'focus')
}

export function installZoomReassertOnWindowEvents(win, reassert, platform = process.platform) {
  if (!win?.on) {
    return
  }

  let resizeTimer
  let settleTimer
  let settleChecks = 0

  const scheduleSettleCheck = () => {
    if (platform !== 'linux') {
      return
    }

    settleChecks += 1

    if (settleChecks > ZOOM_REASSERT_MAX_SETTLE_CHECKS) {
      return
    }

    settleTimer = setTimeout(() => {
      if (!win.isDestroyed?.()) {
        reassert()
        scheduleSettleCheck()
      }
    }, ZOOM_REASSERT_SETTLE_DELAY_MS)
  }

  const reassertWithSettleCheck = () => {
    settleChecks = 0
    reassert()
    scheduleSettleCheck()
  }

  for (const event of zoomReassertWindowEvents(platform)) {
    win.on(event, () => {
      if (win.isDestroyed?.()) {
        return
      }

      if (!isDebouncedReassertEvent(event, platform)) {
        clearTimeout(settleTimer)
        reassertWithSettleCheck()

        return
      }

      clearTimeout(resizeTimer)
      resizeTimer = setTimeout(() => {
        if (!win.isDestroyed?.()) {
          clearTimeout(settleTimer)
          reassertWithSettleCheck()
        }
      }, ZOOM_RESIZE_REASSERT_DELAY_MS)
    })
  }
}

/**
 * Zoom-wiring decision per window kind. Chat windows (main + session + the HUD)
 * keep global UI zoom; the pet overlay and the Quick Entry composer opt out
 * because they size their own OS window and inheriting zoom would crop or
 * overflow them.
 *
 * Extracted so the "helper windows opt out, everything else opts in" contract is
 * unit-testable without booting a BrowserWindow or reading source.
 */
export const ZOOM_WINDOW_CONFIG = {
  chat: { zoom: true },
  petOverlay: { zoom: false },
  quickEntry: { zoom: false },
  wakeIndicator: { zoom: false }
} as const

export function zoomWiringForWindowKind(kind) {
  return ZOOM_WINDOW_CONFIG[kind] ?? ZOOM_WINDOW_CONFIG.chat
}
