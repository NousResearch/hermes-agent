/**
 * Pure helpers for window zoom. The main process owns webContents.setZoomLevel,
 * so the menu items, the Ctrl/Cmd shortcuts, and the settings UI all funnel
 * through this one clamped scale. Percent is the user-facing unit (100 = the
 * default size); Chromium's internal unit is the zoom level, where
 * factor = 1.2 ^ level.
 */

export const ZOOM_STORAGE_KEY = 'hermes:desktop:zoomLevel'

const ZOOM_FACTOR_BASE = 1.2
const MIN_ZOOM_LEVEL = -9
const MAX_ZOOM_LEVEL = 9

export function clampZoomLevel(value) {
  if (!Number.isFinite(value)) {
    return 0
  }

  return Math.min(Math.max(value, MIN_ZOOM_LEVEL), MAX_ZOOM_LEVEL)
}

export function zoomLevelToPercent(level) {
  return Math.round(Math.pow(ZOOM_FACTOR_BASE, clampZoomLevel(level)) * 100)
}

export function percentToZoomLevel(percent) {
  if (!Number.isFinite(percent) || percent <= 0) {
    return 0
  }

  return clampZoomLevel(Math.log(percent / 100) / Math.log(ZOOM_FACTOR_BASE))
}

// Chromium on Windows can drop webContents zoom when a BrowserWindow is minimized
// and restored. Re-apply the persisted level on these lifecycle transitions.
export const ZOOM_REASSERT_WINDOW_EVENTS = ['show', 'restore']

export function installZoomReassertOnWindowEvents(win, reassert) {
  if (!win?.on) {
    return
  }

  for (const event of ZOOM_REASSERT_WINDOW_EVENTS) {
    win.on(event, () => {
      if (win.isDestroyed?.()) {
        return
      }

      reassert()
    })
  }
}

/**
 * Zoom-wiring decision per window kind. Chat windows (main + session) keep
 * global UI zoom; the pet overlay opts out because it sizes its own OS window
 * to the sprite and inheriting zoom would crop it.
 *
 * Extracted so the "pet opts out, everything else opts in" contract is
 * unit-testable without booting a BrowserWindow or reading source.
 */
export const ZOOM_WINDOW_CONFIG = {
  chat: { zoom: true },
  petOverlay: { zoom: false }
} as const

export function zoomWiringForWindowKind(kind) {
  return ZOOM_WINDOW_CONFIG[kind] ?? ZOOM_WINDOW_CONFIG.chat
}
