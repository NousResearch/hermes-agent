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
// and restored, or crosses onto a monitor with different display scaling. macOS
// and Windows provide trailing `resized`/`moved` events; Linux only provides the
// noisy `resize`/`move` pair, so debounce those fallbacks before re-applying the
// persisted level.
export const ZOOM_RESIZE_REASSERT_DELAY_MS = 100

export function zoomReassertWindowEvents(platform = process.platform) {
  return platform === 'linux' ? ['show', 'restore', 'resize', 'move'] : ['show', 'restore', 'resized', 'moved']
}

export function installZoomReassertOnWindowEvents(win, reassert, platform = process.platform) {
  if (!win?.on) {
    return
  }

  let resizeTimer

  for (const event of zoomReassertWindowEvents(platform)) {
    win.on(event, () => {
      if (win.isDestroyed?.()) {
        return
      }

      if (event !== 'resize' && event !== 'move') {
        reassert()

        return
      }

      clearTimeout(resizeTimer)
      resizeTimer = setTimeout(() => {
        if (!win.isDestroyed?.()) {
          reassert()
        }
      }, ZOOM_RESIZE_REASSERT_DELAY_MS)
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
