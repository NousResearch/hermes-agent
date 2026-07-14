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

/**
 * One desired zoom shared by every chat window. A revision rejects an async
 * localStorage read when a newer in-memory choice already exists.
 */
export function createZoomCoordinator() {
  let desiredLevel
  let revision = 0

  return {
    beginRestore() {
      const restoreRevision = revision
      const canRestore = desiredLevel === undefined

      return level => {
        if (!canRestore || revision !== restoreRevision) {
          return undefined
        }

        desiredLevel = clampZoomLevel(level)

        return desiredLevel
      }
    },
    getDesired() {
      return desiredLevel
    },
    setDesired(level) {
      desiredLevel = clampZoomLevel(level)
      revision += 1

      return desiredLevel
    }
  }
}

// Chromium can drop webContents zoom when a BrowserWindow is minimized,
// restored, resized, or crosses onto a monitor with different display scaling.
// WSLg runs Electron as Linux, where the cross-platform `resize` event is
// available but the one-shot `resized` event is not.
export const ZOOM_REASSERT_WINDOW_EVENTS = ['show', 'restore', 'moved', 'resize']
export const ZOOM_RESIZE_REASSERT_DELAY_MS = 120

export function installZoomReassertOnWindowEvents(
  win,
  reassert,
  { clearTimer = clearTimeout, setTimer = setTimeout } = {}
) {
  if (!win?.on) {
    return
  }

  const run = () => {
    if (!win.isDestroyed?.()) {
      reassert()
    }
  }

  win.on('show', run)
  win.on('restore', run)
  win.on('moved', run)

  let resizeTimer
  win.on('resize', () => {
    if (resizeTimer !== undefined) {
      clearTimer(resizeTimer)
    }

    resizeTimer = setTimer(() => {
      resizeTimer = undefined
      run()
    }, ZOOM_RESIZE_REASSERT_DELAY_MS)
  })

  win.once?.('closed', () => {
    if (resizeTimer !== undefined) {
      clearTimer(resizeTimer)
    }
  })
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
