/**
 * Window text size (zoom).
 *
 * The main process owns the zoom level and persists it (see electron/zoom.ts
 * for the scale). The renderer only mirrors the current percent for the
 * settings UI: preset clicks go to the main process over IPC, and every
 * change comes back through onChanged, including ones made with the
 * Ctrl/Cmd +/-/0 shortcuts or the View menu, so the UI never drifts.
 */

import { atom } from 'nanostores'

// Mirror DEFAULT_ZOOM_LEVEL (90%) so Appearance doesn't flash 100% before
// the main-process zoom.get() resolves. Keep in sync with electron/zoom.ts.
export const $zoomPercent = atom<number>(90)

export function setZoomPercent(percent: number): void {
  window.hermesDesktop?.zoom?.setPercent(percent)
}

let zoomInitialized = false

function initZoomStore() {
  if (zoomInitialized || typeof window === 'undefined' || !window.hermesDesktop?.zoom) {
    return
  }

  zoomInitialized = true
  void window.hermesDesktop.zoom.get().then(({ percent }) => $zoomPercent.set(percent))
  window.hermesDesktop.zoom.onChanged(({ percent }) => $zoomPercent.set(percent))
}

/**
 * Re-fetch the current zoom from the main process. The store is lazily
 * initialized (the module is only imported when Settings opens), so the
 * initial ``zoom.get()`` can race with ``restorePersistedZoomLevel`` —
 * ``webContents.getZoomLevel()`` may still read Chromium's baseline (0 →
 * 100%) before the persisted level is reasserted. Calling this from
 * ``AppearanceSettings`` on mount gives the reassert time to land and
 * corrects the displayed value.
 */
export function refreshZoomPercent(): void {
  if (typeof window !== 'undefined' && window.hermesDesktop?.zoom) {
    void window.hermesDesktop.zoom.get().then(({ percent }) => $zoomPercent.set(percent))
  }
}

initZoomStore()
