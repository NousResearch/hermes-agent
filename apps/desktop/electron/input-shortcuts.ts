/**
 * Shared keyboard shortcuts for Close Tab, Reload, and Zoom that run in the
 * main process before Chromium or the renderer see the keystroke.
 *
 * Extracted from main.ts so the `before-input-event` handlers can be
 * unit-tested without a live BrowserWindow.
 */

import type { BrowserWindow, Input } from 'electron'

const IS_MAC = process.platform === 'darwin'

/**
 * Map to remember the instant each BrowserWindow received focus.
 * Used by the focus-grace guard to ignore accelerators for a short window
 * after a focus transfer, so keyup/auto-repeat from another app (e.g. Ctrl+W
 * pressed in Chromium, that window dies, Windows activates Hermes while the
 * keys are still down) can't close a tab or reload a page.
 */
const focusGraceUntil = new Map<number, number>()

const FOCUS_GRACE_MS = 200

export function installInputShortcuts(
  window: BrowserWindow,
  isMac: boolean = IS_MAC,
  opts?: {
    sendClosePreviewRequested?: () => void
    sendPreviewNavCommand?: (command: 'reload') => void
    setAndPersistZoomLevel?: (window: BrowserWindow, level: number) => void
    getDefaultZoomLevel?: () => number
    getZoomStep?: () => number
  }
) {
  const { webContents } = window

  if (!webContents || webContents.isDestroyed()) {
    return () => {}
  }

  const windowId = window.id

  const focusHandler = () => {
    focusGraceUntil.set(windowId, Date.now() + FOCUS_GRACE_MS)
  }

  window.on('focus', focusHandler)

  const inputHandler = (event: Electron.Event, input: Input) => {
    if (input.type !== 'keyDown' || input.isAutoRepeat) {
      return
    }

    const graceDeadline = focusGraceUntil.get(windowId)

    if (graceDeadline != null && Date.now() < graceDeadline) {
      return
    }

    const key = String(input.key || '').toLowerCase()
    const accel = (isMac ? input.meta : input.control) && !input.alt

    if (key === 'w' && accel && !input.shift) {
      if (typeof event.preventDefault === 'function') {
        event.preventDefault()
      }

      opts?.sendClosePreviewRequested?.()

      return
    }

    if (key === 'r' && accel && !input.shift) {
      if (typeof event.preventDefault === 'function') {
        event.preventDefault()
      }

      opts?.sendPreviewNavCommand?.('reload')

      return
    }

    const mod = isMac ? input.meta : input.control

    if (!mod || input.alt) {
      return
    }

    if (key === '0') {
      if (input.shift) {
        return
      }

      if (typeof event.preventDefault === 'function') {
        event.preventDefault()
      }

      const defaultLevel = opts?.getDefaultZoomLevel?.() ?? 0
      opts?.setAndPersistZoomLevel?.(window, defaultLevel)
    } else if (key === '=' || key === '+') {
      if (typeof event.preventDefault === 'function') {
        event.preventDefault()
      }

      const step = opts?.getZoomStep?.() ?? 0.1
      const current = webContents.getZoomLevel()
      opts?.setAndPersistZoomLevel?.(window, current + step)
    } else if (key === '-') {
      if (input.shift) {
        return
      }

      if (typeof event.preventDefault === 'function') {
        event.preventDefault()
      }

      const step = opts?.getZoomStep?.() ?? 0.1
      const current = webContents.getZoomLevel()
      opts?.setAndPersistZoomLevel?.(window, current - step)
    }
  }

  webContents.on('before-input-event', inputHandler)

  const zoomChangedHandler = (event: Electron.Event, zoomDirection: 'in' | 'out') => {
    if (typeof event.preventDefault === 'function') {
      event.preventDefault()
    }

    const step = opts?.getZoomStep?.() ?? 0.1
    const delta = zoomDirection === 'in' ? step : -step
    const current = webContents.getZoomLevel()
    opts?.setAndPersistZoomLevel?.(window, current + delta)
  }

  webContents.on('zoom-changed', zoomChangedHandler)

  return () => {
    window.off('focus', focusHandler)
    focusGraceUntil.delete(windowId)

    if (!webContents.isDestroyed()) {
      webContents.off('before-input-event', inputHandler)
      webContents.off('zoom-changed', zoomChangedHandler)
    }
  }
}
