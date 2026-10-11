import { type BrowserWindow, ipcMain, type Rectangle, screen } from 'electron'

import type { WindowSizeMode } from './window-size-types'
import { windowSize } from './window-state'

interface WindowSizingOptions {
  enabled: boolean
  mainWindow: () => BrowserWindow | null
}

// The bounds this module last gave each window. window-state.json skips a
// window still at them: app-chosen bounds are not where the user left the
// window, and a saved onboarding size reopened the app as a 602x642 chat. A
// user move or resize leaves them, so that placement is saved.
type PlacedWindow = Pick<BrowserWindow, 'getNormalBounds' | 'isMaximized'>

const appSized = new WeakMap<PlacedWindow, Rectangle>()

export function markAppSized(win: PlacedWindow, bounds: Rectangle): void {
  appSized.set(win, bounds)
}

export function isAppSized(win: PlacedWindow): boolean {
  const given = appSized.get(win)

  if (!given || win.isMaximized()) {
    return false
  }

  const bounds = win.getNormalBounds()

  return (['x', 'y', 'width', 'height'] as const).every(key => Math.abs(bounds[key] - given[key]) <= 1)
}

// Onboarding sets the chat size outright. Normal grows each axis to the normal
// size and never shrinks one the user already made bigger.
function sizedBounds(mode: WindowSizeMode, bounds: Rectangle, workArea: Rectangle): Rectangle | null {
  const target = windowSize(mode, workArea)

  if (mode === 'onboarding') {
    return centeredBounds(workArea, target.width, target.height)
  }

  const width = Math.max(bounds.width, target.width)
  const height = Math.max(bounds.height, target.height)

  return width === bounds.width && height === bounds.height ? null : centeredBounds(workArea, width, height)
}

export function registerWindowSizing({ enabled, mainWindow }: WindowSizingOptions): void {
  ipcMain.on('hermes:window:size', (event, mode: WindowSizeMode) => {
    const win = mainWindow()

    if (!enabled || !win || win.isDestroyed() || event.sender !== win.webContents) {
      return
    }

    if ((mode !== 'normal' && mode !== 'onboarding') || win.isMaximized() || win.isFullScreen()) {
      return
    }

    const bounds = win.getBounds()
    const next = sizedBounds(mode, bounds, screen.getDisplayMatching(bounds).workArea)

    if (next) {
      markAppSized(win, next)
      win.setBounds(next, true)
    }
  })
}

function centeredBounds(workArea: Rectangle, width: number, height: number): Rectangle {
  return {
    height,
    width,
    x: Math.round(workArea.x + (workArea.width - width) / 2),
    y: Math.round(workArea.y + (workArea.height - height) / 2)
  }
}
