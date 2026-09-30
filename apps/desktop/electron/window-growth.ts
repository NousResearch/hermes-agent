import { type BrowserWindow, ipcMain, type Rectangle, screen } from 'electron'

import type { WindowSizeMode } from './window-size-types'
import { windowSize } from './window-state'

interface WindowSizingOptions {
  enabled: boolean
  mainWindow: () => BrowserWindow | null
}

// The size this module last gave each window. window-state.json skips a window
// still at it: an app-chosen size is not where the user left the window, and a
// saved onboarding size reopened the app as a 602x642 chat.
const appSized = new WeakMap<BrowserWindow, { height: number; width: number }>()

export function isAppSized(win: BrowserWindow): boolean {
  const size = appSized.get(win)

  if (!size || win.isMaximized()) {
    return false
  }

  const { height, width } = win.getNormalBounds()

  return Math.abs(width - size.width) <= 1 && Math.abs(height - size.height) <= 1
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
      appSized.set(win, { height: next.height, width: next.width })
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
