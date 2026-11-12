/**
 * window-state.json skips a window still at the bounds window sizing gave it,
 * but a user who moves or resizes it afterwards has placed it, and that
 * placement must be saved.
 */
import type { Rectangle, WebContents } from 'electron'
import { describe, expect, it } from 'vitest'

import { isAppSized, markAppSized, registerWindowSizing, type WindowSizingHost } from './window-growth'
import type { WindowSizeMode } from './window-size-types'

const WORK_AREA: Rectangle = { height: 1080, width: 1920, x: 0, y: 0 }

// The size IPC as the main process delivers it, on one display.
function sizingHost() {
  const listeners: Parameters<WindowSizingHost['onSizeRequest']>[0][] = []

  const host: WindowSizingHost = {
    onSizeRequest: listener => listeners.push(listener),
    workAreaOf: () => WORK_AREA
  }

  const send = (sender: WebContents, mode: WindowSizeMode) => listeners.forEach(listener => listener({ sender }, mode))

  return { host, send }
}

function placedWindow(initial: Rectangle) {
  let bounds = { ...initial }

  return {
    getNormalBounds: () => ({ ...bounds }),
    isMaximized: () => false,
    moveTo: (x: number, y: number) => {
      bounds = { ...bounds, x, y }
    }
  }
}

const GIVEN: Rectangle = { height: 642, width: 602, x: 500, y: 200 }

describe('isAppSized', () => {
  it('is true while the window sits at the bounds sizing gave it', () => {
    const win = placedWindow(GIVEN)

    markAppSized(win, GIVEN)

    expect(isAppSized(win)).toBe(true)
  })

  it('is false once the user moves the window without resizing it', () => {
    const win = placedWindow(GIVEN)

    markAppSized(win, GIVEN)
    win.moveTo(40, 60)

    expect(isAppSized(win)).toBe(false)
  })
})

// The main window as the size IPC sees it: the renderer's webContents, and bounds setBounds moves.
function mainWindow(initial: Rectangle) {
  let bounds = { ...initial }
  // SAFETY: the size IPC only compares the sender with the window's webContents by identity.
  const webContents = {} as WebContents

  return {
    getBounds: () => ({ ...bounds }),
    getNormalBounds: () => ({ ...bounds }),
    isDestroyed: () => false,
    isFullScreen: () => false,
    isMaximized: () => false,
    setBounds: (next: Rectangle) => {
      bounds = { ...next }
    },
    webContents
  }
}

describe('window sizing for the first-run intro', () => {
  it("puts an existing user's window back where it was when the intro ends", () => {
    const placed: Rectangle = { height: 900, width: 1500, x: 40, y: 60 }
    const win = mainWindow(placed)
    const { host, send } = sizingHost()
    // SAFETY: registerWindowSizing reads only the members mainWindow() fakes.
    registerWindowSizing({ host, mainWindow: () => win as never })
    const size = (mode: WindowSizeMode) => send(win.webContents, mode)

    size('onboarding')
    expect(win.getBounds()).not.toEqual(placed)

    size('normal')
    expect(win.getBounds()).toEqual(placed)
    expect(isAppSized(win)).toBe(false)
  })
})
