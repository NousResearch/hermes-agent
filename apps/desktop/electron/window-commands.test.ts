import { EventEmitter } from 'node:events'

import type { BrowserWindow, WebContents } from 'electron'
import { describe, expect, it, vi } from 'vitest'

import { createWindowCommands } from './window-commands'

// Just the slice of BrowserWindow / webContents the hooks touch. `sent` records
// every renderer message, so "which window heard it" is the assertion.
function fakeWindow() {
  const webContents = Object.assign(new EventEmitter(), {
    sent: [] as unknown[][],
    zoomLevel: 0,
    getZoomLevel: () => webContents.zoomLevel,
    isDestroyed: () => false,
    send: (...message: unknown[]) => void webContents.sent.push(message)
  })

  const window = Object.assign(new EventEmitter(), {
    destroyed: false,
    isDestroyed: () => window.destroyed,
    webContents
  })

  return window
}

type FakeWindow = ReturnType<typeof fakeWindow>

const asWindow = (fake: FakeWindow) => fake as unknown as BrowserWindow

function setup({ focused = null as unknown } = {}) {
  const primary = fakeWindow()
  const hud = fakeWindow()
  const leaveHud = vi.fn()

  const commands = createWindowCommands({
    focusedWebContents: () => focused as WebContents,
    isHud: window => window === asWindow(hud),
    isMac: true,
    leaveHud,
    primaryWindow: () => asWindow(primary),
    setZoomLevel: (window, level) => {
      const fake = window as unknown as FakeWindow

      fake.webContents.zoomLevel = level
    }
  })

  // What main's wireCommonWindowHandlers does for every window kind.
  const wire = (window: FakeWindow, { zoom = true } = {}) => {
    commands.installPreviewShortcut(asWindow(window))
    commands.installBrowserNavGestures(asWindow(window))

    if (zoom) {
      commands.installZoomShortcuts(asWindow(window))
    }

    return window
  }

  wire(primary)
  wire(hud)

  return { commands, hud, leaveHud, primary, wire }
}

/** A ⌘-chord keyDown through the window's `before-input-event` hooks. */
function press(window: FakeWindow, key: string) {
  const event = { preventDefault: vi.fn() }

  window.webContents.emit('before-input-event', event, { key, meta: true, type: 'keyDown' })

  return event.preventDefault.mock.calls.length > 0
}

describe('window commands act on the window they came from', () => {
  it('⌘R, ⌘W and the back/forward gestures in a second window reach that window, not the primary', () => {
    const { primary, wire } = setup()
    const second = wire(fakeWindow())
    const appCommand = { preventDefault: vi.fn() }

    expect(press(second, 'r')).toBe(true)
    expect(press(second, 'w')).toBe(true)
    second.emit('swipe', {}, 'left')
    second.emit('app-command', appCommand, 'browser-forward')

    expect(second.webContents.sent).toEqual([
      ['hermes:preview-nav', 'reload'],
      ['hermes:close-preview-requested'],
      ['hermes:preview-nav', 'back'],
      ['hermes:preview-nav', 'forward']
    ])
    expect(appCommand.preventDefault).toHaveBeenCalledOnce()
    expect(primary.webContents.sent).toEqual([])
  })

  it('menu items act on the browserWindow Electron hands the click, and the primary only without one', () => {
    const { commands, primary, wire } = setup()
    const second = wire(fakeWindow())

    commands.previewNav(asWindow(second), 'reload')
    commands.closeTab(asWindow(second))
    commands.openFolder(asWindow(second))
    // A preview guest's ⌘⇧W names its host window the same way.
    commands.closePreview(asWindow(second))

    expect(second.webContents.sent).toEqual([
      ['hermes:preview-nav', 'reload'],
      ['hermes:close-preview-requested'],
      ['hermes:open-folder-requested'],
      ['hermes:close-preview-requested']
    ])
    expect(primary.webContents.sent).toEqual([])

    // No focused window, a destroyed one, or a window none of these hooks were
    // installed on (an OAuth sign-in page): the primary, never that page.
    const signIn = fakeWindow()
    second.destroyed = true

    commands.previewNav(undefined, 'reload')
    commands.closeTab(asWindow(second))
    commands.openFolder(asWindow(signIn))

    expect(primary.webContents.sent).toEqual([
      ['hermes:preview-nav', 'reload'],
      ['hermes:close-preview-requested'],
      ['hermes:open-folder-requested']
    ])
    expect(signIn.webContents.sent).toEqual([])
  })

  it('a focused page guest still takes the gesture itself', () => {
    const guest = Object.assign(new EventEmitter(), {
      id: 1,
      navigationHistory: { canGoBack: () => true, canGoForward: () => true, goBack: vi.fn(), goForward: vi.fn() },
      getType: () => 'webview',
      isDestroyed: () => false,
      reload: vi.fn()
    })

    const { commands, primary, wire } = setup({ focused: guest })
    const second = wire(fakeWindow())

    press(second, 'r')
    second.emit('swipe', {}, 'left')
    commands.previewNav(asWindow(second), 'reload')

    expect(guest.reload).toHaveBeenCalledTimes(2)
    expect(guest.navigationHistory.goBack).toHaveBeenCalledOnce()
    expect(second.webContents.sent).toEqual([])
    expect(primary.webContents.sent).toEqual([])
  })

  it('menu zoom acts on the window ⌘+ would: itself when it zooms, else the primary', () => {
    const { commands, primary, wire } = setup()
    const second = wire(fakeWindow())
    // Pet overlay / quick entry / wake indicator: no zoom wiring.
    const helper = wire(fakeWindow(), { zoom: false })

    press(second, '=')
    commands.zoom(asWindow(second), 'zoom-in')

    expect(second.webContents.zoomLevel).toBeCloseTo(0.2)
    expect(primary.webContents.zoomLevel).toBe(0)

    // The helper never claims ⌘+, so on macOS the menu accelerator answers it.
    expect(press(helper, '=')).toBe(false)
    commands.zoom(asWindow(helper), 'zoom-in')

    expect(primary.webContents.zoomLevel).toBeCloseTo(0.1)
    expect(helper.webContents.zoomLevel).toBe(0)

    commands.zoom(asWindow(second), 'zoom-reset')
    expect(second.webContents.zoomLevel).toBe(0)
  })

  it('⌘W and File → Close in the HUD leave HUD mode instead of closing a tab anywhere', () => {
    const { commands, hud, leaveHud, primary } = setup()

    press(hud, 'w')
    commands.closeTab(asWindow(hud))

    expect(leaveHud.mock.calls).toEqual([[asWindow(hud)], [asWindow(hud)]])
    expect(hud.webContents.sent).toEqual([])
    expect(primary.webContents.sent).toEqual([])
  })
})
