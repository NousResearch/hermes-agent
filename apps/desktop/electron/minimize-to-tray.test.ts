import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { BrowserWindow } from 'electron'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

const native = vi.hoisted(() => ({
  app: { on: vi.fn(), quit: vi.fn(), dock: { show: vi.fn(), hide: vi.fn() } },
  ipc: new Map<string, (...args: any[]) => any>(),
  windows: [] as any[],
  trays: [] as any[],
  fail: false
}))

vi.mock('electron', () => ({
  app: native.app,
  ipcMain: { handle: (name: string, fn: (...args: any[]) => any) => native.ipc.set(name, fn) },
  BrowserWindow: { getAllWindows: () => native.windows },
  Menu: { buildFromTemplate: (items: unknown[]) => items },
  nativeImage: { createFromPath: () => ({ isEmpty: () => false, resize: () => ({}) }) },
  Tray: class extends EventEmitter {
    destroyed = false
    menu: any[] = []
    constructor() {
      super()

      if (native.fail) {
        throw new Error('Unavailable tray')
      }

      native.trays.push(this)
    }
    setToolTip() {}
    setContextMenu(menu: any[]) {
      this.menu = menu
    }
    destroy() {
      this.destroyed = true
    }
    isDestroyed() {
      return this.destroyed
    }
  }
}))
vi.mock('./tray-host', () => ({ watchLinuxTrayHost: async () => () => {} }))

import { createMinimizeToTray } from './minimize-to-tray'

class Window extends EventEmitter {
  visible = true
  minimized = false
  destroyed = false
  skipped = false
  focused = false
  webContents = { send: vi.fn() }
  isDestroyed() {
    return this.destroyed
  }
  isVisible() {
    return this.visible
  }
  isMinimized() {
    return this.minimized
  }
  setSkipTaskbar(on: boolean) {
    this.skipped = on
  }
  hide() {
    this.visible = false
    this.focused = false
    this.emit('hide')
  }
  show() {
    this.visible = true
    this.emit('show')
  }
  focus() {
    this.focused = true
  }
  showInactive() {
    this.visible = true
    this.emit('show')
  }
  restore() {
    this.minimized = false
    this.visible = true
    this.emit('restore')
  }
  minimize() {
    this.minimized = true
    this.emit('minimize')
  }
  close() {
    const event = { preventDefault: vi.fn() }
    this.emit('close', event)

    if (!event.preventDefault.mock.calls.length) {
      this.destroyed = true
      this.emit('closed')
    }

    return event
  }
}

let home: string
beforeEach(() => {
  home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-tray-'))
  native.windows = []
  native.trays = []
  native.fail = false
  vi.clearAllMocks()
})
afterEach(() => fs.rmSync(home, { recursive: true, force: true }))

function setup() {
  const main = new Window()
  const peer = new Window()
  native.windows.push(main, peer)
  let handoff = false

  const controller = createMinimizeToTray({
    preferencesPath: path.join(home, 'minimize-to-tray.json'),
    getIconPath: () => 'icon.png',
    restoreMainWindow: () => main.showInactive(),
    isQuittingForHandoff: () => handoff,
    log: vi.fn()
  })

  controller.registerWindow(main as unknown as BrowserWindow, { closeToTray: true })
  controller.registerWindow(peer as unknown as BrowserWindow)

  return {
    controller,
    main,
    peer,
    handoff: () => {
      handoff = true
    }
  }
}

function flushDeferredHide() {
  return new Promise<void>(resolve => setImmediate(resolve))
}

const originalPlatform = Object.getOwnPropertyDescriptor(process, 'platform')

function setPlatform(platform: string) {
  Object.defineProperty(process, 'platform', { value: platform })
}

function restorePlatform() {
  if (originalPlatform) {
    Object.defineProperty(process, 'platform', originalPlatform)
  }
}

test('tray hide defers past the minimize dispatch (#119252)', async () => {
  const { controller, main } = setup()
  await controller.start()
  await controller.setEnabled(true)

  // Hiding synchronously inside the minimize dispatch wedges the native
  // minimized flag on Windows; hide must run after dispatch returns.
  main.minimize()
  expect(main.visible).toBe(true)
  await flushDeferredHide()
  expect(main.visible).toBe(false)
})

test('Windows restore re-activates the window instead of showInactive (#119252)', async () => {
  setPlatform('win32')

  try {
    const { controller, main } = setup()
    await controller.start()
    await controller.setEnabled(true)

    main.minimize()
    await flushDeferredHide()
    controller.restore()
    expect(main.visible).toBe(true)
    expect(main.minimized).toBe(false)
    expect(main.skipped).toBe(false)
    expect(main.focused).toBe(true)
  } finally {
    restorePlatform()
  }
})

test('a restore before the deferred hide fires cancels the stale hide', async () => {
  const { controller, main } = setup()
  await controller.start()
  await controller.setEnabled(true)

  main.minimize()
  // User restores (taskbar/shortcut) before the deferred hide fires.
  main.restore()
  await flushDeferredHide()
  expect(main.visible).toBe(true)
  expect(main.minimized).toBe(false)
})

test('opt-in minimize and primary Close preserve windows while explicit Quit still exits', async () => {
  const { controller, main, peer } = setup()
  expect(await controller.start()).toEqual({ enabled: false, available: false })
  main.minimize()
  expect(main.visible).toBe(true)
  main.restore()
  await native.ipc.get('hermes:minimize-to-tray:set')!(null, true)
  expect(native.ipc.get('hermes:minimize-to-tray:get')!()).toEqual({ enabled: true, available: true })
  main.minimize()
  await flushDeferredHide()
  expect(main.destroyed).toBe(false)
  expect(main.visible).toBe(false)
  expect(peer.visible).toBe(true)

  if (process.platform === 'darwin') {
    expect(native.app.dock.hide).not.toHaveBeenCalled()
  }

  peer.minimize()
  await flushDeferredHide()
  expect(peer.visible).toBe(false)

  if (process.platform === 'darwin') {
    expect(native.app.dock.hide).toHaveBeenCalled()
  }

  if (process.platform === 'win32') {
    expect(main.skipped && peer.skipped).toBe(true)
  }

  native.trays[0].menu[0].click()
  expect(main.visible && peer.visible).toBe(true)
  expect(main.minimized || peer.minimized).toBe(false)
  expect(main.skipped || peer.skipped).toBe(false)
  peer.close()
  expect(peer.destroyed).toBe(true)
  main.minimize()
  native.trays[0].menu[2].click()
  expect(native.app.quit).toHaveBeenCalledOnce()
  // A cancelled guard doesn't call beginQuit; hide remains enabled.
  controller.restore()
  main.minimize()
  await flushDeferredHide()
  expect(main.destroyed).toBe(false)
  expect(main.visible).toBe(false)
  // X/Alt+F4 hides the primary, but an accepted explicit quit closes it.
  expect(main.close().preventDefault).toHaveBeenCalledOnce()
  expect(main.destroyed).toBe(false)
  expect(main.visible).toBe(false)
  expect(native.trays[0].destroyed).toBe(false)
  native.trays[0].menu[0].click()
  expect(main.visible).toBe(true)
  controller.beginQuit()
  expect(main.close().preventDefault).not.toHaveBeenCalled()
  expect(main.destroyed).toBe(true)
  native.app.on.mock.calls.find(([event]) => event === 'will-quit')![1]()
  expect(native.trays[0].destroyed).toBe(true)
})

test('persistence, disabling, failed tray creation, and handoff never strand hidden windows', async () => {
  const first = setup()
  await first.controller.start()
  await first.controller.setEnabled(true)
  first.main.minimize()
  await first.controller.setEnabled(false)
  expect(first.main.visible).toBe(true)
  expect(native.trays[0].destroyed).toBe(true)
  await first.controller.setEnabled(true)
  const restarted = setup()
  expect(await restarted.controller.start()).toEqual({ enabled: true, available: true })
  restarted.handoff()
  restarted.main.close()
  expect(restarted.main.destroyed).toBe(true)
  const failed = setup()
  native.fail = true
  expect(await failed.controller.start()).toEqual({ enabled: true, available: false })
  failed.main.minimize()
  expect(failed.main.visible).toBe(true)
  failed.main.close()
  expect(failed.main.destroyed).toBe(true)
  fs.writeFileSync(path.join(home, 'minimize-to-tray.json'), 'bad json')
  const disabled = setup()
  expect(await disabled.controller.start()).toEqual({ enabled: false, available: false })
  disabled.main.close()
  expect(disabled.main.destroyed).toBe(true)
})
