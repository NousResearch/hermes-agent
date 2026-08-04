import * as path from 'node:path'

import { beforeEach, describe, expect, it, vi } from 'vitest'

const store = new Map<string, string>()
const trayInstances: Array<{ destroyed: boolean; tooltip: string | null }> = []

vi.mock('electron', () => ({
  app: {
    getPath: () => '/userdata',
    getAppPath: () => '/app',
    isReady: () => true
  },
  Menu: {
    buildFromTemplate: (template: unknown) => ({ template })
  },
  nativeImage: {
    createFromPath: () => ({ isEmpty: () => true }),
    createEmpty: () => ({ isEmpty: () => true })
  },
  Tray: class {
    destroyed = false
    tooltip: string | null = null
    constructor() {
      trayInstances.push(this)
    }
    setToolTip(value: string) {
      this.tooltip = value
    }
    setContextMenu() {}
    on() {}
    destroy() {
      this.destroyed = true
    }
  }
}))

vi.mock('node:fs', () => ({
  readFileSync: (p: string) => {
    if (!store.has(p)) {
      throw new Error('ENOENT')
    }

    return store.get(p)
  },
  writeFileSync: (p: string, data: string) => {
    store.set(p, data)
  },
  existsSync: () => false
}))

import { destroyTray, isTrayEnabled, loadTrayPrefs, setTrayEnabled } from './tray'

const callbacks = { onShow: () => {}, onQuit: () => {} }

describe('tray', () => {
  beforeEach(() => {
    store.clear()
    trayInstances.length = 0
    setTrayEnabled(false, callbacks)
  })

  it('defaults to disabled with no prefs file', () => {
    expect(loadTrayPrefs()).toBe(false)
    expect(isTrayEnabled()).toBe(false)
  })

  it('persists the enabled choice and reloads it', () => {
    setTrayEnabled(true, callbacks)
    expect(store.get(path.join('/userdata', 'tray-prefs.json'))).toContain('"closeToTray":true')
    // reload reads the persisted file back into module state
    expect(loadTrayPrefs()).toBe(true)
    expect(isTrayEnabled()).toBe(true)
    setTrayEnabled(false, callbacks)
    expect(loadTrayPrefs()).toBe(false)
  })

  it('creates the tray only while enabled and destroys it on disable', () => {
    setTrayEnabled(true, callbacks)
    expect(trayInstances).toHaveLength(1)
    expect(trayInstances[0].tooltip).toBe('Hermes')
    setTrayEnabled(false, callbacks)
    expect(trayInstances[0].destroyed).toBe(true)
  })

  it('destroyTray is idempotent', () => {
    setTrayEnabled(true, callbacks)
    destroyTray()
    destroyTray()
    expect(trayInstances[0].destroyed).toBe(true)
  })
})
