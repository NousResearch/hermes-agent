import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  closeToTrayLabels,
  closeToTrayMenuTemplate,
  decideCloseAction,
  DEFAULT_CLOSE_TO_TRAY_PREFS,
  type CloseToTrayHost,
  type CloseToTrayPrefs,
  type CloseToTrayWindow,
  type TrayHandle,
  type TrayMenuTemplateItem,
  installCloseToTray,
  normalizeCloseToTrayPrefs,
  readCloseToTrayPrefs,
  writeCloseToTrayPrefs
} from './close-to-tray'

const PREFS_PATH = '/tmp/close-to-tray.json'

function memoryFs(initial: Record<string, string> = {}) {
  const files = new Map(Object.entries(initial))

  return {
    files,
    mkdirSync: () => undefined,
    readFileSync: (filePath: string) => {
      const contents = files.get(filePath)

      if (contents === undefined) {
        const error = new Error(`ENOENT: ${filePath}`) as Error & { code?: string }
        error.code = 'ENOENT'
        throw error
      }

      return contents
    },
    renameSync: (from: string, to: string) => {
      files.set(to, files.get(from) ?? '')
      files.delete(from)
    },
    writeFileSync: (filePath: string, contents: string) => {
      files.set(filePath, contents)
    }
  }
}

function harness({ hasIcon = true, prefsFile }: { hasIcon?: boolean; prefsFile?: string } = {}) {
  const fakeFs = memoryFs(prefsFile === undefined ? {} : { [PREFS_PATH]: prefsFile })
  const menus: TrayMenuTemplateItem[][] = []
  const logs: string[] = []
  const balloons: unknown[] = []
  const trayEvents = new Map<string, () => void>()
  let createdTrays = 0
  let destroyedTrays = 0
  let hidden = 0
  let quitting = false
  let quitRequests = 0
  let shownWindows = 0

  const window: CloseToTrayWindow = {
    hide: () => {
      hidden += 1
    },
    isDestroyed: () => false
  }

  const host: CloseToTrayHost = {
    buildMenu: template => {
      menus.push(template)

      return template
    },
    createTray: () => {
      createdTrays += 1

      const tray: TrayHandle = {
        destroy: () => {
          destroyedTrays += 1
        },
        displayBalloon: options => {
          balloons.push(options)
        },
        on: (event, listener) => {
          trayEvents.set(event, listener)
        },
        setContextMenu: () => undefined,
        setToolTip: () => undefined
      }

      return tray
    },
    fileSystem: fakeFs as unknown as typeof import('node:fs'),
    getMainWindow: () => window,
    getLocale: () => 'zh-CN',
    isQuitting: () => quitting,
    log: message => {
      logs.push(message)
    },
    prefsPath: PREFS_PATH,
    requestQuit: () => {
      quitRequests += 1
    },
    resolveIconPath: () => (hasIcon ? '/icon.ico' : undefined),
    showMainWindow: () => {
      shownWindows += 1
    }
  }

  return {
    balloons,
    controller: installCloseToTray(host),
    counts: () => ({ createdTrays, destroyedTrays, hidden, quitRequests, shownWindows }),
    fakeFs,
    logs,
    menus,
    setQuitting: (next: boolean) => {
      quitting = next
    },
    trayEvents
  }
}

test('decideCloseAction hides only when the pref is on, a tray exists, and this is not a quit', () => {
  const base = { enabled: true, isQuitting: false, trayAvailable: true }

  assert.equal(decideCloseAction(base), 'hide')
  assert.equal(decideCloseAction({ ...base, enabled: false }), 'close')
  // Hiding with no tray icon strands the process with no window and no way
  // back — the pref must never outrank the tray actually existing.
  assert.equal(decideCloseAction({ ...base, trayAvailable: false }), 'close')
  assert.equal(decideCloseAction({ ...base, isQuitting: true }), 'close')
})

test('normalizeCloseToTrayPrefs falls back to the shipped default for junk', () => {
  assert.deepEqual(normalizeCloseToTrayPrefs(null), DEFAULT_CLOSE_TO_TRAY_PREFS)
  assert.deepEqual(normalizeCloseToTrayPrefs({ enabled: 'yes', noticeShown: 1 }), DEFAULT_CLOSE_TO_TRAY_PREFS)
  assert.deepEqual(normalizeCloseToTrayPrefs({ enabled: false, noticeShown: true }), {
    enabled: false,
    noticeShown: true
  })
})

test('prefs survive a round trip, and a corrupt file reads as the default', () => {
  const fileSystem = memoryFs() as unknown as typeof import('node:fs')
  const prefs: CloseToTrayPrefs = { enabled: false, noticeShown: true }

  assert.equal(writeCloseToTrayPrefs(PREFS_PATH, prefs, fileSystem), true)
  assert.deepEqual(readCloseToTrayPrefs(PREFS_PATH, fileSystem), prefs)
  assert.deepEqual(readCloseToTrayPrefs('/missing.json', fileSystem), DEFAULT_CLOSE_TO_TRAY_PREFS)

  const corrupt = memoryFs({ [PREFS_PATH]: '{not json' }) as unknown as typeof import('node:fs')
  assert.deepEqual(readCloseToTrayPrefs(PREFS_PATH, corrupt), DEFAULT_CLOSE_TO_TRAY_PREFS)
})

test('close hides the window and raises the tray, once', () => {
  const h = harness()

  assert.equal(h.controller.decideClose(), 'hide')
  h.controller.hideMainWindow()
  h.controller.hideMainWindow()

  assert.deepEqual(h.counts(), {
    createdTrays: 1,
    destroyedTrays: 0,
    hidden: 2,
    quitRequests: 0,
    shownWindows: 0
  })
  // The "where did it go" balloon is a one-time courtesy, not a nag.
  assert.equal(h.balloons.length, 1)
})

test('a build with no resolvable icon keeps the old close semantics', () => {
  const h = harness({ hasIcon: false })

  assert.equal(h.controller.decideClose(), 'close')
  assert.equal(h.counts().createdTrays, 0)
  assert.equal(h.controller.keepsProcessAlive(), false)
})

test('a tray that cannot be created never strands the app', () => {
  const h = harness()
  const controller = installCloseToTray({
    buildMenu: () => ({}),
    createTray: () => {
      throw new Error('no tray in this session')
    },
    fileSystem: h.fakeFs as unknown as typeof import('node:fs'),
    getMainWindow: () => null,
    getLocale: () => 'en',
    isQuitting: () => false,
    log: message => h.logs.push(message),
    prefsPath: PREFS_PATH,
    requestQuit: () => undefined,
    resolveIconPath: () => '/icon.ico',
    showMainWindow: () => undefined
  })

  assert.equal(controller.decideClose(), 'close')
  assert.equal(controller.keepsProcessAlive(), false)
  assert.equal(h.logs.length, 1)
})

test('a real quit is never turned into a hide', () => {
  const h = harness()

  h.setQuitting(true)

  assert.equal(h.controller.decideClose(), 'close')
  assert.equal(h.controller.keepsProcessAlive(), false)
})

test('the tray menu shows the window, toggles the pref, and quits', () => {
  const h = harness()

  h.controller.decideClose()
  const [firstMenu] = h.menus
  const menu = firstMenu.map(item => item.label ?? item.type)
  assert.deepEqual(menu, [
    closeToTrayLabels('zh-CN').show,
    'separator',
    closeToTrayLabels('zh-CN').toggle,
    'separator',
    closeToTrayLabels('zh-CN').quit
  ])
  assert.equal(firstMenu[2].checked, true)

  h.trayEvents.get('click')?.()
  assert.equal(h.counts().shownWindows, 1)

  // Unchecking the box is the only in-app way back to the old behaviour, so it
  // must take effect on the very next close and survive a restart.
  firstMenu[2].click?.()
  assert.equal(h.controller.decideClose(), 'close')
  assert.deepEqual(readCloseToTrayPrefs(PREFS_PATH, h.fakeFs as unknown as typeof import('node:fs')).enabled, false)
  assert.equal(h.menus.at(-1)?.[2].checked, false)

  h.menus.at(-1)?.[4].click?.()
  assert.equal(h.counts().quitRequests, 1)
})

test('window-all-closed only stays alive while the tray is up', () => {
  const h = harness()

  assert.equal(h.controller.keepsProcessAlive(), false)
  h.controller.decideClose()
  assert.equal(h.controller.keepsProcessAlive(), true)

  h.controller.destroyTray()
  assert.equal(h.controller.keepsProcessAlive(), false)
})

test('the menu template is data: prefs in, checkable row out', () => {
  const labels = closeToTrayLabels('en')
  const [show, , toggle, , quit] = closeToTrayMenuTemplate({ enabled: false, noticeShown: false }, labels)

  assert.deepEqual(show, { kind: 'show', label: labels.show })
  assert.deepEqual(toggle, { checked: false, kind: 'toggle', label: labels.toggle })
  assert.deepEqual(quit, { kind: 'quit', label: labels.quit })
})

test('tray copy follows the OS locale', () => {
  assert.equal(closeToTrayLabels('zh-CN').quit, closeToTrayLabels('zh-Hant-TW').quit)
  assert.equal(closeToTrayLabels('en-US').quit, 'Quit Hermes')
  assert.equal(closeToTrayLabels('').quit, 'Quit Hermes')
})