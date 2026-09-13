import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  configureDesktopLoginStartup,
  DESKTOP_LOGIN_ITEM_NAME,
  isDesktopLoginStartupSupported,
  isStartMinimizedLaunch,
  readDesktopLoginStartup,
  shouldFocusSecondInstance,
  shouldStartMinimizedFirstLaunch
} from './desktop-login-startup'

test('startup readback requires Windows approval for the exact user registration', () => {
  const executablePath = 'C:\\Program Files\\Hermes\\Hermes.exe'
  let launchItems = [
    { name: DESKTOP_LOGIN_ITEM_NAME, path: executablePath, args: [] as string[], scope: 'user', enabled: true }
  ]

  const options = {
    platform: 'win32',
    executablePath,
    app: {
      setLoginItemSettings() {},
      getLoginItemSettings: () => ({ openAtLogin: true, launchItems })
    }
  }

  assert.equal(readDesktopLoginStartup(options).openAtLogin, true)
  launchItems = [
    { ...launchItems[0]!, enabled: false },
    { ...launchItems[0]!, name: 'another-registration', enabled: true }
  ]
  assert.equal(readDesktopLoginStartup(options).openAtLogin, false)
  launchItems = [{ ...launchItems[0]!, enabled: true, path: 'C:\\OldHermes\\Hermes.exe' }]
  assert.equal(readDesktopLoginStartup(options).openAtLogin, false)
  launchItems = [{ ...launchItems[0]!, path: executablePath, scope: 'machine' }]
  assert.equal(readDesktopLoginStartup(options).openAtLogin, false)
  launchItems = []
  assert.equal(readDesktopLoginStartup(options).openAtLogin, false)
})

test('configureDesktopLoginStartup sends explicit disable and reads the qualified registration back', () => {
  const executablePath = 'C:\\Users\\vip\\AppData\\Local\\Hermes\\Hermes.exe'
  const calls: { get?: unknown; set?: unknown } = {}

  const app = {
    getLoginItemSettings(options?: { path?: string; args?: string[] }) {
      calls.get = options

      return { openAtLogin: false }
    },
    setLoginItemSettings(settings: { openAtLogin: boolean; path: string; args: string[] }) {
      calls.set = settings
    }
  }

  const status = configureDesktopLoginStartup({
    app,
    executablePath,
    platform: 'win32',
    openAtLogin: false
  })

  assert.deepEqual(calls.set, {
    openAtLogin: false,
    name: DESKTOP_LOGIN_ITEM_NAME,
    path: executablePath,
    args: ['--start-minimized']
  })
  assert.deepEqual(calls.get, { path: `"${executablePath}"`, args: ['--start-minimized'] })
  assert.equal(status.supported, true)
  assert.equal(status.openAtLogin, false)
})

test('non-Windows configuration is a no-op and minimized parsing requires the exact token', () => {
  let boundaryCalls = 0

  const app = {
    getLoginItemSettings() {
      boundaryCalls += 1

      return { openAtLogin: true }
    },
    setLoginItemSettings() {
      boundaryCalls += 1
    }
  }

  const status = configureDesktopLoginStartup({
    app,
    executablePath: 'C:\\Hermes.exe',
    platform: 'darwin',
    openAtLogin: true
  })

  assert.equal(status.supported, false)
  assert.equal(status.openAtLogin, false)
  assert.equal(boundaryCalls, 0)
  assert.equal(isStartMinimizedLaunch(['Hermes.exe', '--start-minimized']), true)
  assert.equal(isStartMinimizedLaunch(['Hermes.exe', '--start-minimized=true']), false)
  assert.equal(isStartMinimizedLaunch(['Hermes.exe', 'x--start-minimized']), false)
  assert.equal(isDesktopLoginStartupSupported({ platform: 'win32', isPackaged: false, isTest: false }), false)
  assert.equal(
    shouldStartMinimizedFirstLaunch({
      platform: 'win32',
      argv: ['Hermes.exe', '--start-minimized'],
      hasColdDeepLink: false
    }),
    true
  )
  assert.equal(
    shouldStartMinimizedFirstLaunch({
      platform: 'win32',
      argv: ['Hermes.exe', '--start-minimized'],
      hasColdDeepLink: true
    }),
    false
  )
  assert.equal(
    shouldFocusSecondInstance({
      platform: 'win32',
      argv: ['Hermes.exe', '--start-minimized'],
      hasDeepLink: false
    }),
    false
  )
  assert.equal(
    shouldFocusSecondInstance({
      platform: 'win32',
      argv: ['Hermes.exe', '--start-minimized'],
      hasDeepLink: true
    }),
    false
  )
  assert.equal(shouldFocusSecondInstance({ platform: 'win32', argv: ['Hermes.exe'], hasDeepLink: false }), true)
})
