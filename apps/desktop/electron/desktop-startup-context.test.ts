import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

vi.mock('electron', () => ({ nativeImage: { createFromPath: () => ({ isEmpty: () => false }) } }))

import { createDesktopStartupContext } from './desktop-startup-context'

test('startup context applies launch flags and native shell identity before first window', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-startup-context-'))
  const userData = path.join(root, 'user-data')
  const hermesHome = path.join(userData, 'hermes-home')
  fs.mkdirSync(hermesHome, { recursive: true })
  fs.writeFileSync(path.join(hermesHome, 'config.yaml'), 'desktop:\n  electron_flags: [--disable-gpu]\n')

  const order: string[] = []
  const switches: Array<[string, string | undefined]> = []

  const app = {
    getPath: () => userData,
    commandLine: { appendSwitch: (name: string, value?: string) => {
      switches.push([name, value])
      order.push(`switch:${name}`)
    } },
    setName: (name: string) => order.push(`name:${name}`),
    setAppUserModelId: (id: string) => order.push(`aumid:${id}`)
  }

  const Menu = { setApplicationMenu: (menu: unknown) => order.push(`menu:${menu}`) }
  const nativeTheme = { themeSource: 'system', shouldUseDarkColors: false, on: vi.fn() }

  vi.stubEnv('HERMES_HOME', '')

  try {
    const context = createDesktopStartupContext({
      app, BrowserWindow: { getAllWindows: () => [] }, Menu, crashReporter: { start: vi.fn() }, nativeTheme,
      APP_ROOT: root, USER_DATA_OVERRIDE: userData,
      IS_PACKAGED: false, IS_MAC: false, IS_WINDOWS: true, IS_WSL: false,
      DARWIN_MAJOR: 0, GLASS_SUPPORTED: true,
      directoryExists: (target: string) => fs.existsSync(target),
      unpackedPathFor: (target: string) => target
    })

    expect(context.HERMES_HOME).toBe(hermesHome)
    expect(switches).toContainEqual(['disable-gpu', undefined])
    expect(order).toEqual(['switch:disable-gpu', 'name:Hermes', 'menu:null', 'aumid:com.nousresearch.hermes'])
    expect(nativeTheme.themeSource).toBe('system')
  } finally {
    vi.unstubAllEnvs()
    fs.rmSync(root, { recursive: true, force: true })
  }
})
