import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { createNativeAppearanceController } from './native-appearance-controller'

const tempDirs: string[] = []

function fixture(platform: 'darwin' | 'win32' = 'win32') {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-appearance-'))

  tempDirs.push(userDataDir)
  const listeners: Array<() => void> = []

  const nativeTheme = {
    themeSource: 'system' as 'dark' | 'light' | 'system',
    shouldUseDarkColors: false,
    on: vi.fn((_event: string, listener: () => void) => listeners.push(listener))
  }

  const windows: any[] = []
  const log = vi.fn()

  const options = {
    userDataDir,
    nativeTheme: nativeTheme as any,
    getAllWindows: () => windows,
    log,
    isMac: platform === 'darwin',
    isWindows: platform === 'win32',
    isWsl: false,
    darwinMajor: platform === 'darwin' ? 24 : 0,
    glassSupported: true,
    titlebarHeight: 34
  }

  return { listeners, log, nativeTheme, options, userDataDir, windows }
}

function windowFixture() {
  return {
    isDestroyed: vi.fn(() => false),
    getOpacity: vi.fn(() => 1),
    setOpacity: vi.fn(),
    setBackgroundColor: vi.fn(),
    setBackgroundMaterial: vi.fn(),
    setVibrancy: vi.fn(),
    setTitleBarOverlay: vi.fn()
  }
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }

  vi.restoreAllMocks()
})

describe('native appearance controller', () => {
  it('loads persisted native appearance before the first chat window is constructed', () => {
    const fx = fixture('darwin')
    fs.writeFileSync(path.join(fx.userDataDir, 'native-theme.json'), JSON.stringify({ themeSource: 'dark' }))
    fs.writeFileSync(
      path.join(fx.userDataDir, 'translucency.json'),
      JSON.stringify({ intensity: 60, fade: 0, mode: 'glass', material: 'under-window', scope: 'all' })
    )

    const appearance = createNativeAppearanceController(fx.options)

    expect(fx.nativeTheme.themeSource).toBe('dark')
    expect(appearance.chatWindowSurfaceOptions()).toMatchObject({ visualEffectState: 'active' })
    expect(appearance.chatWindowSurfaceOptions()).not.toHaveProperty('backgroundColor')
    expect(appearance.chatWindowSurfaceOptions()).not.toHaveProperty('opacity')
    expect(appearance.getTitleBarOverlayOptions()).toEqual({ height: 34 })

    appearance.setNativeTheme('invalid')
    expect(fx.nativeTheme.themeSource).toBe('dark')
    appearance.setNativeTheme('light')
    expect(fx.nativeTheme.themeSource).toBe('light')
    expect(JSON.parse(fs.readFileSync(path.join(fx.userDataDir, 'native-theme.json'), 'utf8'))).toEqual({
      themeSource: 'light'
    })
    appearance.flushTranslucencyWrite()
  })

  it('updates only registered chat backings, diffed native properties, and the persisted setting', () => {
    const fx = fixture()
    fs.writeFileSync(
      path.join(fx.userDataDir, 'translucency.json'),
      JSON.stringify({ intensity: 0, fade: 0, mode: 'clear', material: 'under-window', scope: 'all' })
    )
    const appearance = createNativeAppearanceController(fx.options)
    const chat = windowFixture()
    const overlay = windowFixture()
    fx.windows.push(chat, overlay)
    appearance.registerChatWindow(chat as any)
    const onChanged = vi.fn()

    appearance.setTranslucency(
      { intensity: 40, fade: 0, mode: 'clear', material: 'under-window', scope: 'all' },
      onChanged
    )
    expect(onChanged).toHaveBeenCalledTimes(1)
    expect(chat.setOpacity).toHaveBeenCalledTimes(1)
    expect(overlay.setOpacity).toHaveBeenCalledTimes(1)
    expect(chat.setBackgroundColor).not.toHaveBeenCalled()
    expect(overlay.setBackgroundColor).not.toHaveBeenCalled()

    appearance.setTranslucency(
      { intensity: 40, fade: 0, mode: 'clear', material: 'under-window', scope: 'all' },
      onChanged
    )
    expect(onChanged).toHaveBeenCalledTimes(1)

    appearance.setTranslucency(
      { intensity: 60, fade: 0, mode: 'glass', material: 'under-window', scope: 'all' },
      onChanged
    )
    expect(chat.setBackgroundColor).toHaveBeenCalledWith('#00000000')
    expect(chat.setBackgroundMaterial).toHaveBeenCalled()
    expect(overlay.setBackgroundColor).not.toHaveBeenCalled()
    expect(overlay.setBackgroundMaterial).not.toHaveBeenCalled()
    appearance.flushTranslucencyWrite()
    expect(JSON.parse(fs.readFileSync(path.join(fx.userDataDir, 'translucency.json'), 'utf8'))).toEqual(
      appearance.getTranslucencyState()
    )

    appearance.installNativeThemeListener()
    appearance.installNativeThemeListener()
    expect(fx.nativeTheme.on).toHaveBeenCalledTimes(1)
    appearance.setTitleBarTheme({ background: '#123456', foreground: '#abcdef' })
    expect(chat.setTitleBarOverlay).toHaveBeenCalled()
    expect(overlay.setTitleBarOverlay).toHaveBeenCalled()
    const previousCalls = chat.setTitleBarOverlay.mock.calls.length
    appearance.setTitleBarTheme({ background: 'bad', foreground: '#abcdef' })
    expect(chat.setTitleBarOverlay).toHaveBeenCalledTimes(previousCalls)
    fx.listeners[0]()
    expect(chat.setTitleBarOverlay).toHaveBeenCalledTimes(previousCalls + 1)
  })
})
