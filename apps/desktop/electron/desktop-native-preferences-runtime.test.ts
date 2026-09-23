import { describe, expect, it, vi } from 'vitest'

import { createDesktopNativePreferencesRuntime, registerDesktopF12PreferenceIpc } from './desktop-native-preferences-runtime'

describe('desktop native preference registration', () => {
  it('keeps early synchronous flags and quit cleanup on their original channels', () => {
    const appEvents = new Map<string, () => void>()
    const ipcEvents = new Map<string, (event: any, value?: any) => void>()
    const flushTranslucencyWrite = vi.fn()
    const stopAll = vi.fn()
    const dispose = vi.fn()
    const arm = vi.fn()
    const cancel = vi.fn()

    createDesktopNativePreferencesRuntime({
      GLASS_SUPPORTED: true,
      GUEST_ONBOARDING: false,
      SKIP_INTRO: true,
      TRANSLUCENCY_SUPPORTED: true,
      app: { getPath: () => 'test-user-data', on: (name: string, handler: () => void) => appEvents.set(name, handler) },
      appearance: { flushTranslucencyWrite, setTitleBarTheme: vi.fn(), setNativeTheme: vi.fn(), setTranslucency: vi.fn() },
      createKeepAwake: () => ({ set: vi.fn() }),
      destroyKeepaliveAgents: vi.fn(),
      hudIpc: { applyHudFrost: vi.fn() },
      ipcMain: { on: (name: string, handler: (event: any, value?: any) => void) => ipcEvents.set(name, handler) },
      nativeNotifications: { dispose },
      powerSaveBlocker: {},
      quitFinalization: { arm, cancel },
      rememberLog: vi.fn(),
      sshIsolatedKeepalives: { stopAll }
    })

    const support = { returnValue: null as any }

    ipcEvents.get('hermes:translucency:support')!(support)

    expect(support.returnValue).toEqual({ glass: true, translucency: true })
    expect([...appEvents.keys()]).toEqual(['before-quit', 'will-quit', 'quit'])

    appEvents.get('before-quit')!()
    appEvents.get('will-quit')!()
    appEvents.get('quit')!()

    expect(flushTranslucencyWrite).toHaveBeenCalledOnce()
    expect(stopAll).toHaveBeenCalledOnce()
    expect(dispose).toHaveBeenCalledOnce()
    expect(arm).toHaveBeenCalledOnce()
    expect(cancel).toHaveBeenCalledOnce()
  })

  it('keeps F12 changes connected to the live main-process state', () => {
    const handlers = new Map<string, (event: any, value: any) => void>()
    const f12State = { blocked: false }

    registerDesktopF12PreferenceIpc({
      app: { getPath: () => 'C:/nonexistent-desktop-preferences-test' },
      f12State,
      ipcMain: { on: (name: string, handler: (event: any, value: any) => void) => handlers.set(name, handler) },
      rememberLog: vi.fn()
    })

    expect(handlers.has('hermes:devtools:disable-f12')).toBe(true)
    expect(f12State.blocked).toBe(false)
  })
})
