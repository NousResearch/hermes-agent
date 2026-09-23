import { describe, expect, it, vi } from 'vitest'

import { createDesktopPowerRuntime } from './desktop-power-runtime'

describe('desktop power registration', () => {
  it('registers once and shares battery and resume events with windows', () => {
    const listeners = new Map<string, () => void>()
    const handlers = new Map<string, () => boolean>()
    const send = vi.fn()
    const window = { isDestroyed: () => false, webContents: { isDestroyed: () => false, send } }

    const powerMonitor = {
      isOnBatteryPower: () => false,
      on: vi.fn((name: string, callback: () => void) => listeners.set(name, callback))
    }

    const attachPowerResumeRemoteRevalidation = vi.fn()

    const runtime = createDesktopPowerRuntime({
      BrowserWindow: { getAllWindows: () => [window] },
      attachPowerResumeRemoteRevalidation,
      getMainWindow: () => window as any,
      ipcMain: { handle: (name: string, callback: () => boolean) => handlers.set(name, callback) },
      powerMonitor,
      rememberLog: vi.fn(),
      revalidateSuspectPoolAfterResume: vi.fn()
    })

    runtime.registerPowerResumeListeners()
    runtime.registerPowerResumeListeners()

    expect(powerMonitor.on).toHaveBeenCalledTimes(4)
    expect(attachPowerResumeRemoteRevalidation).toHaveBeenCalledOnce()
    expect(handlers.get('hermes:power-battery:get')!()).toBe(false)

    listeners.get('resume')!()
    listeners.get('on-battery')!()

    expect(send).toHaveBeenCalledWith('hermes:power-resume')
    expect(send).toHaveBeenCalledWith('hermes:power-battery', true)
    expect(handlers.get('hermes:power-battery:get')!()).toBe(true)
  })
})
