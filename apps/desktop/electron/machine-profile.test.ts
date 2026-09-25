import { statSync } from 'node:fs'
import * as os from 'node:os'

import { beforeEach, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  getGPUInfo: vi.fn(),
  getLocale: vi.fn((): string => 'fr-CA'),
  handle: vi.fn()
}))

vi.mock('electron', () => ({
  app: { getGPUInfo: host.getGPUInfo, getLocale: host.getLocale },
  ipcMain: { handle: host.handle }
}))

import type { MachineProfile } from './machine-profile'
import { registerMachineProfile } from './machine-profile'

beforeEach((): void => {
  host.handle.mockReset()
  host.getGPUInfo.mockReset()
  host.getLocale.mockClear()
})

function handler(): () => Promise<MachineProfile> {
  registerMachineProfile()
  const registration = host.handle.mock.calls.find(([channel]): boolean => channel === 'hermes:machine:profile')
  expect(registration).toBeDefined()

  return registration![1] as () => Promise<MachineProfile>
}

it('registers the renderer channel and reports native machine facts with the OS locale and GPU', async (): Promise<void> => {
  host.getGPUInfo.mockResolvedValue({ gpuDevice: [{ vendorId: 0x8086 }, { vendorId: 0x10de }] })
  const result = await handler()()
  const { birthtimeMs } = statSync(os.homedir())

  expect(result).toMatchObject({
    ageDays: birthtimeMs > 0 ? Math.max(0, Math.floor((Date.now() - birthtimeMs) / 86_400_000)) : null,
    arch: process.arch,
    locale: 'fr-CA',
    nvidia: true,
    platform: process.platform,
    release: os.release(),
    username: os.userInfo().username
  })
  expect(typeof result.model).toBe('string')
  expect(host.getGPUInfo).toHaveBeenCalledWith('basic')
})

it('keeps machine facts available when Chromium cannot enumerate GPUs', async (): Promise<void> => {
  host.getGPUInfo.mockRejectedValue(new Error('GPU info unavailable'))
  const result = await handler()()

  expect(result.nvidia).toBe(false)
  expect(result.platform).toBe(process.platform)
  expect(result.locale).toBe('fr-CA')
})
