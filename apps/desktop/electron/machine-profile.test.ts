import { statSync } from 'node:fs'
import * as os from 'node:os'

import { beforeEach, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  getLocale: vi.fn((): string => 'fr-CA'),
  handle: vi.fn()
}))

vi.mock('electron', () => ({
  app: { getLocale: host.getLocale },
  ipcMain: { handle: host.handle }
}))

import type { MachineProfile } from './machine-profile'
import { registerMachineProfile } from './machine-profile'

beforeEach((): void => {
  host.handle.mockReset()
  host.getLocale.mockClear()
})

function handler(): () => MachineProfile {
  registerMachineProfile()
  const registration = host.handle.mock.calls.find(([channel]): boolean => channel === 'hermes:machine:profile')
  expect(registration).toBeDefined()

  return registration![1] as () => MachineProfile
}

it('registers the renderer channel and reports native machine facts with the OS locale', (): void => {
  const result = handler()()
  const { birthtimeMs } = statSync(os.homedir())

  expect(result).toEqual({
    ageDays: birthtimeMs > 0 ? Math.max(0, Math.floor((Date.now() - birthtimeMs) / 86_400_000)) : null,
    arch: process.arch,
    locale: 'fr-CA',
    platform: process.platform,
    release: os.release(),
    username: os.userInfo().username
  })
})
