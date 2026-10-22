import { statSync } from 'node:fs'
import * as os from 'node:os'

import { beforeEach, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  getLocale: vi.fn((): string => 'fr-CA'),
  handle: vi.fn()
}))

vi.mock('node:os', async importOriginal => {
  const actual = await importOriginal<typeof os>()

  return { ...actual, hostname: vi.fn(actual.hostname) }
})

vi.mock('electron', () => ({
  app: { getLocale: host.getLocale },
  ipcMain: { handle: host.handle }
}))

import type { MachineProfile } from './machine-profile'
import { registerMachineProfile } from './machine-profile'

beforeEach((): void => {
  vi.mocked(os.hostname).mockReset()
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
    username: os.userInfo().username,
    hostname: expect.any(String)
  })
})

it.each([
  ['192.168.1.20', ''],
  ['::1', ''],
  ['localhost', ''],
  ['LOCALHOST.local', ''],
  ['foo.local', 'foo'],
  ['  foo.local  ', 'foo'],
  ['', '']
])('reports hostname %j as the display name %j', async (hostname, expected): Promise<void> => {
  vi.mocked(os.hostname).mockReturnValue(hostname)
  expect((await handler()()).hostname).toBe(expected)
})
