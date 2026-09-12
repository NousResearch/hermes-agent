import { expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  exposeInMainWorld: vi.fn(),
  send: vi.fn<(channel: string, line: string) => void>(),
  sendSync: vi.fn((channel: string): unknown => channel === 'hermes:feature-flags'
    ? { localModels: true, guestOnboarding: true }
    : {})
}))

vi.mock('electron', () => ({
  contextBridge: { exposeInMainWorld: host.exposeInMainWorld },
  ipcRenderer: { send: host.send, sendSync: host.sendSync },
  webFrame: {},
  webUtils: {}
}))

it('publishes the feature flags answered by main before the renderer starts', async (): Promise<void> => {
  await import('./preload')
  const registration = host.exposeInMainWorld.mock.calls.find(([name]): boolean => name === 'hermesDesktop')

  expect(registration).toBeDefined()
  expect(registration![1]).toMatchObject({ localModelsEnabled: true, guestOnboardingEnabled: true })
  expect(host.sendSync).toHaveBeenCalledWith('hermes:feature-flags')
})

it('forwards full renderer error lines through the exposed bridge', async (): Promise<void> => {
  await import('./preload')
  const registration = host.exposeInMainWorld.mock.calls.find(([name]): boolean => name === 'hermesDesktop')
  const bridge = registration?.[1] as { logLine?: (line: string) => void } | undefined
  const line: string = '[renderer error:main] Prompt failed: database is locked\n    at saveSession (session.ts:12)'

  expect(bridge?.logLine).toBeTypeOf('function')
  bridge?.logLine?.(line)
  expect(host.send).toHaveBeenCalledExactlyOnceWith('hermes:logs:renderer-line', line)
})
