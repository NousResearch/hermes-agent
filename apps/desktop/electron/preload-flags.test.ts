import { expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  exposeInMainWorld: vi.fn(),
  sendSync: vi.fn((channel: string): unknown => channel === 'hermes:feature-flags'
    ? { localModels: true, guestOnboarding: true }
    : {})
}))

vi.mock('electron', () => ({
  contextBridge: { exposeInMainWorld: host.exposeInMainWorld },
  ipcRenderer: { sendSync: host.sendSync },
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
