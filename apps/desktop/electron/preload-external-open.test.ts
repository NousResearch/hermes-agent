import { expect, it, vi } from 'vitest'

// Tripwire for the dead-wiring class: main broadcasts `hermes:external-open-
// failed` and the dialog subscribes via `onExternalOpenFailed` — with no
// preload forwarder in between the dialog never fires and the failure is
// silent in every window. The dialog guards with an optional-chain, so
// typecheck can never catch the missing listener; this does.

const host = vi.hoisted(() => ({
  exposeInMainWorld: vi.fn(),
  send: vi.fn<(channel: string, ...args: unknown[]) => void>(),
  sendSync: vi.fn((channel: string): unknown => (channel === 'hermes:feature-flags' ? {} : {})),
  on: vi.fn((channel: string, listener: (...args: unknown[]) => void) => {
    handlers.set(channel, listener)
  }),
  removeListener: vi.fn((channel: string) => {
    handlers.delete(channel)
  })
}))

const handlers = new Map<string, (...args: unknown[]) => void>()

vi.mock('electron', () => ({
  contextBridge: { exposeInMainWorld: host.exposeInMainWorld },
  ipcRenderer: { send: host.send, sendSync: host.sendSync, on: host.on, removeListener: host.removeListener },
  webFrame: {},
  webUtils: {}
}))

it('exposes onExternalOpenFailed and forwards the failure payload', async (): Promise<void> => {
  await import('./preload')
  const registration = host.exposeInMainWorld.mock.calls.find(([name]): boolean => name === 'hermesDesktop')

  expect(registration).toBeDefined()
  const bridge = registration![1] as Record<string, (callback: (payload: unknown) => void) => () => void>

  expect(typeof bridge.onExternalOpenFailed).toBe('function')

  const received: unknown[] = []
  const unsubscribe = bridge.onExternalOpenFailed(payload => received.push(payload))

  const listener = handlers.get('hermes:external-open-failed')

  expect(listener).toBeDefined()
  listener?.({}, { url: 'file:///tmp/gone.html', message: 'Open external file failed: missing.', code: 'missing-file' })

  expect(received).toEqual([
    { url: 'file:///tmp/gone.html', message: 'Open external file failed: missing.', code: 'missing-file' }
  ])

  unsubscribe()
  expect(handlers.has('hermes:external-open-failed')).toBe(false)
})
