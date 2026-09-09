import { expect, it, vi } from 'vitest'

import { registerScopedBackendRecycleIpc } from './backend-recycle'
import { normalizeRegistry } from './connection-registry'

const bridge = vi.hoisted(() => ({
  exposed: {} as Record<string, (...args: unknown[]) => Promise<unknown>>,
  handlers: new Map<string, (_event: unknown, target: unknown) => unknown>(),
  invoke: vi.fn(async (channel: string, target: unknown) => {
    const handler = bridge.handlers.get(channel)

    if (!handler) {
      throw new Error(`No handler for ${channel}`)
    }

    return handler({}, target)
  })
}))

vi.mock('electron', () => ({
  contextBridge: {
    exposeInMainWorld: (_name: string, api: typeof bridge.exposed) => {
      bridge.exposed = api
    }
  },
  ipcRenderer: { sendSync: () => ({}), invoke: bridge.invoke },
  webFrame: {},
  webUtils: {}
}))

it('exposes the scoped lifecycle capability end-to-end through preload and IPC without invoking legacy recycle', async () => {
  const stopPrimary = vi.fn(async () => {})
  const stopPool = vi.fn(async () => {})
  const startLocal = vi.fn(async () => ({ mode: 'local', profile: 'coder', wsUrl: 'ws://127.0.0.1:9876/api/ws' }))

  const primary = {
    process: { killed: false, exitCode: null },
    connectionPromise: Promise.resolve({ mode: 'local', profile: 'coder' })
  }

  registerScopedBackendRecycleIpc(
    {
      handle: (channel, handler) => {
        bridge.handlers.set(channel, handler)
      }
    },
    {
      readState: () => ({
        registry: normalizeRegistry(null),
        routeOptions: { primaryProfile: 'coder' },
        primary,
        pool: new Map()
      }),
      stopPrimary,
      stopPool,
      startLocal
    }
  )
  await import('./preload')
  const target = { connectionId: 'local', profile: 'coder' }

  expect(await bridge.exposed.backendRestartStatus(target)).toEqual({ supported: true })
  expect(stopPrimary).not.toHaveBeenCalled()
  expect(await bridge.exposed.restartBackendFor(target)).toEqual({
    mode: 'local',
    profile: 'coder',
    wsUrl: 'ws://127.0.0.1:9876/api/ws'
  })
  expect(startLocal).toHaveBeenCalledWith(target, { primary: true, key: 'coder' })
  expect(stopPrimary.mock.invocationCallOrder[0]).toBeLessThan(startLocal.mock.invocationCallOrder[0])
  await expect(bridge.exposed.restartBackendFor({ profile: 'coder' })).rejects.toThrow('invalid-target')
  expect(startLocal).toHaveBeenCalledOnce()
  expect(stopPrimary).toHaveBeenCalledOnce()
  expect(stopPool).not.toHaveBeenCalled()
  expect(bridge.invoke.mock.calls.map(([channel]) => channel)).toEqual([
    'hermes:backend:restart-capability',
    'hermes:backend:restart-for',
    'hermes:backend:restart-for'
  ])
  expect(typeof bridge.exposed.recycleBackend).toBe('function')
})
