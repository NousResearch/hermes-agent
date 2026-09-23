import { expect, test, vi } from 'vitest'

import { createManagedRolloutsBridge } from './preload-managed-rollouts'

test('typed managed-rollout preload bridge maps every explicit provider route', async () => {
  const invoke = vi.fn(async (channel: string, payload?: unknown) => ({ ok: true, value: { channel, payload } }))
  const bridge = createManagedRolloutsBridge({ invoke })

  await bridge.capabilities()
  await bridge.inventory()
  await bridge.resolveTarget({ connectionIds: [], inventoryRevision: 'r1', retryOf: null })
  await bridge.preflight({ draft: {} })
  await bridge.start({ token: 't', requestId: '12345678-1234-4678-9234-567812345678' })
  await bridge.activeRevision()
  await bridge.read(null)
  await bridge.get('12345678-1234-4678-9234-567812345678')
  await bridge.command({ kind: 'stop' })
  await bridge.history({ limit: 1 })
  await bridge.events({ id: '12345678-1234-4678-9234-567812345678', limit: 1 })

  expect(invoke.mock.calls.map(([channel]) => channel)).toEqual([
    'hermes:managed-rollouts:capabilities',
    'hermes:managed-rollouts:inventory',
    'hermes:managed-rollouts:resolveTarget',
    'hermes:managed-rollouts:preflight',
    'hermes:managed-rollouts:start',
    'hermes:managed-rollouts:activeRevision',
    'hermes:managed-rollouts:read',
    'hermes:managed-rollouts:get',
    'hermes:managed-rollouts:command',
    'hermes:managed-rollouts:history',
    'hermes:managed-rollouts:events'
  ])
})

test('returns the IPC value and rejects a refused managed-rollout request', async () => {
  const capabilities = {
    protocol: 1,
    available: true,
    reason: null,
    maxConcurrency: 1,
    maxInstallations: 10
  }

  const invoke = vi.fn(async (channel: string) => channel.endsWith(':capabilities')
    ? { ok: true, value: capabilities }
    : { ok: false, code: 'forbidden', message: 'Managed rollout IPC requires a trusted sender.' })

  const bridge = createManagedRolloutsBridge({ invoke })

  expect(await bridge.capabilities()).toEqual(capabilities)

  await expect(bridge.inventory()).rejects.toThrow('Managed rollout IPC requires a trusted sender.')
})
