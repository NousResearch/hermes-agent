import { test, expect } from 'vitest'

import { registerManagedRolloutIpc, UNAVAILABLE_MESSAGE } from './managed-rollout-ipc-runtime'

test('registers every managed-rollout channel behind the trusted sender gate', async () => {
  const handlers = new Map<string, (event: { sender: unknown }, payload: unknown) => Promise<unknown>>()
  registerManagedRolloutIpc(
    { handle: (channel, listener) => handlers.set(channel, listener as typeof handlers extends Map<string, infer V> ? V : never) },
    sender => sender === 'trusted'
  )

  expect(handlers.size).toBe(11)
  const capabilities = await handlers.get('hermes:managed-rollouts:capabilities')!({ sender: 'trusted' }, undefined)
  expect(capabilities).toMatchObject({ ok: true, value: { available: false, maxConcurrency: 0, maxInstallations: 0 } })

  const inventory = await handlers.get('hermes:managed-rollouts:inventory')!({ sender: 'trusted' }, undefined)
  expect(inventory).toEqual({ ok: false, code: 'unavailable', message: UNAVAILABLE_MESSAGE })

  const forbidden = await handlers.get('hermes:managed-rollouts:capabilities')!({ sender: 'forged' }, undefined)
  expect(forbidden).toEqual({ ok: false, code: 'forbidden', message: 'Managed rollout IPC requires a trusted sender.' })
})
