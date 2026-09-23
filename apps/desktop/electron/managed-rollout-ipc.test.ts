import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createManagedRolloutIpcHandler, MAX_PAGE_SIZE, type ManagedRolloutIpcAdapter } from './managed-rollout-ipc'

const ID = '12345678-1234-4678-9234-567812345678'
const REQUEST = '22345678-1234-4678-9234-567812345678'

function adapter(): { value: ManagedRolloutIpcAdapter; calls: string[] } {
  const calls: string[] = []
  return {
    calls,
    value: {
      capabilities: async () => ({ protocol: 1, available: true, reason: null, maxConcurrency: 1, maxInstallations: 0 }),
      activeRevision: async () => 4,
      read: async sinceRevision => ({ revision: 4, snapshot: sinceRevision === 4 ? null : { id: ID } }),
      get: async id => ({ id }),
      command: async command => {
        calls.push(command.kind)
        return { ok: true, id: command.id, revision: command.revision }
      },
      history: async page => ({ page }),
      events: async page => ({ page })
    }
  }
}

test('untrusted senders and forged methods refuse before adapter invocation', async () => {
  const fixture = adapter()
  const handler = createManagedRolloutIpcHandler(fixture.value, sender => sender === 'trusted')

  assert.equal((await handler({ sender: 'forged' }, 'command', {})).ok, false)
  assert.equal((await handler({ sender: 'trusted' }, 'arbitrary', undefined)).ok, false)
  assert.deepEqual(fixture.calls, [])
})

test('command accepts only exact typed commands and never forwards extra renderer fields', async () => {
  const fixture = adapter()
  const handler = createManagedRolloutIpcHandler(fixture.value, () => true)

  const accepted = await handler(
    { sender: {} },
    'command',
    { id: ID, expectedRevision: 4, requestId: REQUEST, action: 'pause', installId: null, reason: null, promotionPolicy: null }
  )
  const forged = await handler(
    { sender: {} },
    'command',
    { id: ID, expectedRevision: 4, requestId: REQUEST, action: 'pause', installId: null, reason: null, promotionPolicy: null, targetSha: 'a'.repeat(40) }
  )
  const acceptedExclude = await handler(
    { sender: {} },
    'command',
    {
      id: ID,
      expectedRevision: 4,
      requestId: REQUEST,
      action: 'exclude',
      installId: 'a'.repeat(32),
      reason: 'operator-request',
      promotionPolicy: null
    }
  )
  const acceptedPolicy = await handler(
    { sender: {} },
    'command',
    {
      id: ID,
      expectedRevision: 4,
      requestId: REQUEST,
      action: 'set-policy',
      installId: null,
      reason: null,
      promotionPolicy: 'manual'
    }
  )
  assert.equal(accepted.ok, true)
  assert.equal(acceptedExclude.ok, true)
  assert.equal(acceptedPolicy.ok, true)
  assert.equal(forged.ok, false)
  assert.deepEqual(fixture.calls, ['pause', 'exclude', 'set-policy'])
})

test('pages and snapshots are bounded while revision reads carry no arbitrary payload', async () => {
  const fixture = adapter()
  const handler = createManagedRolloutIpcHandler(fixture.value, () => true)

  assert.equal((await handler({ sender: {} }, 'activeRevision', {})).ok, false)
  assert.equal((await handler({ sender: {} }, 'history', { cursor: 'page-1', limit: MAX_PAGE_SIZE + 1 })).ok, false)
  assert.equal((await handler({ sender: {} }, 'events', { id: ID, cursor: 'page-1', limit: 1 })).ok, true)
  assert.equal((await handler({ sender: {} }, 'get', { id: ID })).ok, true)
})

test('new provider routes are explicit and fail closed when the provider method is unavailable', async () => {
  const fixture = adapter()
  const handler = createManagedRolloutIpcHandler(fixture.value, () => true)

  const inventory = await handler({ sender: {} }, 'inventory', undefined)
  const target = await handler(
    { sender: {} },
    'resolveTarget',
    { connectionIds: [ID], inventoryRevision: 'inventory-1', retryOf: null }
  )
  const preflight = await handler({ sender: {} }, 'preflight', { draft: {} })
  const start = await handler({ sender: {} }, 'start', { token: 'token-1', requestId: REQUEST })

  for (const result of [inventory, target, preflight, start]) {
    assert.deepEqual(result, {
      ok: false,
      code: 'unavailable',
      message: 'Managed rollout service is unavailable.'
    })
  }
})

test('read accepts only a bounded sinceRevision and preserves the revision acknowledgement shape', async () => {
  const fixture = adapter()
  const handler = createManagedRolloutIpcHandler(fixture.value, () => true)

  assert.deepEqual(await handler({ sender: {} }, 'read', { sinceRevision: 4 }), {
    ok: true,
    value: { revision: 4, snapshot: null }
  })
  assert.deepEqual(await handler({ sender: {} }, 'read', { sinceRevision: null }), {
    ok: true,
    value: { revision: 4, snapshot: { id: ID } }
  })
  assert.equal((await handler({ sender: {} }, 'read', { sinceRevision: -1 })).ok, false)
  assert.equal((await handler({ sender: {} }, 'read', { sinceRevision: 4, extra: true })).ok, false)
})

test('oversized requests fail before an adapter can mutate state', async () => {
  const fixture = adapter()
  const handler = createManagedRolloutIpcHandler(fixture.value, () => true)
  const payload = {
    id: ID,
    expectedRevision: 4,
    requestId: REQUEST,
    action: 'pause',
    installId: null,
    reason: null,
    promotionPolicy: null,
    padding: 'x'.repeat(256 * 1024)
  }

  const result = await handler({ sender: {} }, 'command', payload)

  assert.equal(result.ok, false)
  assert.deepEqual(fixture.calls, [])
})
