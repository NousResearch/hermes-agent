import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  MAX_PAGE_SIZE,
  MAX_SNAPSHOT_BYTES,
  createManagedRolloutIpcHandler,
  type ManagedRolloutIpcAdapter
} from './managed-rollout-ipc'

const ID = '12345678-1234-4678-9234-567812345678'
const REQUEST = '22345678-1234-4678-9234-567812345678'

function targetRows(count: number): Array<Record<string, unknown>> {
  return Array.from({ length: count }, (_, index) => ({
    installId: `install-${index}`,
    phase: index === count - 1 ? 'updated' : 'queued',
    wave: Math.floor(index / 120),
    identity: {
      label: `fixture-target-${index}`,
      displayAddress: `fixture-${index}.invalid`
    }
  }))
}

function adapterFor(snapshot: unknown, events: unknown[] = []): {
  adapter: ManagedRolloutIpcAdapter
  calls: { history: number[]; events: number[]; commands: string[] }
} {
  const calls = { history: [] as number[], events: [] as number[], commands: [] as string[] }
  const adapter: ManagedRolloutIpcAdapter = {
    capabilities: async () => ({ protocol: 1, available: true, reason: null, maxConcurrency: 4, maxInstallations: 500 }),
    activeRevision: async () => 7,
    get: async () => snapshot,
    command: async command => {
      calls.commands.push(command.kind)
      return { ok: true, id: command.id, revision: command.revision }
    },
    history: async page => {
      calls.history.push(page.limit)
      return { items: events.slice(0, page.limit), nextCursor: null }
    },
    events: async page => {
      calls.events.push(page.limit)
      return { items: events.slice(0, page.limit), nextCursor: null }
    }
  }

  return { adapter, calls }
}

test('500-target snapshot stays below the encoded 8 MiB IPC bound', async () => {
  const snapshot = {
    revision: 7,
    rolloutId: ID,
    phase: 'running',
    data: { targets: targetRows(500) }
  }
  const encodedBytes = Buffer.byteLength(JSON.stringify(snapshot), 'utf8')
  assert.ok(encodedBytes < MAX_SNAPSHOT_BYTES)

  const { adapter } = adapterFor(snapshot)
  const handler = createManagedRolloutIpcHandler(adapter, () => true)
  const result = await handler({ sender: {} }, 'get', { id: ID })

  assert.equal(result.ok, true)
})

test('unchanged revision polling and terminal command do not require a full snapshot', async () => {
  let revisionCalls = 0
  const snapshot = { revision: 9, rolloutId: ID, phase: 'completed', data: { targets: targetRows(500) } }
  const fixture = adapterFor(snapshot)
  fixture.adapter.activeRevision = async () => {
    revisionCalls += 1
    return 9
  }
  const handler = createManagedRolloutIpcHandler(fixture.adapter, () => true)

  assert.deepEqual(await handler({ sender: {} }, 'activeRevision', undefined), { ok: true, value: 9 })
  assert.deepEqual(await handler({ sender: {} }, 'activeRevision', undefined), { ok: true, value: 9 })
  const command = await handler(
    { sender: {} },
    'command',
    {
      id: ID,
      expectedRevision: 9,
      requestId: REQUEST,
      action: 'stop',
      installId: null,
      reason: null,
      promotionPolicy: null
    }
  )

  assert.equal(revisionCalls, 2)
  assert.equal(command.ok, true)
  assert.deepEqual(fixture.calls.commands, ['stop'])
})

test('history and event pages remain bounded at the protocol limit', async () => {
  const events = Array.from({ length: 100 }, (_, index) => ({ sequence: index + 1, kind: 'completed' }))
  const fixture = adapterFor({ revision: 1, rolloutId: ID, phase: 'completed', data: {} }, events)
  const handler = createManagedRolloutIpcHandler(fixture.adapter, () => true)

  const history = await handler({ sender: {} }, 'history', { cursor: 'h0', limit: MAX_PAGE_SIZE })
  const page = await handler({ sender: {} }, 'events', { id: ID, cursor: 'e0', limit: MAX_PAGE_SIZE })

  assert.equal(history.ok, true)
  assert.equal(page.ok, true)
  assert.deepEqual(fixture.calls.history, [MAX_PAGE_SIZE])
  assert.deepEqual(fixture.calls.events, [MAX_PAGE_SIZE])
})

test('an oversized terminal snapshot fails closed without losing the command path', async () => {
  const oversized = {
    revision: 11,
    rolloutId: ID,
    phase: 'completed',
    data: { terminalReceipt: 'x'.repeat(MAX_SNAPSHOT_BYTES + 1) }
  }
  const fixture = adapterFor(oversized)
  const handler = createManagedRolloutIpcHandler(fixture.adapter, () => true)

  const snapshot = await handler({ sender: {} }, 'get', { id: ID })
  const command = await handler(
    { sender: {} },
    'command',
    {
      id: ID,
      expectedRevision: 11,
      requestId: REQUEST,
      action: 'stop',
      installId: null,
      reason: null,
      promotionPolicy: null
    }
  )

  assert.deepEqual(snapshot, {
    ok: false,
    code: 'snapshot-too-large',
    message: 'Managed rollout snapshot exceeds 8 MiB.'
  })
  assert.equal(command.ok, true)
  assert.deepEqual(fixture.calls.commands, ['stop'])
})
