import assert from 'node:assert/strict'

import { test } from 'vitest'

import { updateConnectionsBeforeLocal } from './update-order'

test('a local handoff waits for remote updates that outlive its exit deadline', async () => {
  let finishRemote: () => void = () => {}

  const remote = new Promise<void>(resolve => {
    finishRemote = resolve
  })

  const started: string[] = []
  const connections = [{ kind: 'local' }, { kind: 'ssh' }, { kind: 'remote' }]

  const result = updateConnectionsBeforeLocal(connections, async connection => {
    started.push(connection.kind)

    if (connection.kind === 'ssh') {
      await remote
    }

    return connection.kind
  })

  await Promise.resolve()
  await Promise.resolve()
  const startedBeforeRemoteCompletion = [...started]
  finishRemote()
  assert.deepEqual(await result, ['local', 'ssh', 'remote'])
  assert.deepEqual(startedBeforeRemoteCompletion, ['ssh', 'remote'])
  assert.deepEqual(started, ['ssh', 'remote', 'local'])
})

test('an empty selection does nothing and a local-only selection runs once', async () => {
  let calls = 0
  const update = async () => ++calls
  assert.deepEqual(await updateConnectionsBeforeLocal([], update), [])
  assert.deepEqual(await updateConnectionsBeforeLocal([{ kind: 'local' }], update), [1])
})

test('remote updates still run concurrently with each other', async () => {
  let running = 0
  let maxRunning = 0

  const results = await updateConnectionsBeforeLocal(
    [{ kind: 'ssh' }, { kind: 'remote' }, { kind: 'ssh' }],
    async connection => {
      running += 1
      maxRunning = Math.max(maxRunning, running)
      await new Promise(resolve => setTimeout(resolve, 5))
      running -= 1

      return connection.kind
    }
  )

  assert.deepEqual(results, ['ssh', 'remote', 'ssh'])
  assert.equal(maxRunning, 3)
})
