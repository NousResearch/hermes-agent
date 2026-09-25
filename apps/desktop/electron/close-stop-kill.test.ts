import assert from 'node:assert/strict'

import { test } from 'vitest'

import { finishWindowsCloseStop, type RuntimeLock } from './close-stop-kill'

function killThatThrows(message: string) {
  return () => {
    throw new Error(message)
  }
}

test('close/stop surfaces a taskkill failure and does not clear a lock a live holder still owns', () => {
  const killed: number[] = []
  const cleared: string[] = []
  const locks: RuntimeLock[] = [
    { path: 'C:\\Users\\me\\.hermes\\gateway.lock', holderPids: [4242], held: true },
    { path: 'C:\\Users\\me\\.hermes\\profiles\\other\\gateway.lock', holderPids: [], held: false }
  ]

  const result = finishWindowsCloseStop([4242], locks, {
    killTree: pid => {
      killed.push(pid)
      throw new Error('Access is denied')
    },
    isPidAlive: pid => pid === 4242,
    clearLock: path => {
      cleared.push(path)
    }
  })

  assert.deepEqual(killed, [4242], 'tree-kill stays on the owned PID only')
  assert.equal(result.taskkillFailures.length, 1)
  assert.match(result.taskkillFailures[0].error, /Access is denied/)
  assert.equal(result.taskkillFailures[0].pid, 4242)
  assert.deepEqual(result.remainingPids, [4242])
  assert.equal(result.liveFailure, true)
  assert.deepEqual(cleared, ['C:\\Users\\me\\.hermes\\profiles\\other\\gateway.lock'])
  assert.deepEqual(result.retainedLocks, ['C:\\Users\\me\\.hermes\\gateway.lock'])
})

test('close/stop inventories owned PIDs after the tree kill and clears only unheld locks', () => {
  const killed: number[] = []
  const cleared: string[] = []
  const alive = new Set([9001])

  const result = finishWindowsCloseStop(
    [4242, 9001],
    [
      { path: 'gone.lock', holderPids: [4242] },
      { path: 'still-held.lock', holderPids: [9001] },
      { path: 'foreign-held.lock', holderPids: [7777], held: true },
      { path: 'no-holder.lock', holderPids: [] }
    ],
    {
      killTree: pid => {
        killed.push(pid)
        alive.delete(pid === 4242 ? 4242 : -1)
      },
      isPidAlive: pid => alive.has(pid),
      clearLock: path => {
        cleared.push(path)
      }
    }
  )

  assert.deepEqual(killed, [4242, 9001], 'does not widen the tree-kill to foreign holders')
  assert.deepEqual(result.remainingPids, [9001])
  assert.deepEqual(result.taskkillFailures, [])
  assert.equal(result.liveFailure, true)
  assert.deepEqual(cleared.sort(), ['gone.lock', 'no-holder.lock'].sort())
  assert.deepEqual(result.retainedLocks.sort(), ['foreign-held.lock', 'still-held.lock'].sort())
})

test('a taskkill error is not discarded when the owned PID is already gone', () => {
  const result = finishWindowsCloseStop([4242], [{ path: 'stale.lock', holderPids: [4242] }], {
    killTree: killThatThrows('The process "4242" not found'),
    isPidAlive: () => false,
    clearLock: () => {}
  })

  assert.equal(result.taskkillFailures.length, 1)
  assert.match(result.taskkillFailures[0].error, /not found/)
  assert.deepEqual(result.remainingPids, [])
  assert.equal(result.liveFailure, false)
  assert.deepEqual(result.clearedLocks, ['stale.lock'])
})
