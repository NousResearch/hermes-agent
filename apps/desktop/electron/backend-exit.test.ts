import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test, vi } from 'vitest'

import { waitForBackendExit } from './backend-child'

test('backend exit escalation remains bounded when no exit or close arrives', async () => {
  vi.useFakeTimers()

  try {
    const child = Object.assign(new EventEmitter(), { exitCode: null, signalCode: null, kill: vi.fn() })
    const waiting = waitForBackendExit(child, { forceKillProcessTree: () => {} }, 20)
    await vi.advanceTimersByTimeAsync(1020)
    await waiting
    assert.equal(child.kill.mock.calls.length, 1)
    assert.equal(child.kill.mock.calls[0][0], 'SIGKILL')
    assert.equal(child.listenerCount('exit'), 0)
    assert.equal(vi.getTimerCount(), 0)
  } finally {
    vi.useRealTimers()
  }
})
