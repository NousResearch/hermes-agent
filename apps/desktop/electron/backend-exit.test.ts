import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test, vi } from 'vitest'

import { waitForBackendExit } from './backend-child'

test('backend exit escalation rejects within its bound when no exit or close arrives', async (): Promise<void> => {
  vi.useFakeTimers()

  try {
    const child = Object.assign(new EventEmitter(), { exitCode: null, signalCode: null, kill: vi.fn() })

    const waiting = assert.rejects(
      waitForBackendExit(child, { forceKillProcessTree: (): void => {} }, 20),
      /did not exit after escalation/
    )

    await Promise.all([waiting, vi.advanceTimersByTimeAsync(1020)])
    assert.equal(child.kill.mock.calls.length, 1)
    assert.equal(child.kill.mock.calls[0][0], 'SIGKILL')
    assert.equal(child.listenerCount('exit'), 0)
    assert.equal(vi.getTimerCount(), 0)
  } finally {
    vi.useRealTimers()
  }
})
