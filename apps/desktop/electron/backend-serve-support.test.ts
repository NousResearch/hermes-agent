import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import * as probes from './backend-probes'
import { createBackendServeSupportResolver } from './backend-serve-support'

test('concurrent serve checks share one pending probe and retain its negative result', async () => {
  let fail!: (error: Error) => void

  const pending = new Promise<void>((_resolve, reject) => {
    fail = reject
  })

  const probe = vi.spyOn(probes, 'execProbe').mockReturnValue(pending)
  const supportsServe = createBackendServeSupportResolver('/unused', () => {})
  const backend = { command: '/unused/hermes', args: ['serve'] }

  try {
    const first = supportsServe(backend)
    const second = supportsServe(backend)
    const callsWhilePending = probe.mock.calls.length
    fail(new Error('unsupported'))
    assert.deepEqual(await Promise.all([first, second]), [false, false])
    assert.equal(callsWhilePending, 1)
    assert.equal(await supportsServe(backend), false)
    assert.equal(probe.mock.calls.length, 1)
  } finally {
    probe.mockRestore()
  }
})
