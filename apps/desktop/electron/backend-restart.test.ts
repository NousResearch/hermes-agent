import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { isBackendExitPending, restartLocalBackend, waitForBackendExit } from './backend-restart'

test('pending exit waits for process exit even after a kill signal was sent', () => {
  assert.equal(isBackendExitPending({ exitCode: null, signalCode: null }), true)
  assert.equal(isBackendExitPending({ exitCode: 137, signalCode: null }), false)
  assert.equal(isBackendExitPending({ exitCode: null, signalCode: 'SIGKILL' }), false)
})

test('backend exit wait escalates after timeout but resolves only after exit', async () => {
  vi.useFakeTimers()

  try {
    let exitListener!: () => void

    let forceKillCalls = 0

    let settled = false

    const child = {
      exitCode: null as number | null,
      signalCode: null as string | null,
      kill: () => undefined,
      once: (_event: 'exit', listener: () => void) => {
        exitListener = listener
      }
    }

    const wait = waitForBackendExit(child, {
      timeoutMs: 5000,
      onTimeout: () => {
        forceKillCalls += 1
      }
    }).then(() => {
      settled = true
    })

    await vi.advanceTimersByTimeAsync(5000)
    assert.equal(forceKillCalls, 1)
    assert.equal(settled, false)

    child.exitCode = 137
    exitListener()
    await wait

    assert.equal(settled, true)
  } finally {
    vi.useRealTimers()
  }
})

test('backend exit wait resolves after escalation grace when exit never arrives', async () => {
  vi.useFakeTimers()

  try {
    let forceKillCalls = 0
    let settled = false

    const child = {
      exitCode: null as number | null,
      signalCode: null as string | null,
      kill: () => undefined,
      once: () => undefined
    }

    const wait = waitForBackendExit(child, {
      timeoutMs: 5000,
      onTimeout: () => {
        forceKillCalls += 1
      }
    }).then(() => {
      settled = true
    })

    await vi.advanceTimersByTimeAsync(5000)
    assert.equal(forceKillCalls, 1)
    // Escalation ran but the child never emits exit: still pending, no hang.
    assert.equal(settled, false)

    await vi.advanceTimersByTimeAsync(2000)
    await wait

    assert.equal(settled, true)
  } finally {
    vi.useRealTimers()
  }
})

test('local restart waits for teardown before starting a new backend', async () => {
  let releaseTeardown!: () => void

  const teardownDone = new Promise<void>(resolve => {
    releaseTeardown = resolve
  })

  const events: string[] = []

  const restart = restartLocalBackend({
    teardown: async () => {
      events.push('teardown-start')
      await teardownDone
      events.push('teardown-end')
    },
    start: async () => {
      events.push('start')
      assert.deepEqual(events, ['teardown-start', 'teardown-end', 'start'])
    },
    notifyApplied: () => events.push('applied')
  })

  await Promise.resolve()
  assert.deepEqual(events, ['teardown-start'])
  releaseTeardown()

  assert.deepEqual(await restart, { ok: true, mode: 'local' })
  assert.deepEqual(events, ['teardown-start', 'teardown-end', 'start', 'applied'])
})

test('local restart reports startup failure and leaves reconnect notification available', async () => {
  let applied = 0

  const result = await restartLocalBackend({
    teardown: async () => {},
    start: async () => {
      throw new Error('backend did not become ready')
    },
    notifyApplied: () => {
      applied += 1
    }
  })

  assert.deepEqual(result, {
    ok: false,
    reason: 'restart-failed',
    message: 'backend did not become ready'
  })
  assert.equal(applied, 1)
})
