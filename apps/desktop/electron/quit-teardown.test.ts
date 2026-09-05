import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { backendQuitNeedsWait, createQuitTeardownCoordinator } from './quit-teardown'

function deferred() {
  let resolve!: () => void

  const promise = new Promise<void>(done => {
    resolve = done
  })

  return { promise, resolve }
}

test('remote-only cleanup does not cancel the original quit', () => {
  const cleanup = vi.fn()
  const requestFinalQuit = vi.fn()
  const coordinator = createQuitTeardownCoordinator(requestFinalQuit)

  const shouldPrevent = coordinator.begin([{ run: cleanup, waitForCompletion: false }])

  assert.equal(shouldPrevent, false)
  assert.equal(cleanup.mock.calls.length, 1)
  assert.equal(requestFinalQuit.mock.calls.length, 0)
})

test('a settled remote descriptor is not backend work that requires a deferred quit', () => {
  assert.equal(
    backendQuitNeedsWait({
      connectionPending: false,
      poolPending: false,
      processAttached: false,
      shutdownPending: false
    }),
    false
  )

  for (const key of ['connectionPending', 'poolPending', 'processAttached', 'shutdownPending'] as const) {
    assert.equal(
      backendQuitNeedsWait({
        connectionPending: false,
        poolPending: false,
        processAttached: false,
        shutdownPending: false,
        [key]: true
      }),
      true,
      `${key} must defer quit`
    )
  }
})

test('one final quit waits for every required teardown branch', async () => {
  const backend = deferred()
  const ssh = deferred()
  const requestFinalQuit = vi.fn()
  const backendRun = vi.fn(() => backend.promise)
  const sshRun = vi.fn(() => ssh.promise)
  const coordinator = createQuitTeardownCoordinator(requestFinalQuit)

  const tasks = [
    { run: backendRun, waitForCompletion: true },
    { run: sshRun, waitForCompletion: true }
  ]

  assert.equal(coordinator.begin(tasks), true)
  assert.equal(coordinator.begin(tasks), true, 'a re-entrant quit stays deferred while teardown is pending')
  assert.equal(backendRun.mock.calls.length, 1)
  assert.equal(sshRun.mock.calls.length, 1)

  backend.resolve()
  await Promise.resolve()
  assert.equal(requestFinalQuit.mock.calls.length, 0)

  ssh.resolve()
  await Promise.all([backend.promise, ssh.promise])
  await Promise.resolve()
  assert.equal(requestFinalQuit.mock.calls.length, 1)

  assert.equal(coordinator.begin(tasks), false, 'the coordinator must not cancel its own final quit')
  assert.equal(backendRun.mock.calls.length, 1)
  assert.equal(sshRun.mock.calls.length, 1)
})

test('teardown failure still releases the final quit after all branches settle', async () => {
  const requestFinalQuit = vi.fn()
  const coordinator = createQuitTeardownCoordinator(requestFinalQuit)

  assert.equal(
    coordinator.begin([
      {
        run: () => {
          throw new Error('cleanup failed')
        },
        waitForCompletion: true
      }
    ]),
    true
  )

  await Promise.resolve()
  await Promise.resolve()
  assert.equal(requestFinalQuit.mock.calls.length, 1)
})
