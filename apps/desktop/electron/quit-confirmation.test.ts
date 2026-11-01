import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createQuitConfirmation } from './quit-confirmation'
import { createQuitTeardownCoordinator } from './quit-teardown'
import { createBootstrapCoordinator } from './ssh-bootstrap-coordinator'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

test('late work during teardown cannot cancel the final quit and strand sealed SSH', async () => {
  const ssh = createBootstrapCoordinator()
  const cleanup = deferred<void>()
  const dialog = deferred<{ response: number }>()
  let activeWork = false
  let exits = 0
  const showPrompt = vi.fn(() => (activeWork ? dialog.promise : null))
  const hold = createQuitConfirmation({ shouldSkip: () => false, showPrompt, requestQuit: quit })
  const teardown = createQuitTeardownCoordinator(quit)

  function quit() {
    let prevented = false

    const event = {
      preventDefault: () => {
        prevented = true
      }
    }

    if (hold(event)) {
      return
    }

    ssh.shutdown()

    if (teardown.begin([{ run: () => cleanup.promise, waitForCompletion: true }])) {
      event.preventDefault()
    }

    if (!prevented) {
      exits += 1
    }
  }

  await ssh.start('vps', 'config', async () => 'connected')
  quit()
  activeWork = true // renderer catches up while asynchronous cleanup is pending
  quit() // repeated Cmd-Q must not reopen confirmation either
  cleanup.resolve()
  await cleanup.promise
  await Promise.resolve()
  await Promise.resolve()
  dialog.resolve({ response: 0 }) // a cancelled late prompt used to leave SSH sealed
  await dialog.promise
  await Promise.resolve()

  assert.equal(exits, 1, 'the admitted quit must finish instead of leaving a disconnected window')
  assert.equal(showPrompt.mock.calls.length, 1, 'only pre-teardown work can require confirmation')
})

test('Keep Running leaves SSH usable; a later confirmed quit waits for cleanup once', async () => {
  const ssh = createBootstrapCoordinator()
  let dialog = deferred<{ response: number }>()
  const showPrompt = vi.fn(() => dialog.promise)
  const cleanup = deferred<void>()

  const stop = vi.fn(() => {
    ssh.shutdown()

    return cleanup.promise
  })

  let exits = 0
  const hold = createQuitConfirmation({ shouldSkip: () => false, showPrompt, requestQuit: quit })
  const teardown = createQuitTeardownCoordinator(quit)

  function quit() {
    const event = { preventDefault: vi.fn() }

    if (!hold(event) && !teardown.begin([{ run: stop, waitForCompletion: true }])) {
      exits += 1
    }
  }

  quit()
  quit()
  assert.equal(showPrompt.mock.calls.length, 1)
  dialog.resolve({ response: 0 })
  await dialog.promise
  assert.equal(stop.mock.calls.length, 0)
  assert.equal(await ssh.start('vps', 'config', async () => 'reconnected'), 'reconnected')

  dialog = deferred<{ response: number }>()
  quit()
  dialog.resolve({ response: 1 })
  await dialog.promise
  assert.equal(stop.mock.calls.length, 1)
  assert.equal(exits, 0)
  cleanup.resolve()
  await cleanup.promise
  await Promise.resolve()
  await Promise.resolve()
  assert.equal(exits, 1)
  assert.equal(stop.mock.calls.length, 1)
  assert.equal(showPrompt.mock.calls.length, 2)
})
