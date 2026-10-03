import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import { showHandoffDialog } from './handoff-dialog'
import { runPrimaryBackendStartup } from './primary-backend-startup'

class DialogWindow extends EventEmitter {
  visible = true
  destroyed = false
  isVisible = () => this.visible
  isDestroyed = () => this.destroyed
}

test('an unanswered handoff dialog is parented while local backend startup continues', async () => {
  const window = new DialogWindow()
  let dialogParent: unknown
  let respond!: (result: { response: number; checkboxChecked: boolean }) => void
  let notification: ReturnType<typeof showHandoffDialog> | undefined

  const startup = await runPrimaryBackendStartup({
    assertCurrentAttempt: () => {},
    resolveRemote: async () => null,
    connectRemote: async () => null,
    waitForLocalStart: async () => {
      notification = showHandoffDialog(window, { message: 'Update failed' }, parent => {
        dialogParent = parent

        return new Promise(resolve => {
          respond = resolve
        })
      })
    },
    prepareLocalBackend: () => ({ command: 'hermes' }),
    waitForDecision: async () => 'continue-local',
    ensureLocalRuntime: async backend => backend
  })

  assert.equal(dialogParent, window)
  assert.equal(startup.kind, 'local')
  respond({ response: 2, checkboxChecked: false })
  assert.deepEqual(await notification, { response: 2, checkboxChecked: false })
})

test('a hidden handoff parent defers the dialog and closing it never opens a parentless modal', async () => {
  const window = new DialogWindow()

  window.visible = false

  const parents: unknown[] = []

  const showMessageBox = async (parent: DialogWindow) => {
    parents.push(parent)

    return { response: 0, checkboxChecked: false }
  }

  const pending = showHandoffDialog(window, { message: 'Update needs attention' }, showMessageBox)
  assert.equal(parents.length, 0)
  window.visible = true
  window.emit('show')
  assert.equal((await pending)?.response, 0)
  assert.deepEqual(parents, [window])

  const closed = new DialogWindow()
  closed.visible = false
  const cancelled = showHandoffDialog(closed, { message: 'Update needs attention' }, showMessageBox)
  closed.destroyed = true
  closed.emit('closed')
  assert.equal(await cancelled, null)
  assert.deepEqual(parents, [window])
  assert.equal(closed.listenerCount('show'), 0)
})
