import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import type { BrowserWindow } from 'electron'
import { test } from 'vitest'

import { confirmUpdateDiscardsUnsaved, installUnsavedEditsUnloadGuard, UNSAVED_EDITS_ERROR } from './unsaved-edits-guard'

function fakeWindow() {
  const webContents = new EventEmitter()

  return { isDestroyed: () => false, webContents } as unknown as BrowserWindow & { webContents: EventEmitter }
}

function unload(win: BrowserWindow & { webContents: EventEmitter }): boolean {
  let allowed = false

  win.webContents.emit('will-prevent-unload', { preventDefault: () => (allowed = true) })

  return allowed
}

test('leaving or updating with unsaved edits asks first; a confirmed quit lets the window go', async () => {
  const asked: string[] = []
  let answer = 0
  let quitting = false

  const host = {
    showMessageBox: async (_parent: BrowserWindow, options: { message?: string }) => {
      asked.push(options.message ?? '')

      return { response: answer }
    },
    showMessageBoxSync: (_parent: BrowserWindow, options: { message?: string }) => {
      asked.push(options.message ?? '')

      return answer
    }
  }

  const win = fakeWindow()

  installUnsavedEditsUnloadGuard(win, { allowUnload: () => quitting, host, unsavedFiles: () => ['notes.md'] })

  assert.equal(unload(win), false, 'Keep Editing keeps the window and its draft')
  answer = 1
  assert.equal(unload(win), true, 'Discard Changes closes or reloads')
  assert.deepEqual(asked, ['You have unsaved changes in 1 file.', 'You have unsaved changes in 1 file.'])

  quitting = true
  assert.equal(unload(win), true)
  assert.equal(asked.length, 2, 'the quit prompt already asked; the unload must not ask again')

  answer = 0
  assert.equal((await confirmUpdateDiscardsUnsaved([{ count: 0, titles: [], unsavedFiles: ['notes.md'] }], win, host))?.error, UNSAVED_EDITS_ERROR)
  answer = 1
  assert.equal(await confirmUpdateDiscardsUnsaved([{ count: 0, titles: [], unsavedFiles: ['notes.md'] }], win, host), null)
  assert.equal(await confirmUpdateDiscardsUnsaved([{ count: 1, titles: ['Fix login'] }], win, host), null)
  assert.equal(asked.length, 4, 'no unsaved files: the update never asks')
})
