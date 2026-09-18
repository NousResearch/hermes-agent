/**
 * `hermes:fs:reveal` answers what it did. `shell.showItemInFolder` selects an existing
 * item and silently no-ops on a missing one, and a remote backend's paths are missing on
 * this computer by construction — so a `true` for them left the renderer nothing to say.
 */
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, test, vi } from 'vitest'

const handlers = new Map<string, (...args: unknown[]) => unknown>()
const shown: string[] = []

vi.mock('electron', () => ({
  ipcMain: {
    handle: (channel: string, fn: (...args: unknown[]) => unknown) => {
      handlers.set(channel, fn)
    }
  },
  shell: {
    showItemInFolder: (target: string) => {
      shown.push(target)
    },
    openPath: async () => ''
  }
}))

const { registerFsIpc } = await import('./fs-ipc')

let home: string

beforeEach(() => {
  home = mkdtempSync(path.join(os.tmpdir(), 'hermes-reveal-'))
  shown.length = 0
  handlers.clear()
  registerFsIpc({
    hermesHome: home,
    readActiveDesktopProfile: () => null,
    expandUserPath: value => (value.startsWith('~/') ? path.join(home, value.slice(2)) : value),
    resolveRequestedPathForIpc: value => value,
    directoryExists: () => true,
    resolveGitBinary: () => 'git'
  })
})

afterEach(() => {
  rmSync(home, { force: true, recursive: true })
})

async function reveal(target: string) {
  const fn = handlers.get('hermes:fs:reveal')

  assert.ok(fn, 'handler registered for hermes:fs:reveal')

  return fn({}, target)
}

test('reveal reports false for a path that is not on this computer, and only shows one that is', async () => {
  const here = path.join(home, 'notes.md')

  writeFileSync(here, 'x')

  assert.equal(await reveal('/home/hermes/.hermes/attachments/report.zip'), false)
  assert.deepEqual(shown, [])

  assert.equal(await reveal(here), true)
  assert.deepEqual(shown, [here])
})

test('reveal expands ~ the way every other fs door does before deciding', async () => {
  writeFileSync(path.join(home, 'tilde.txt'), 'x')

  assert.equal(await reveal('~/tilde.txt'), true)
  assert.deepEqual(shown, [path.join(home, 'tilde.txt')])
})
