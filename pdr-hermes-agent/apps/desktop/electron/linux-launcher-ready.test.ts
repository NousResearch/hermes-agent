import assert from 'node:assert/strict'

import { test } from 'vitest'

import { LAUNCHER_READY_FD_ENV, notifyLauncherWindowRevealed } from './linux-launcher-ready'

function createIo() {
  const writes: Array<[number, string]> = []
  const closes: number[] = []

  return {
    closes,
    io: {
      closeSync: (fd: number) => {
        closes.push(fd)
      },
      writeSync: (fd: number, data: string) => {
        writes.push([fd, data])

        return data.length
      }
    },
    writes
  }
}

test('writes one byte to the launcher fd, closes it and consumes the variable', () => {
  const env: NodeJS.ProcessEnv = { [LAUNCHER_READY_FD_ENV]: '7' }
  const { closes, io, writes } = createIo()

  assert.equal(notifyLauncherWindowRevealed(env, io), true)
  assert.deepEqual(writes, [[7, 'r']])
  assert.deepEqual(closes, [7])
  assert.equal(LAUNCHER_READY_FD_ENV in env, false)

  // A second reveal must not touch fd 7 again: the number may now belong to another file.
  assert.equal(notifyLauncherWindowRevealed(env, io), false)
  assert.equal(writes.length, 1)
})

test('a launch without the variable, or with a garbage value, is a no-op', () => {
  const { io, writes } = createIo()

  assert.equal(notifyLauncherWindowRevealed({}, io), false)
  assert.equal(notifyLauncherWindowRevealed({ [LAUNCHER_READY_FD_ENV]: 'seven' }, io), false)
  assert.equal(notifyLauncherWindowRevealed({ [LAUNCHER_READY_FD_ENV]: '-1' }, io), false)
  assert.deepEqual(writes, [])
})
