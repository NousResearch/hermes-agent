import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

const childProcess = vi.hoisted(() => ({ execFile: vi.fn() }))

vi.mock('node:child_process', async importOriginal => ({
  ...(await importOriginal<typeof import('node:child_process')>()),
  execFile: childProcess.execFile
}))

import { execText } from './backend-claim'

test('execText tags an ssh -G timeout even when SIGTERM cleanup exits zero', async () => {
  // Node's execFile can report no error when the timed-out child handles
  // SIGTERM and exits zero. child.killed is then the only timeout signal.
  const child = { killed: true, stdin: { end: vi.fn() } }
  childProcess.execFile.mockImplementation((_command, _args, _options, callback) => {
    queueMicrotask(() => callback(null, ''))
    return child
  })

  await assert.rejects(execText('ssh', ['-G', '--', 'example.test'], { timeout: 10_000 }), (error: Error & {
    timedOut?: boolean
  }) => {
    assert.equal(error.timedOut, true)
    return true
  })

  assert.equal(childProcess.execFile.mock.calls.length, 1)
  assert.equal(child.stdin.end.mock.calls.length, 1)
})
