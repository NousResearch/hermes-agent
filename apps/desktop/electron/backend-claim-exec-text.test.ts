import assert from 'node:assert/strict'

import { expect, test, vi } from 'vitest'

const execFileMock = vi.hoisted(() => vi.fn())

vi.mock('node:child_process', () => ({ execFile: execFileMock }))

import { execText } from './backend-claim'

test('SSH effective-config probes keep stdin open while retaining their timeout', async () => {
  const stdinEnd = vi.fn()

  execFileMock.mockImplementationOnce((_command, _args, options, done) => {
    queueMicrotask(() => done(null, 'hostname remote.example'))

    return { killed: false, stdin: { end: stdinEnd } }
  })

  await assert.doesNotReject(execText('ssh', ['-G', '--', 'remote.example'], { timeout: 10_000, keepStdinOpen: true }))

  expect(execFileMock.mock.calls[0]?.slice(0, 3)).toEqual([
    'ssh',
    ['-G', '--', 'remote.example'],
    expect.objectContaining({ encoding: 'utf8', timeout: 10_000 })
  ])
  assert.equal(stdinEnd.mock.calls.length, 0)
})

test('ordinary noninteractive probes still close stdin', async () => {
  const stdinEnd = vi.fn()

  execFileMock.mockImplementationOnce((_command, _args, _options, done) => {
    queueMicrotask(() => done(null, 'ok'))

    return { killed: false, stdin: { end: stdinEnd } }
  })

  await assert.doesNotReject(execText('ps', ['-p', '1'], { timeout: 3_000 }))

  assert.equal(stdinEnd.mock.calls.length, 1)
})

test('SSH effective-config probes preserve execText error handling', async () => {
  const stdinEnd = vi.fn()
  const timeout = Object.assign(new Error('Command timed out'), { code: 'ETIMEDOUT' })

  execFileMock.mockImplementationOnce((_command, _args, _options, done) => {
    queueMicrotask(() => done(timeout, ''))

    return { killed: false, stdin: { end: stdinEnd } }
  })

  await assert.rejects(
    execText('ssh', ['-G', '--', 'remote.example'], { timeout: 10_000, keepStdinOpen: true }),
    error => error === timeout
  )

  assert.equal(stdinEnd.mock.calls.length, 0)
})
