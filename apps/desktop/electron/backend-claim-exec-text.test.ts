import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { expect, test, vi } from 'vitest'

const execFileMock = vi.hoisted(() => vi.fn())
const spawnMock = vi.hoisted(() => vi.fn())

vi.mock('node:child_process', () => ({ execFile: execFileMock, spawn: spawnMock }))

import { execText, execTextWithIgnoredStdin } from './backend-claim'

function spawnedChild() {
  const child = Object.assign(new EventEmitter(), {
    killed: false,
    kill: vi.fn(() => {
      child.killed = true

      return true
    }),
    stderr: new EventEmitter(),
    stdout: new EventEmitter()
  })

  return child
}

test('SSH effective-config probes use ignored stdin with bounded captured output', async () => {
  const child = spawnedChild()

  spawnMock.mockImplementationOnce(() => {
    queueMicrotask(() => {
      child.stdout.emit('data', ' hostname remote.example\\n')
      child.emit('close', 0, null)
    })

    return child
  })

  await assert.doesNotReject(execTextWithIgnoredStdin('ssh', ['-G', '--', 'remote.example'], { timeout: 10_000 }))

  expect(spawnMock).toHaveBeenCalledWith(
    'ssh',
    ['-G', '--', 'remote.example'],
    expect.objectContaining({ stdio: ['ignore', 'pipe', 'pipe'] })
  )
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

test('ignored-stdin probes reject once for a nonzero exit and retain stderr', async () => {
  const child = spawnedChild()

  spawnMock.mockImplementationOnce(() => {
    queueMicrotask(() => {
      child.stderr.emit('data', 'bad config')
      child.emit('close', 255, null)
      child.emit('error', new Error('late spawn error'))
    })

    return child
  })

  await assert.rejects(execTextWithIgnoredStdin('ssh', ['-G', '--', 'remote.example']), error => {
    assert.match(String(error), /bad config/)

    assert.equal((error as { stderr?: string }).stderr, 'bad config')

    return true
  })
})

test('ignored-stdin probes kill and reject when captured output exceeds their bound', async () => {
  const child = spawnedChild()

  spawnMock.mockImplementationOnce(() => {
    queueMicrotask(() => child.stdout.emit('data', 'too much'))

    return child
  })

  await assert.rejects(execTextWithIgnoredStdin('ssh', ['-G', '--', 'remote.example'], { maxBuffer: 3 }), error => {
    assert.equal((error as { code?: string }).code, 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER')

    return true
  })

  assert.equal(child.kill.mock.calls.length, 1)
})
