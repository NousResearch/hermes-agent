import assert from 'node:assert/strict'

import { beforeEach, test, vi } from 'vitest'

const { execMock } = vi.hoisted(() => ({ execMock: vi.fn() }))

vi.mock('node:child_process', () => ({ execFileSync: execMock }))

import { readProcessCommandLineSync } from './process-command-line'

beforeEach(() => {
  execMock.mockReset()
})

test('windows platform probes PowerShell CIM, never bare ps (#102660)', () => {
  execMock.mockReturnValue('python -m hermes gateway run\n')

  const out = readProcessCommandLineSync(1234, 'win32')

  assert.equal(out, 'python -m hermes gateway run\n')
  assert.equal(execMock.mock.calls.length, 1)
  const [command, args] = execMock.mock.calls[0]
  assert.equal(command, 'powershell.exe')
  assert.deepEqual(args.slice(0, 3), ['-NoProfile', '-NonInteractive', '-Command'])
  // The pid rides as its own argv element, never inside the command text.
  assert.equal(args[args.length - 1], '1234')
})

test('posix platform keeps the ps probe as a literal argv list', () => {
  execMock.mockReturnValue('python hermes\n')

  readProcessCommandLineSync(1234, 'darwin')

  const [command, args] = execMock.mock.calls[0]
  assert.equal(command, 'ps')
  assert.deepEqual(args, ['-p', '1234', '-o', 'args='])
})

test('probe failure degrades to null', () => {
  execMock.mockImplementation(() => {
    throw new Error('probe failed')
  })

  assert.equal(readProcessCommandLineSync(1234, 'win32'), null)
})

test('invalid pid is rejected without executing anything', () => {
  assert.equal(readProcessCommandLineSync(0, 'win32'), null)
  assert.equal(readProcessCommandLineSync(-1, 'darwin'), null)
  assert.equal(readProcessCommandLineSync(1.5, 'win32'), null)
  assert.equal(readProcessCommandLineSync(Number.NaN, 'win32'), null)
  assert.equal(execMock.mock.calls.length, 0)
})
