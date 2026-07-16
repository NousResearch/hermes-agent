// Real-behavior tests for the Windows 8.3 short-path helper.
//
// These exercise the exported toShortPath() function by stubbing
// node:child_process, so they run on any OS and verify the actual logic
// rather than checking source text.

import assert from 'node:assert/strict'
import type { execFileSync } from 'node:child_process'

import { beforeEach, test } from 'vitest'

import { toShortPath } from './windows-short-path'

let stubExecFileSync: typeof execFileSync
let captured: { file: string; args: string[]; options: Record<string, unknown> } | null

beforeEach(() => {
  stubExecFileSync = ((() => {
    throw new Error('unexpected execFileSync call')
  }) as unknown) as typeof execFileSync
  captured = null
})

function withExecFileSync(value: string | Error, fn: () => void) {
  stubExecFileSync = ((...args: Parameters<typeof execFileSync>) => {
    captured = { file: args[0], args: args[1] as string[], options: (args[2] as Record<string, unknown>) ?? {} }

    if (value instanceof Error) {throw value}

    return value
  }) as typeof execFileSync

  fn()
}

test('toShortPath returns the short path when cmd expansion succeeds', () => {
  const original = 'C:\\Program Files\\Git\\cmd\\git.exe'
  const short = 'C:\\PROGRA~1\\Git\\cmd\\git.exe'

  let result = original

  withExecFileSync(short, () => {
    result = toShortPath(original, { execFileSync: stubExecFileSync })
  })

  assert.equal(result, short)
})

test('toShortPath falls back to the original path when cmd expansion throws', () => {
  const original = 'C:\\Program Files\\Git\\cmd\\git.exe'

  let result = ''

  withExecFileSync(new Error('cmd not found'), () => {
    result = toShortPath(original, { execFileSync: stubExecFileSync })
  })

  assert.equal(result, original)
})

test('toShortPath falls back to the original path when expansion returns the same path', () => {
  const original = 'C:\\some\\path\\git.exe'

  let result = ''

  withExecFileSync(original, () => {
    result = toShortPath(original, { execFileSync: stubExecFileSync })
  })

  assert.equal(result, original)
})

test('toShortPath falls back to the original path when expansion returns empty', () => {
  const original = 'C:\\Program Files\\Git\\cmd\\git.exe'

  let result = ''

  withExecFileSync('', () => {
    result = toShortPath(original, { execFileSync: stubExecFileSync })
  })

  assert.equal(result, original)
})

test('toShortPath invokes cmd.exe with the correct for-variable short-path expansion', () => {
  const original = 'C:\\Program Files\\Git\\cmd\\git.exe'
  const short = 'C:\\PROGRA~1\\Git\\cmd\\git.exe'

  withExecFileSync(short, () => {
    toShortPath(original, { execFileSync: stubExecFileSync })
  })

  assert.equal(captured?.file, 'cmd.exe')
  assert.deepEqual(captured?.args, ['/c', `for %A in ("${original}") do @echo %~sA`])
  assert.equal(captured?.options.timeout, 5000)
  assert.equal(captured?.options.windowsHide, true)
  assert.equal(captured?.options.encoding, 'utf8')
})
