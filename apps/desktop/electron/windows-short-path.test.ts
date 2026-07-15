import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

import { toWindowsShortPath } from './windows-short-path'

test('toWindowsShortPath uses verbatim cmd arguments and returns a valid short path', () => {
  const original = String.raw`C:\Program Files\Git\cmd\git.exe`
  const expected = String.raw`C:\PROGRA~1\Git\cmd\git.exe`
  let invocation: { args: string[]; command: string; options: Record<string, any> } | null = null

  const result = toWindowsShortPath(original, (command, args, options) => {
    invocation = { args, command, options }

    return expected
  })

  assert.equal(result, expected)
  assert.equal(path.win32.isAbsolute(invocation?.command || ''), true)
  assert.equal(path.win32.basename(invocation?.command || '').toLowerCase(), 'cmd.exe')
  assert.deepEqual(invocation?.args, ['/d', '/s', '/c', 'for %A in ("%HERMES_GIT_BINARY_PATH%") do @echo %~sA'])
  assert.equal(invocation?.options.windowsVerbatimArguments, true)
  assert.equal(invocation?.options.env.HERMES_GIT_BINARY_PATH, original)
})

test('toWindowsShortPath does not invoke cmd for an already safe path', () => {
  const original = String.raw`C:\PortableGit\cmd\git.exe`
  let invoked = false

  const result = toWindowsShortPath(original, () => {
    invoked = true

    return ''
  })

  assert.equal(result, original)
  assert.equal(invoked, false)
})

test('toWindowsShortPath keeps the original when short-name expansion is unavailable', () => {
  const original = String.raw`D:\Git Install\cmd\git.exe`

  assert.equal(
    toWindowsShortPath(original, () => original),
    original
  )
  assert.equal(
    toWindowsShortPath(original, () => {
      throw new Error('8.3 names disabled')
    }),
    original
  )
})

test('toWindowsShortPath resolves the default Git for Windows install when available', () => {
  if (process.platform !== 'win32') {
    return
  }

  const git = String.raw`C:\Program Files\Git\cmd\git.exe`

  if (!fs.existsSync(git)) {
    return
  }

  const resolved = toWindowsShortPath(git)

  assert.equal(fs.existsSync(resolved), true)
  assert.equal(/\s/.test(resolved), false)
})
