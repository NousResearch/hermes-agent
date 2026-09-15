import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import {
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  normalizeHermesHomeRoot,
  pathEnvKey,
  POSIX_SANE_PATH_ENTRIES
} from './backend-env'

test('backend env scrubs PYTHONPATH and PYTHONHOME', () => {
  const env = buildDesktopBackendEnv({
    currentEnv: {
      PATH: '/usr/bin:/bin',
      PYTHONPATH: '/leaked/other/checkout',
      PYTHONHOME: '/leaked/python'
    },
    platform: 'darwin'
  })

  assert.equal(env.PYTHONPATH, '')
  assert.equal(env.PYTHONHOME, '')
})

test('POSIX backend PATH keeps the inherited PATH first and appends missing sane entries', () => {
  const env = buildDesktopBackendEnv({
    currentEnv: { PATH: '/opt/homebrew/bin:/usr/bin:/bin' },
    platform: 'darwin'
  })

  const entries = env.PATH.split(':')
  assert.equal(entries[0], '/opt/homebrew/bin', 'inherited PATH keeps precedence')
  assert.equal(entries.filter(entry => entry === '/opt/homebrew/bin').length, 1, 'no duplicates')

  for (const expected of POSIX_SANE_PATH_ENTRIES) {
    assert.ok(entries.includes(expected), `${expected} should be present`)
  }
})

test('Windows PATH casing and delimiter are preserved without POSIX sane entries', () => {
  const env = buildDesktopBackendEnv({
    currentEnv: { Path: 'C:\\Windows\\System32;C:\\Windows' },
    platform: 'win32'
  })

  assert.equal(env.Path, 'C:\\Windows\\System32;C:\\Windows')
  assert.equal(env.PATH, undefined)
})

test('buildDesktopBackendEnv forces PYTHONUTF8 unless the user set it explicitly', () => {
  const defaulted = buildDesktopBackendEnv({
    currentEnv: { PATH: '/usr/bin' },
    platform: 'darwin'
  })

  assert.equal(defaulted.PYTHONUTF8, '1')

  const optedOut = buildDesktopBackendEnv({
    currentEnv: { PATH: '/usr/bin', PYTHONUTF8: '0' },
    platform: 'darwin'
  })

  assert.equal(optedOut.PYTHONUTF8, '0')
})

test('normalizeHermesHomeRoot maps profile homes back to the global Hermes root', () => {
  assert.equal(
    normalizeHermesHomeRoot('/Users/test/.hermes/profiles/oracle', { pathModule: path.posix }),
    '/Users/test/.hermes'
  )
  assert.equal(
    normalizeHermesHomeRoot('C:\\Users\\test\\AppData\\Local\\hermes\\profiles\\oracle', { pathModule: path.win32 }),
    'C:\\Users\\test\\AppData\\Local\\hermes'
  )
  assert.equal(normalizeHermesHomeRoot('/Users/test/.hermes', { pathModule: path.posix }), '/Users/test/.hermes')
})

test('pathEnvKey finds the platform-cased PATH key', () => {
  assert.equal(pathEnvKey({ Path: 'x' }, 'win32'), 'Path')
  assert.equal(pathEnvKey({ PATH: 'x' }, 'win32'), 'PATH')
  assert.equal(pathEnvKey({}, 'win32'), 'PATH')
  assert.equal(pathEnvKey({ Path: 'x' }, 'darwin'), 'PATH')
})

test('appendUniquePathEntries flattens, dedupes, and preserves first occurrence', () => {
  assert.equal(appendUniquePathEntries(['/a:/b', ['/b', '/c'], '', null], { delimiter: ':' }), '/a:/b:/c')
})
