import assert from 'node:assert/strict'

import { test } from 'vitest'

import { expandWindowsEnvRefs, parseRegQueryValue, readWindowsUserEnvVar } from './windows-user-env'

// ── parseRegQueryValue ─────────────────────────────────────────────────────

test('parseRegQueryValue extracts a REG_SZ value', () => {
  const out = ['', 'HKEY_CURRENT_USER\\Environment', '    RABBIT_HOME    REG_SZ    F:\\Rabbit\\data', ''].join('\r\n')
  assert.equal(parseRegQueryValue(out, 'RABBIT_HOME'), 'F:\\Rabbit\\data')
})

test('parseRegQueryValue matches the name case-insensitively', () => {
  const out = 'HKEY_CURRENT_USER\\Environment\r\n    Rabbit_Home    REG_EXPAND_SZ    %USERPROFILE%\\h\r\n'
  assert.equal(parseRegQueryValue(out, 'RABBIT_HOME'), '%USERPROFILE%\\h')
})

test('parseRegQueryValue preserves spaces inside the value', () => {
  const out = '    RABBIT_HOME    REG_SZ    C:\\Program Files\\Rabbit\r\n'
  assert.equal(parseRegQueryValue(out, 'RABBIT_HOME'), 'C:\\Program Files\\Rabbit')
})

test('parseRegQueryValue returns null when the value line is absent', () => {
  const out = 'HKEY_CURRENT_USER\\Environment\r\n    Path    REG_SZ    C:\\x\r\n'
  assert.equal(parseRegQueryValue(out, 'RABBIT_HOME'), null)
  assert.equal(parseRegQueryValue('', 'RABBIT_HOME'), null)
  assert.equal(parseRegQueryValue('garbage', 'RABBIT_HOME'), null)
})

// ── expandWindowsEnvRefs ───────────────────────────────────────────────────

test('expandWindowsEnvRefs expands %VAR% case-insensitively', () => {
  assert.equal(expandWindowsEnvRefs('%UserProfile%\\h', { USERPROFILE: 'C:\\Users\\jeff' }), 'C:\\Users\\jeff\\h')
})

test('expandWindowsEnvRefs leaves literal paths and unknown refs intact', () => {
  assert.equal(expandWindowsEnvRefs('F:\\Rabbit\\data', {}), 'F:\\Rabbit\\data')
  assert.equal(expandWindowsEnvRefs('%NOPE%\\x', {}), '%NOPE%\\x')
})

// ── readWindowsUserEnvVar ──────────────────────────────────────────────────

test('readWindowsUserEnvVar returns null off Windows without spawning', () => {
  let spawned = false

  const exec = () => {
    spawned = true

    return ''
  }

  assert.equal(readWindowsUserEnvVar('RABBIT_HOME', { platform: 'linux', exec }), null)
  assert.equal(spawned, false)
})

test('readWindowsUserEnvVar queries HKCU\\Environment and expands the value', () => {
  const calls = []

  const exec = (cmd, args) => {
    calls.push([cmd, args])

    return 'HKEY_CURRENT_USER\\Environment\r\n    RABBIT_HOME    REG_EXPAND_SZ    %DRIVE%\\Rabbit\r\n'
  }

  const value = readWindowsUserEnvVar('RABBIT_HOME', {
    platform: 'win32',
    env: { DRIVE: 'F:' },
    exec
  })

  assert.equal(value, 'F:\\Rabbit')
  assert.deepEqual(calls, [['reg', ['query', 'HKCU\\Environment', '/v', 'RABBIT_HOME']]])
})

test('readWindowsUserEnvVar returns null when reg exits non-zero (value missing)', () => {
  const exec = () => {
    throw new Error('reg exited 1')
  }

  assert.equal(readWindowsUserEnvVar('RABBIT_HOME', { platform: 'win32', exec }), null)
})

test('readWindowsUserEnvVar returns null for an empty value', () => {
  const exec = () => '    RABBIT_HOME    REG_SZ    \r\n'
  assert.equal(readWindowsUserEnvVar('RABBIT_HOME', { platform: 'win32', exec }), null)
})
