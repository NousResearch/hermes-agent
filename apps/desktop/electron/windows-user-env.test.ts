import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  expandWindowsEnvRefs,
  parseRegQueryValue,
  readWindowsHostPath,
  readWindowsUserEnvVar
} from './windows-user-env'

// ── parseRegQueryValue ─────────────────────────────────────────────────────

test('parseRegQueryValue extracts a REG_SZ value', () => {
  const out = ['', 'HKEY_CURRENT_USER\\Environment', '    HERMES_HOME    REG_SZ    F:\\Hermes\\data', ''].join('\r\n')
  assert.equal(parseRegQueryValue(out, 'HERMES_HOME'), 'F:\\Hermes\\data')
})

test('parseRegQueryValue matches the name case-insensitively', () => {
  const out = 'HKEY_CURRENT_USER\\Environment\r\n    Hermes_Home    REG_EXPAND_SZ    %USERPROFILE%\\h\r\n'
  assert.equal(parseRegQueryValue(out, 'HERMES_HOME'), '%USERPROFILE%\\h')
})

test('parseRegQueryValue preserves spaces inside the value', () => {
  const out = '    HERMES_HOME    REG_SZ    C:\\Program Files\\Hermes\r\n'
  assert.equal(parseRegQueryValue(out, 'HERMES_HOME'), 'C:\\Program Files\\Hermes')
})

test('parseRegQueryValue returns null when the value line is absent', () => {
  const out = 'HKEY_CURRENT_USER\\Environment\r\n    Path    REG_SZ    C:\\x\r\n'
  assert.equal(parseRegQueryValue(out, 'HERMES_HOME'), null)
  assert.equal(parseRegQueryValue('', 'HERMES_HOME'), null)
  assert.equal(parseRegQueryValue('garbage', 'HERMES_HOME'), null)
})

// ── expandWindowsEnvRefs ───────────────────────────────────────────────────

test('expandWindowsEnvRefs expands %VAR% case-insensitively', () => {
  assert.equal(expandWindowsEnvRefs('%UserProfile%\\h', { USERPROFILE: 'C:\\Users\\jeff' }), 'C:\\Users\\jeff\\h')
})

test('expandWindowsEnvRefs leaves literal paths and unknown refs intact', () => {
  assert.equal(expandWindowsEnvRefs('F:\\Hermes\\data', {}), 'F:\\Hermes\\data')
  assert.equal(expandWindowsEnvRefs('%NOPE%\\x', {}), '%NOPE%\\x')
})

// ── readWindowsUserEnvVar ──────────────────────────────────────────────────

test('readWindowsUserEnvVar returns null off Windows without spawning', () => {
  let spawned = false

  const exec = () => {
    spawned = true

    return ''
  }

  assert.equal(readWindowsUserEnvVar('HERMES_HOME', { platform: 'linux', exec }), null)
  assert.equal(spawned, false)
})

test('readWindowsUserEnvVar queries HKCU\\Environment and expands the value', () => {
  const calls = []

  const exec = (cmd, args) => {
    calls.push([cmd, args])

    return 'HKEY_CURRENT_USER\\Environment\r\n    HERMES_HOME    REG_EXPAND_SZ    %DRIVE%\\Hermes\r\n'
  }

  const value = readWindowsUserEnvVar('HERMES_HOME', {
    platform: 'win32',
    env: { DRIVE: 'F:' },
    exec
  })

  assert.equal(value, 'F:\\Hermes')
  assert.deepEqual(calls, [['reg', ['query', 'HKCU\\Environment', '/v', 'HERMES_HOME']]])
})

test('readWindowsUserEnvVar returns null when reg exits non-zero (value missing)', () => {
  const exec = () => {
    throw new Error('reg exited 1')
  }

  assert.equal(readWindowsUserEnvVar('HERMES_HOME', { platform: 'win32', exec }), null)
})

test('readWindowsUserEnvVar returns null for an empty value', () => {
  const exec = () => '    HERMES_HOME    REG_SZ    \r\n'
  assert.equal(readWindowsUserEnvVar('HERMES_HOME', { platform: 'win32', exec }), null)
})

test('readWindowsHostPath combines live machine and user PATH in Windows order', () => {
  const calls = []

  const exec = (cmd, args) => {
    calls.push([cmd, args])

    if (args[1].startsWith('HKLM\\')) {
      return (
        'HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment\r\n' +
        '    Path    REG_EXPAND_SZ    %SystemRoot%\\system32;C:\\Host\\System\r\n'
      )
    }

    return 'HKEY_CURRENT_USER\\Environment\r\n' + '    Path    REG_SZ    C:\\Host\\User;C:\\Host\\GitHubCLI\r\n'
  }

  const path = readWindowsHostPath({
    platform: 'win32',
    env: { SystemRoot: 'C:\\Windows' },
    exec
  })

  assert.equal(path, 'C:\\Windows\\system32;C:\\Host\\System;C:\\Host\\User;C:\\Host\\GitHubCLI')
  assert.deepEqual(calls, [
    ['reg', ['query', 'HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment', '/v', 'Path']],
    ['reg', ['query', 'HKCU\\Environment', '/v', 'Path']]
  ])
})

test('readWindowsHostPath ignores a user PATH when the machine PATH is unavailable', () => {
  const exec = (_cmd, args) => {
    if (args[1].startsWith('HKLM\\')) {
      throw new Error('machine environment unavailable')
    }

    return 'HKEY_CURRENT_USER\\Environment\r\n    Path    REG_SZ    C:\\Host\\User\r\n'
  }

  assert.equal(readWindowsHostPath({ platform: 'win32', exec }), null)
})

test('readWindowsHostPath keeps the machine PATH when the user PATH is unavailable', () => {
  const exec = (_cmd, args) => {
    if (args[1].startsWith('HKCU\\')) {
      throw new Error('user environment unavailable')
    }

    return (
      'HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment\r\n' +
      '    Path    REG_SZ    C:\\Windows\\System32;C:\\Host\\System\r\n'
    )
  }

  assert.equal(readWindowsHostPath({ platform: 'win32', exec }), 'C:\\Windows\\System32;C:\\Host\\System')
})

test('readWindowsHostPath is a no-op off Windows', () => {
  let spawned = false
  const exec = () => {
    spawned = true
    return ''
  }

  assert.equal(readWindowsHostPath({ platform: 'linux', exec }), null)
  assert.equal(spawned, false)
})
