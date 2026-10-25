import assert from 'node:assert/strict'

import { test } from 'vitest'

import { expandWindowsEnvRefs, parseRegQueryValue, readWindowsUserEnvVar } from './windows-user-env'

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

// ── code page: the reg.exe stdout boundary ─────────────────────────────────

test('an ASCII reg value is decoded without spawning anything else', () => {
  const calls = []

  const exec = (cmd, args) => {
    calls.push(cmd)

    return Buffer.from(
      'HKEY_CURRENT_USER\\Environment\r\n    HERMES_HOME    REG_SZ    F:\\Hermes\r\n',
      'utf8'
    )
  }

  const value = readWindowsUserEnvVar('HERMES_HOME', { platform: 'win32', env: {}, exec })

  assert.equal(value, 'F:\\Hermes')
  assert.deepEqual(calls, ['reg'])
})

test('a non-ASCII reg value is re-read as base64 instead of guessed at', () => {
  // C:\<U+5341 U+80FD U+4E88>\hermes in CP932. Each of the three characters
  // has 0x5C as its trail byte, so a UTF-8 decode of these bytes yields three
  // extra path separators rather than three replacement characters.
  const cp932Value = Buffer.from([
    0x43, 0x3a, 0x5c, 0x8f, 0x5c, 0x94, 0x5c, 0x97, 0x5c, 0x5c,
    0x68, 0x65, 0x72, 0x6d, 0x65, 0x73
  ])
  const regOutput = Buffer.concat([
    Buffer.from('HKEY_CURRENT_USER\\Environment\r\n    HERMES_HOME    REG_SZ    ', 'utf8'),
    cp932Value,
    Buffer.from('\r\n', 'utf8')
  ])
  const expected = 'C:\\\u5341\u80fd\u4e88\\hermes'

  // What the old code returned, kept as an assertion so the failure mode is
  // pinned rather than described.
  const mangled = regOutput.toString('utf8').split('REG_SZ    ')[1].trim()

  assert.notEqual(mangled, expected)
  assert.equal((mangled.match(/\\/g) || []).length, 5)
  assert.equal((expected.match(/\\/g) || []).length, 2)

  const calls = []

  const exec = (cmd, args) => {
    calls.push(cmd)

    if (cmd === 'reg') {
      return regOutput
    }

    assert.equal(cmd, 'powershell')
    assert.ok(String(args[args.length - 1]).includes('ToBase64String'))
    assert.ok(String(args[args.length - 1]).includes("'HERMES_HOME','User'"))

    return Buffer.from(expected, 'utf8').toString('base64') + '\r\n'
  }

  const value = readWindowsUserEnvVar('HERMES_HOME', { platform: 'win32', env: {}, exec })

  assert.equal(value, expected)
  assert.deepEqual(calls, ['reg', 'powershell'])
})

test('a non-ASCII value returns null when the base64 re-read fails', () => {
  const exec = (cmd) => {
    if (cmd === 'reg') {
      return Buffer.from([
        0x20, 0x48, 0x45, 0x52, 0x4d, 0x45, 0x53, 0x5f, 0x48, 0x4f, 0x4d, 0x45,
        0x20, 0x52, 0x45, 0x47, 0x5f, 0x53, 0x5a, 0x20, 0x8f, 0x5c, 0x0d, 0x0a
      ])
    }

    throw new Error('powershell missing')
  }

  assert.equal(readWindowsUserEnvVar('HERMES_HOME', { platform: 'win32', env: {}, exec }), null)
})

test('a name that is not a plain identifier is never interpolated into PowerShell', () => {
  const calls = []

  const exec = (cmd, args) => {
    calls.push(cmd)

    if (cmd === 'reg') {
      return Buffer.from([0x20, 0x52, 0x45, 0x47, 0x5f, 0x53, 0x5a, 0x20, 0x8f, 0x5c, 0x0d, 0x0a])
    }

    return ''
  }

  assert.equal(
    readWindowsUserEnvVar("X'; Remove-Item C:\\ -Recurse; '", {
      platform: 'win32',
      env: {},
      exec
    }),
    null
  )
  assert.deepEqual(calls, ['reg'])
})
