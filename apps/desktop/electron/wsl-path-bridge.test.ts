import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  isWslBridgeActive,
  parseDefaultDistro,
  resolveLocalReadPath,
  resolvePickerDefaultPath,
  setWslBridgeActive,
  wslPosixToWindowsAccessible
} from './wsl-path-bridge'

test('parseDefaultDistro reads the first distro from clean utf-8 output', () => {
  assert.equal(parseDefaultDistro('Ubuntu\nDebian\n'), 'Ubuntu')
})

test('parseDefaultDistro survives UTF-16LE NUL bytes older wsl.exe leaves in (WSL#4607)', () => {
  // `wsl.exe -l -q` emits UTF-16LE without a BOM on builds that ignore
  // WSL_UTF8; decoded as utf8 that reads as NUL-interleaved text.
  const utf16ish = '\0U\0b\0u\0n\0t\0u\0\r\0\n\0D\0e\0b\0i\0a\0n\0'
  assert.equal(parseDefaultDistro(utf16ish), 'Ubuntu')
})

test('parseDefaultDistro strips the default-marker and blank lines', () => {
  assert.equal(parseDefaultDistro('\n* Ubuntu\nDebian\n'), 'Ubuntu')
  assert.equal(parseDefaultDistro('   \n\n'), null)
})

test('wslPosixToWindowsAccessible maps a drvfs mount to its Windows drive', () => {
  assert.equal(wslPosixToWindowsAccessible('/mnt/c/Users/alex', 'Ubuntu'), 'C:\\Users\\alex')
  assert.equal(wslPosixToWindowsAccessible('/mnt/d', 'Ubuntu'), 'D:\\')
})

test('wslPosixToWindowsAccessible maps an in-distro POSIX path to a UNC share', () => {
  assert.equal(wslPosixToWindowsAccessible('/home/alex/proj', 'Ubuntu'), '\\\\wsl.localhost\\Ubuntu\\home\\alex\\proj')
})

test('wslPosixToWindowsAccessible leaves non-absolute / already-Windows paths alone', () => {
  assert.equal(wslPosixToWindowsAccessible('C:\\Users\\alex', 'Ubuntu'), 'C:\\Users\\alex')
  assert.equal(wslPosixToWindowsAccessible('relative/dir', 'Ubuntu'), 'relative/dir')
})

test('resolvePickerDefaultPath bridges a WSL cwd but passes Windows paths and empties through', () => {
  assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu'), '\\\\wsl.localhost\\Ubuntu\\home\\alex')
  assert.equal(resolvePickerDefaultPath('C:\\proj', 'Ubuntu'), 'C:\\proj')
  assert.equal(resolvePickerDefaultPath(undefined, 'Ubuntu'), undefined)
})

// ---------------------------------------------------------------------------
// Regression: remote-gateway mode must not spawn wsl.exe (#66433).
//
// The WSL bridge translates POSIX paths into Windows-accessible forms by
// probing `wsl.exe -l -q` for the default distro. In remote-gateway mode the
// POSIX paths belong to a host the Windows desktop cannot open locally, so the
// bridge must no-op — never probing wsl.exe (which pops the interactive
// "Install WSL" console prompt on WSL-less machines). main.ts toggles the
// bridge off via setWslBridgeActive(false) once it resolves a remote backend.
// ---------------------------------------------------------------------------

test('setWslBridgeActive / isWslBridgeActive round-trip and defaults to active', () => {
  assert.equal(isWslBridgeActive(), true)
  setWslBridgeActive(false)
  assert.equal(isWslBridgeActive(), false)
  setWslBridgeActive(true)
  assert.equal(isWslBridgeActive(), true)
})

test('resolveLocalReadPath returns POSIX paths unchanged when bridge is inactive', () => {
  setWslBridgeActive(false)
  try {
    // These POSIX paths WOULD be bridged on Windows; with the bridge off they
    // pass through verbatim on every platform — the early return skips distro
    // resolution entirely so wsl.exe is never probed.
    assert.equal(resolveLocalReadPath('/home/ubuntu/project'), '/home/ubuntu/project')
    assert.equal(resolveLocalReadPath('/mnt/c/Users'), '/mnt/c/Users')
    assert.equal(resolveLocalReadPath('C:\\Users'), 'C:\\Users')
    assert.equal(resolveLocalReadPath(''), '')
  } finally {
    setWslBridgeActive(true)
  }
})

test('resolvePickerDefaultPath returns POSIX paths unchanged when bridge is inactive', () => {
  setWslBridgeActive(false)
  try {
    assert.equal(resolvePickerDefaultPath('/home/ubuntu'), '/home/ubuntu')
    assert.equal(resolvePickerDefaultPath('C:\\proj'), 'C:\\proj')
    assert.equal(resolvePickerDefaultPath(undefined), undefined)
  } finally {
    setWslBridgeActive(true)
  }
})
