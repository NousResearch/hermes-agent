import assert from 'node:assert/strict'

import { afterEach, test } from 'vitest'

import {
  isWslBridgeActive,
  parseDefaultDistro,
  resolveLocalReadPath,
  resolvePickerDefaultPath,
  setWslBridgeActive,
  wslPosixToWindowsAccessible
} from './wsl-path-bridge'

// ── helpers ──────────────────────────────────────────────────────────

/** Reset the bridge to its default active state after every test so no test
 *  leaks global state into the next one. */
afterEach(() => {
  setWslBridgeActive(true)
})

// ── distro parsing (unchanged) ───────────────────────────────────────

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

// ── wslPosixToWindowsAccessible ──────────────────────────────────────

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

// ── resolvePickerDefaultPath (bridge active) ─────────────────────────

test('resolvePickerDefaultPath bridges a WSL cwd but passes Windows paths and empties through', () => {
  assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu'), '\\\\wsl.localhost\\Ubuntu\\home\\alex')
  assert.equal(resolvePickerDefaultPath('C:\\proj', 'Ubuntu'), 'C:\\proj')
  assert.equal(resolvePickerDefaultPath(undefined, 'Ubuntu'), undefined)
})

// ── bridge active / inactive ─────────────────────────────────────────

test('bridge defaults to active', () => {
  assert.equal(isWslBridgeActive(), true)
})

test('setWslBridgeActive(false) → resolvePickerDefaultPath passes raw path through without bridging', () => {
  setWslBridgeActive(false)
  // Even a clear WSL POSIX path must pass through unchanged when the bridge
  // is inactive — no distro probe, no wsl.exe, no install prompt.
  assert.equal(resolvePickerDefaultPath('/home/alex'), '/home/alex')
  assert.equal(resolvePickerDefaultPath('/mnt/c/Users/alex'), '/mnt/c/Users/alex')
  // Windows paths and empties are unaffected either way.
  assert.equal(resolvePickerDefaultPath('C:\\proj'), 'C:\\proj')
  assert.equal(resolvePickerDefaultPath(undefined), undefined)
})

test('setWslBridgeActive(false) → resolveLocalReadPath passes raw path through without bridging', () => {
  setWslBridgeActive(false)
  // resolveLocalReadPath is used by fs-read-dir to make WSL paths readable
  // on the Windows host. When the bridge is inactive (remote gateway), the
  // raw POSIX path must be returned as-is — no UNC rewriting, no distro
  // resolution. The downstream fs call will fail gracefully on non-WSL
  // hosts, which is the desired behaviour.
  assert.equal(resolveLocalReadPath('/home/alex/proj'), '/home/alex/proj')
  assert.equal(resolveLocalReadPath('/mnt/c/Users/alex'), '/mnt/c/Users/alex')
  // Non-POSIX paths are never bridged regardless of state.
  assert.equal(resolveLocalReadPath('C:\\Users\\alex'), 'C:\\Users\\alex')
  assert.equal(resolveLocalReadPath(''), '')
})

test('setWslBridgeActive(true) restores picker bridging', () => {
  setWslBridgeActive(false)
  assert.equal(resolvePickerDefaultPath('/home/alex'), '/home/alex')

  setWslBridgeActive(true)
  assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu'), '\\\\wsl.localhost\\Ubuntu\\home\\alex')
})

test('toggling the bridge is idempotent and does not corrupt cached state', () => {
  // Toggle twice each way.
  setWslBridgeActive(false)
  assert.equal(isWslBridgeActive(), false)
  setWslBridgeActive(false)
  assert.equal(isWslBridgeActive(), false)

  setWslBridgeActive(true)
  assert.equal(isWslBridgeActive(), true)
  setWslBridgeActive(true)
  assert.equal(isWslBridgeActive(), true)

  // Bridging still works after the toggles.
  assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu'), '\\\\wsl.localhost\\Ubuntu\\home\\alex')
})

// ── state isolation: every test sees a clean active bridge ────────────

test('state isolation: bridge is active after a previous test toggled it off', () => {
  // This test relies on afterEach resetting the bridge.
  // If isolation is broken, isWslBridgeActive() would be false here.
  assert.equal(isWslBridgeActive(), true)
})
