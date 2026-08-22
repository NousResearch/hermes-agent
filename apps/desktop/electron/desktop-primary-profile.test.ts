import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { test } from 'vitest'

import {
  isValidProfileName,
  resolveDesktopPrimaryProfile
} from './desktop-primary-profile'

const MAIN_TS_PATH = resolve(__dirname, 'main.ts')
const mainSource = readFileSync(MAIN_TS_PATH, 'utf8')

test('returns the desktop-stored profile when set', () => {
  assert.equal(resolveDesktopPrimaryProfile('security-analyst', 'software-engineer'), 'security-analyst')
})

test('returns the desktop-stored profile even when it equals the CLI sticky', () => {
  // The desktop pref is the user telling us "always use this", so a stored
  // value of 'default' still pins explicitly rather than falling through.
  assert.equal(resolveDesktopPrimaryProfile('default', 'software-engineer'), 'default')
})

test('falls back to the CLI sticky profile when the desktop pref is unset', () => {
  // The bug we're fixing: desktop used to fall straight to 'default' here.
  assert.equal(resolveDesktopPrimaryProfile(null, 'software-engineer'), 'software-engineer')
})

test('falls back to the CLI sticky profile when the desktop pref is empty string', () => {
  assert.equal(resolveDesktopPrimaryProfile('', 'security-analyst'), 'security-analyst')
})

test('falls back to "default" when neither preference is set', () => {
  assert.equal(resolveDesktopPrimaryProfile(null, null), 'default')
  assert.equal(resolveDesktopPrimaryProfile('', ''), 'default')
  assert.equal(resolveDesktopPrimaryProfile(null, ''), 'default')
})

test('trims whitespace around the CLI sticky file value', () => {
  // The CLI writes "<name>\n" — defensive trim keeps a stray newline from
  // slipping through to the backend as "--profile software-engineer\n".
  assert.equal(resolveDesktopPrimaryProfile(null, '  software-engineer  '), 'software-engineer')
})

test('accepts the canonical profile names', () => {
  assert.equal(isValidProfileName('default'), true)
  assert.equal(isValidProfileName('software-engineer'), true)
  assert.equal(isValidProfileName('security-analyst'), true)
  assert.equal(isValidProfileName('work-vps-1'), true)
})

test('rejects invalid profile names', () => {
  assert.equal(isValidProfileName(''), false)
  assert.equal(isValidProfileName('   '), false)
  assert.equal(isValidProfileName('-leading-dash'), false)
  assert.equal(isValidProfileName('Has Spaces'), false)
  assert.equal(isValidProfileName('A'.repeat(65)), false)
})

// ---------------------------------------------------------------------------
// Wiring invariants — `startHermes()` and `hermes:profile:get` must both go
// through `primaryProfileKey()` (which already falls back through the CLI
// sticky), not read the desktop-only preference directly. Without this the
// backend launches into one profile while the renderer's profile switcher
// displays another — same root cause, different symptom (#57757).
// ---------------------------------------------------------------------------

test('startHermes resolves the --profile flag via primaryProfileKey', () => {
  // Find the startHermes body by anchoring on the --port 0 backendArgs.
  const anchor = "const backendArgs = ['serve', '--host', '127.0.0.1', '--port', '0']"
  const start = mainSource.indexOf(anchor)
  assert.notEqual(start, -1, 'backendArgs anchor not found in main.ts')

  const end = mainSource.indexOf("const setup = await runPrimaryBackendStartup({", start)
  assert.notEqual(end, -1, 'runPrimaryBackendStartup call not found after anchor')
  const body = mainSource.slice(start, end)

  assert.match(
    body,
    /const activeProfile = primaryProfileKey\(\)/,
    'startHermes should resolve the --profile flag through primaryProfileKey(), not readActiveDesktopProfile()'
  )
  assert.match(
    body,
    /if \(activeProfile !== 'default'\)/,
    'startHermes should skip --profile when the resolved profile is "default" (legacy behavior)'
  )
})

test('hermes:profile:get IPC handler reports primaryProfileKey, not just the desktop pref', () => {
  assert.match(
    mainSource,
    /ipcMain\.handle\('hermes:profile:get', async \(\) => \(\{ profile: primaryProfileKey\(\) \}\)\)/,
    'hermes:profile:get must report primaryProfileKey() so the renderer-side switcher agrees with the launched backend'
  )
})
