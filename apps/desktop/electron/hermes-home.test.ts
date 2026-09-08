/**
 * Tests for electron/hermes-home.ts.
 *
 * Run with: npx vitest run --project electron electron/hermes-home.test.ts
 */

import assert from 'node:assert/strict'

import path from 'node:path'

import { test } from 'vitest'

import { resolveHermesHomeFromInputs } from './hermes-home'

/** Injected-deps base: non-Windows, nothing set, no fs hits. */
const base = {
  env: {} as Record<string, string | undefined>,
  isWindows: false,
  appHome: '/Users/tester',
  userDataOverride: undefined as string | undefined,
  readWindowsUserEnvVar: (_name: string) => undefined as string | undefined,
  directoryExists: (_p: string) => false
}

/** Windows variant of base: win paths + win32 path module. */
const winBase = {
  ...base,
  isWindows: true,
  appHome: 'C:\\Users\\tester',
  pathModule: path.win32,
  env: { LOCALAPPDATA: 'C:\\Users\\tester\\AppData\\Local' } as Record<string, string | undefined>
}

test('HERMES_HOME wins over everything (normalized)', () => {
  const h = resolveHermesHomeFromInputs({ ...base, env: { HERMES_HOME: '/tmp/hh' } })
  assert.equal(h, '/tmp/hh')
})

test('HERMES_DESKTOP_USER_DATA_DIR-only launch maps to <userData>/hermes-home', () => {
  // The review's deterministic case 1: a fresh/sandbox launch with ONLY the
  // userData override must land on <userData>/hermes-home, NOT ~/.hermes.
  const h = resolveHermesHomeFromInputs({ ...base, userDataOverride: '/tmp/probe-userdata' })
  assert.equal(h, '/tmp/probe-userdata/hermes-home')
})

test('non-Windows fallback is ~/.hermes', () => {
  assert.equal(resolveHermesHomeFromInputs(base), '/Users/tester/.hermes')
})

test('HERMES_HOME beats the userData override', () => {
  const h = resolveHermesHomeFromInputs({
    ...base,
    env: { HERMES_HOME: '/tmp/explicit' },
    userDataOverride: '/tmp/probe-userdata'
  })
  assert.equal(h, '/tmp/explicit')
})

test('windows: user-scoped registry HERMES_HOME beats LOCALAPPDATA', () => {
  const h = resolveHermesHomeFromInputs({
    ...winBase,
    readWindowsUserEnvVar: () => 'C:\\hermes-registry'
  })
  assert.equal(h, 'C:\\hermes-registry')
})

test('windows: legacy ~/.hermes kept when no LOCALAPPDATA install exists but legacy does', () => {
  const h = resolveHermesHomeFromInputs({
    ...winBase,
    directoryExists: (p) => p === 'C:\\Users\\tester\\.hermes'
  })
  assert.equal(h, 'C:\\Users\\tester\\.hermes')
})

test('windows: LOCALAPPDATA\\hermes when the install exists (or nothing does)', () => {
  const h = resolveHermesHomeFromInputs(winBase)
  assert.equal(h, 'C:\\Users\\tester\\AppData\\Local\\hermes')
})
