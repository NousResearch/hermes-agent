/**
 * Tests for electron/dev-cdp.ts.
 *
 * Run with: npx vitest run --project electron electron/dev-cdp.test.ts
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { DEFAULT_PORT, describeDevCdpDecision, resolveDevCdpPort } from './dev-cdp'

const DEV_SERVER = 'http://127.0.0.1:5174'

/** The ordinary `npm run dev` / `hgui` run. */
const devRun = { env: {}, isPackaged: false, devServer: DEV_SERVER }

test('a dev-server run opens the default port with no opt-in', () => {
  assert.deepEqual(resolveDevCdpPort(devRun), { port: DEFAULT_PORT, reason: null })
})

test('the default matches what the scripts/ tooling reaches for', () => {
  // scripts/eval.mjs and scripts/perf/lib/cdp.mjs both default here; if this
  // drifts, `node scripts/eval.mjs ...` stops finding a live renderer.
  assert.equal(DEFAULT_PORT, 9222)
})

test('a packaged build never opens the port, however loudly the env asks', () => {
  const decision = resolveDevCdpPort({ ...devRun, env: { HERMES_DESKTOP_CDP_PORT: '9222' }, isPackaged: true })

  assert.deepEqual(decision, { port: null, reason: 'packaged' })
})

test('packaged is checked before every other gate', () => {
  // Belt-and-suspenders: dev server present, valid port requested, still shut.
  for (const value of ['9222', '', 'off', 'garbage']) {
    const decision = resolveDevCdpPort({
      env: { HERMES_DESKTOP_CDP_PORT: value },
      isPackaged: true,
      devServer: DEV_SERVER
    })

    assert.equal(decision.port, null, `expected packaged to refuse ${JSON.stringify(value)}`)
    assert.equal(decision.reason, 'packaged')
  }
})

test('an unpackaged dist run (no dev server) does not qualify', () => {
  // `electron .` against dist/ is how the packaged app gets smoke tested; it
  // should behave like the packaged app, not like a source-tree dev run.
  assert.deepEqual(resolveDevCdpPort({ ...devRun, devServer: undefined }), { port: null, reason: 'no-dev-server' })
})

test('the port is overridable', () => {
  assert.equal(resolveDevCdpPort({ ...devRun, env: { HERMES_DESKTOP_CDP_PORT: '9333' } }).port, 9333)
})

test('tolerates surrounding whitespace on the override', () => {
  assert.equal(resolveDevCdpPort({ ...devRun, env: { HERMES_DESKTOP_CDP_PORT: ' 9333 ' } }).port, 9333)
})

test('can be switched off on a dev run', () => {
  for (const value of ['0', 'off', 'OFF', 'false', 'no']) {
    const decision = resolveDevCdpPort({ ...devRun, env: { HERMES_DESKTOP_CDP_PORT: value } })

    assert.equal(decision.port, null, `expected ${JSON.stringify(value)} to close the port`)
    assert.equal(decision.reason, 'opted-out')
  }
})

test('refuses ports that are not usable integers', () => {
  for (const value of ['80', '-1', '70000', 'yes', '9222.5', '92 22']) {
    const decision = resolveDevCdpPort({ ...devRun, env: { HERMES_DESKTOP_CDP_PORT: value } })

    assert.equal(decision.port, null, `expected ${JSON.stringify(value)} to be refused`)
    assert.equal(decision.reason, 'invalid-port')
  }
})

test('explains itself when an explicit setting was not honoured', () => {
  // A typo'd port or a deliberate opt-out should say so — silently doing
  // something other than what the env asked for is the bad failure mode.
  for (const value of ['garbage', 'off']) {
    const decision = resolveDevCdpPort({ ...devRun, env: { HERMES_DESKTOP_CDP_PORT: value } })

    assert.ok(describeDevCdpDecision(decision), `expected an explanation for ${JSON.stringify(value)}`)
  }
})

test('stays quiet when the port opened, or is closed by design', () => {
  assert.equal(describeDevCdpDecision(resolveDevCdpPort(devRun)), null)
  assert.equal(describeDevCdpDecision(resolveDevCdpPort({ ...devRun, isPackaged: true })), null)
  assert.equal(describeDevCdpDecision(resolveDevCdpPort({ ...devRun, devServer: undefined })), null)
})
