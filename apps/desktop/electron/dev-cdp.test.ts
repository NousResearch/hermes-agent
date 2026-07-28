/**
 * Tests for electron/dev-cdp.ts.
 *
 * Run with: npx vitest run --project electron electron/dev-cdp.test.ts
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { describeDevCdpDecision, resolveDevCdpPort } from './dev-cdp'

const DEV_SERVER = 'http://127.0.0.1:5174'

/** A dev-server run that asked for a port — the one combination that opens it. */
const opted = { env: { HERMES_DESKTOP_CDP_PORT: '9222' }, isPackaged: false, devServer: DEV_SERVER }

test('opens the requested port for an opted-in dev-server run', () => {
  assert.deepEqual(resolveDevCdpPort(opted), { port: 9222, reason: null })
})

test('stays closed unless the developer asks for it', () => {
  // The default `npm run dev` / `hgui` path: dev server, no opt-in.
  assert.deepEqual(resolveDevCdpPort({ ...opted, env: {} }), { port: null, reason: 'not-requested' })
})

test('a packaged build never opens the port, however loudly the env asks', () => {
  assert.deepEqual(resolveDevCdpPort({ ...opted, isPackaged: true }), { port: null, reason: 'packaged' })
})

test('packaged wins over every other gate', () => {
  // Belt-and-suspenders: even with a dev server present and a valid port
  // requested, packaged is checked first and short-circuits.
  const decision = resolveDevCdpPort({
    env: { HERMES_DESKTOP_CDP_PORT: '9222' },
    isPackaged: true,
    devServer: DEV_SERVER
  })

  assert.equal(decision.port, null)
  assert.equal(decision.reason, 'packaged')
})

test('an unpackaged dist run (no dev server) does not qualify', () => {
  // `electron .` against dist/ is how the packaged app gets smoke tested; it
  // should behave like the packaged app, not like a source-tree dev run.
  assert.deepEqual(resolveDevCdpPort({ ...opted, devServer: undefined }), { port: null, reason: 'no-dev-server' })
})

test('rejects ports that are not usable integers', () => {
  for (const value of ['0', '80', '-1', '70000', 'yes', '9222.5', '92 22', '']) {
    const decision = resolveDevCdpPort({ ...opted, env: { HERMES_DESKTOP_CDP_PORT: value } })

    assert.equal(decision.port, null, `expected ${JSON.stringify(value)} to be refused`)
  }
})

test('tolerates surrounding whitespace on the requested port', () => {
  assert.equal(resolveDevCdpPort({ ...opted, env: { HERMES_DESKTOP_CDP_PORT: ' 9333 ' } }).port, 9333)
})

test('every refusal that followed an explicit request explains itself', () => {
  // An opt-in that gets ignored must say why — a silent no-op is the failure
  // mode where someone burns an hour wondering why nothing is listening.
  const refusals = [
    resolveDevCdpPort({ ...opted, isPackaged: true }),
    resolveDevCdpPort({ ...opted, devServer: undefined }),
    resolveDevCdpPort({ ...opted, env: { HERMES_DESKTOP_CDP_PORT: 'nope' } })
  ]

  for (const decision of refusals) {
    assert.equal(decision.port, null)
    assert.ok(describeDevCdpDecision(decision), `expected an explanation for ${decision.reason}`)
  }
})

test('says nothing when the port opened, or when it was never requested', () => {
  assert.equal(describeDevCdpDecision(resolveDevCdpPort(opted)), null)
  assert.equal(describeDevCdpDecision(resolveDevCdpPort({ ...opted, env: {} })), null)
})
