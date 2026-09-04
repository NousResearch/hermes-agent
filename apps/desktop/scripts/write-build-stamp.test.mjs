import assert from 'node:assert/strict'
import { test } from 'vitest'

import {
  FALLBACK_BRANCH,
  FALLBACK_COMMIT,
  fromCI,
  fromFallback,
  fromLocalGit,
  isFallbackCommit,
  resolveStamp
} from './write-build-stamp.mjs'

test('fromCI reads GITHUB_SHA / GITHUB_REF_NAME', () => {
  assert.deepEqual(
    fromCI({ GITHUB_SHA: 'a'.repeat(40), GITHUB_REF_NAME: 'release' }),
    { commit: 'a'.repeat(40), branch: 'release', dirty: false, source: 'ci' }
  )
  assert.equal(fromCI({}), null)
})

test('fromLocalGit returns null when git rev-parse fails', () => {
  const stamp = fromLocalGit('/tmp/not-a-repo', () => null)
  assert.equal(stamp, null)
})

test('fromLocalGit reads HEAD + branch + dirty status', () => {
  const calls = []
  const execFn = (cmd) => {
    calls.push(cmd)
    if (cmd === 'git rev-parse HEAD') return 'b'.repeat(40)
    if (cmd === 'git rev-parse --abbrev-ref HEAD') return 'main'
    if (cmd === 'git status --porcelain -uno') return ' M apps/desktop/package.json'
    return null
  }
  assert.deepEqual(fromLocalGit('/repo', execFn), {
    commit: 'b'.repeat(40),
    branch: 'main',
    dirty: true,
    source: 'local'
  })
  assert.ok(calls.includes('git rev-parse HEAD'))
})

test('fromFallback uses the all-zero placeholder commit', () => {
  assert.deepEqual(fromFallback(), {
    commit: FALLBACK_COMMIT,
    branch: FALLBACK_BRANCH,
    dirty: false,
    source: 'fallback'
  })
  assert.equal(isFallbackCommit(FALLBACK_COMMIT), true)
  assert.equal(isFallbackCommit('a'.repeat(40)), false)
})

test('resolveStamp prefers CI over local git over fallback', () => {
  const ci = resolveStamp({
    env: { GITHUB_SHA: 'c'.repeat(40), GITHUB_REF_NAME: 'main' },
    execFn: () => 'should-not-run'
  })
  assert.equal(ci.source, 'ci')
  assert.equal(ci.commit, 'c'.repeat(40))

  const local = resolveStamp({
    env: {},
    execFn: (cmd) => {
      if (cmd === 'git rev-parse HEAD') return 'd'.repeat(40)
      if (cmd === 'git rev-parse --abbrev-ref HEAD') return 'main'
      if (cmd === 'git status --porcelain -uno') return ''
      return null
    }
  })
  assert.equal(local.source, 'local')
  assert.equal(local.commit, 'd'.repeat(40))
  assert.equal(local.dirty, false)
})

test('resolveStamp falls back when neither CI nor git is available', () => {
  const stamp = resolveStamp({ env: {}, execFn: () => null })
  assert.deepEqual(stamp, {
    commit: FALLBACK_COMMIT,
    branch: FALLBACK_BRANCH,
    dirty: false,
    source: 'fallback'
  })
})

// ── buildStampPayload — the staged-desktop full schema ─────────────────────

import { buildStampPayload } from './write-build-stamp.mjs'

const baseStamp = {
  commit: 'c'.repeat(40),
  branch: 'main',
  builtAt: '2026-08-14T00:00:00Z',
  dirty: false,
  source: 'ci'
}

test('buildStampPayload without a variant keeps the legacy 5-field shape', () => {
  const payload = buildStampPayload(baseStamp, {})
  assert.deepEqual(payload, {
    schemaVersion: 1,
    commit: baseStamp.commit,
    branch: 'main',
    builtAt: payload.builtAt, // stamped at write time, not the fixture's
    dirty: false,
    source: 'ci'
  })
  assert.equal(typeof payload.builtAt, 'string')
  assert.ok(!('payload' in payload))
  assert.ok(!('store' in payload))
  assert.ok(!('tag' in payload))
})

test('buildStampPayload with bundled variant stamps payload bundled, store false', () => {
  const payload = buildStampPayload(baseStamp, {
    HERMES_DESKTOP_VARIANT: 'bundled',
    HERMES_PAYLOAD_TAG: 'v0.27.1-canary.20260901072553'
  })
  assert.equal(payload.payload, 'bundled')
  assert.equal(payload.store, false)
  assert.equal(payload.distribution, 'desktop-app')
  assert.equal(payload.updateMechanism, 'external')
  assert.equal(payload.tag, 'v0.27.1-canary.20260901072553')
})

test('buildStampPayload with store variant stamps payload bundled, store true', () => {
  const payload = buildStampPayload(baseStamp, {
    HERMES_DESKTOP_VARIANT: 'store',
    HERMES_PAYLOAD_TAG: 'v0.27.1'
  })
  assert.equal(payload.payload, 'bundled')
  assert.equal(payload.store, true)
})

test('buildStampPayload with light variant stamps payload light', () => {
  const payload = buildStampPayload(baseStamp, {
    HERMES_DESKTOP_VARIANT: 'light',
    HERMES_PAYLOAD_TAG: 'v0.27.1'
  })
  assert.equal(payload.payload, 'light')
  assert.equal(payload.store, false)
})

test('buildStampPayload keeps schemaVersion + provenance in the staged shape', () => {
  const payload = buildStampPayload(baseStamp, {
    HERMES_DESKTOP_VARIANT: 'bundled'
  })
  assert.equal(payload.schemaVersion, 1)
  assert.equal(payload.commit, baseStamp.commit)
  assert.equal(payload.source, 'ci')
  assert.equal(payload.tag, null)
})

