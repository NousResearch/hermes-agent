import assert from 'node:assert/strict'
import { test } from 'vitest'

import { spawnSync } from 'node:child_process'
import path from 'node:path'

import { nativeQuad, releaseVersion } from '../../../scripts/msix-shared.mjs'

// 2026-09-22T00:14:03Z, worked by hand: day-of-year 264 (2026 is not a leap
// year; Jan 1 is hour 0), so hour-of-year is 264*24 = 6336. The 14th minute
// holds 14*60+3 = 843 seconds.
const EPOCH = Date.parse('2026-09-22T00:14:03Z') / 1000

test('stable keeps the store quad while canary uses yy.mmdd.hh.mmss', () => {
  assert.equal(nativeQuad('v0.21.5', EPOCH), '2026.6336.843.0')
  assert.equal(nativeQuad('canary.20260922T001403Z-v0.21.4', EPOCH), '26.922.0.1403')
})

test('a commit build claim is versioned from its stamp exactly like a canary', () => {
  assert.equal(nativeQuad('commit.20260922T001403Z-v0.21.4', EPOCH), '26.922.0.1403')
  assert.equal(releaseVersion('commit.20260922T001403Z-v0.21.4'), '0.21.4+commit.20260922T001403Z')
  assert.equal(releaseVersion('v0.21.4+commit.20260922T001403Z'), null)
})

test('a commit claim packages as its commit identity and never as a release', () => {
  const root = path.resolve(import.meta.dirname, '../../..')
  const probe = `import{appIdentity}from'./scripts/msix-shared.mjs';const v=appIdentity(process.cwd()+'/apps/desktop');console.log(JSON.stringify({version:v.version,fileVersion:v.fileVersion,app:v.identity.msixAppIdWithOrg}))`
  const base = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith('HERMES_')))
  const run = env => spawnSync(process.execPath, ['--input-type=module', '-e', probe],
    { cwd: root, encoding: 'utf8', env: { ...base, HERMES_DESKTOP_VARIANT: 'bundled', ...env }, timeout: 30000 })
  const claim = { HERMES_PAYLOAD_TAG: 'commit.20260922T001403Z-v0.21.4', HERMES_BUILD_COMMIT: 'a'.repeat(40) }
  const result = run(claim)
  assert.equal(result.status, 0, result.stderr)
  const identity = JSON.parse(result.stdout)
  assert.equal(identity.version, '26.922.0.1403')
  assert.equal(identity.fileVersion, '0.21.4+commit.20260922T001403Z')
  assert.equal(identity.app, 'NousResearch.HermesBundledCommitaaaaaaa')
  // Without the commit identity a commit claim would build an updatable release: refused.
  assert.notEqual(run({ HERMES_PAYLOAD_TAG: claim.HERMES_PAYLOAD_TAG }).status, 0)
  // Nor does a commit identity take any other release tag.
  assert.notEqual(run({ ...claim, HERMES_PAYLOAD_TAG: 'canary.20260922T001403Z-v0.21.4' }).status, 0)
})

test('historical release identities are rejected', () => {
  // The retired suffix shape and its dash spelling are no longer release identities.
  assert.throws(() => nativeQuad('v0.21.4+canary.20260922T001403Z', EPOCH))
  assert.throws(() => nativeQuad('v0.21.4-canary.20260922T001403Z', EPOCH))
  assert.throws(() => nativeQuad('v2026.9.21', EPOCH))
  assert.throws(() => nativeQuad('canary.20260922T001403Z-v2026.9.21', EPOCH))
})

test('a later canary sorts above an earlier canary across a month boundary', () => {
  const earlier = nativeQuad(
    'canary.20260131T235959Z-v0.21.4',
    Date.parse('2026-01-31T23:59:59Z') / 1000
  ).split('.').map(Number)
  const later = nativeQuad(
    'canary.20260201T000000Z-v0.21.4',
    Date.parse('2026-02-01T00:00:00Z') / 1000
  ).split('.').map(Number)
  const first = later.findIndex((value, index) => value !== earlier[index])
  assert.ok(first >= 0)
  assert.ok(later[first] > earlier[first])
})

test('every field stays inside 16 bits', () => {
  for (const stamp of ['2026-01-01T00:00:00Z', '2026-12-31T23:59:59Z', '2028-12-31T23:59:59Z']) {
    for (const ref of ['v0.21.5', `canary.${stamp.replace(/[-:]/g, '').replace('.000', '')}-v0.21.4`]) {
      const parts = nativeQuad(ref, Date.parse(stamp) / 1000).split('.').map(Number)
      assert.equal(parts.length, 4)
      assert.ok(parts.every(value => value >= 0 && value <= 65535))
    }
  }
})
