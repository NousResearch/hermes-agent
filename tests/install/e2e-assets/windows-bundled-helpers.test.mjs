// windows-bundled-helpers — unit tests for the pure helpers of the
// Windows packaged-app E2E arm. node:test, no production imports needed:
// the publisher rule takes the production constant as injected data (the
// driver binds it to OUT_OF_STORE_PUBLISHER at runtime, from the same
// module it builds the feed descriptors with).
//
//   node --test tests/install/e2e-assets/windows-bundled-helpers.test.mjs
import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { validateBundledManifest, fourPartNewer, feedLayout, descriptorArgs } from './windows-bundled-helpers.mjs'

const PUB = 'CN=Nous Research Inc., O=Nous Research Inc., L=Austin, S=Texas, C=US'
const OTHER_PUB = 'CN=Someone Else'

function side(over = {}) {
  return {
    tag: 'v0.3.0',
    version: '0.3.0.0',
    commit: 'a'.repeat(40),
    identity: 'NousResearch.HermesBundled',
    publisher: PUB,
    applicationId: 'Hermes',
    artifact: {
      url: 'https://example.invalid/HermesBundled-0.3.0.0-win.msixbundle',
      sha256: 'b'.repeat(64),
      path: 'C:/staging/old/HermesBundled-0.3.0.0-win.msixbundle'
    },
    ...over
  }
}

function manifest(over = {}) {
  return { schema: 1, platform: 'windows', arch: 'x64', old: side(), new: side({ tag: 'v0.4.0', version: '0.4.0.0', commit: 'c'.repeat(40), artifact: { url: 'https://example.invalid/HermesBundled-0.4.0.0-win.msixbundle', sha256: 'd'.repeat(64), path: 'C:/staging/new/HermesBundled-0.4.0.0-win.msixbundle' } }), ...over }
}

function validate(m, extra = {}) {
  return validateBundledManifest(m, { expectedPublisher: PUB, artifactPathsExist: false, ...extra })
}

test('a well-formed schema1 windows manifest validates', () => {
  const r = validate(manifest())
  assert.deepEqual(r, { ok: true, errors: [] })
})

test('rejects wrong schema, platform, and arch', () => {
  assert.ok(!validate(manifest({ schema: 2 })).ok)
  assert.ok(!validate(manifest({ platform: 'macos' })).ok)
  assert.ok(!validate(manifest({ arch: 'arm64' })).ok)
  assert.ok(validate(manifest({ arch: 'arm64' }), { arch: 'arm64' }).ok)
})

test('rejects a publisher that is not the production OUT_OF_STORE_PUBLISHER', () => {
  const m = manifest()
  m.old.publisher = OTHER_PUB
  const r = validate(m)
  assert.ok(!r.ok)
  assert.ok(r.errors.some(e => e.includes('OUT_OF_STORE_PUBLISHER')))
})

test('rejects side/identity/applicationId disagreement', () => {
  const m = manifest()
  m.new.identity = 'NousResearch.Other'
  assert.ok(!validate(m).ok)
  const m2 = manifest()
  m2.new.applicationId = 'Other'
  assert.ok(!validate(m2).ok)
  const m3 = manifest()
  m3.new.publisher = OTHER_PUB
  const r3 = validate(m3)
  assert.ok(r3.errors.some(e => e.includes('old.publisher and new.publisher disagree')))
})

test('rejects non-monotonic versions and 3-part versions', () => {
  const m = manifest()
  m.new.version = '0.3.0.0'
  assert.ok(validate(m).errors.some(e => e.includes('strictly newer')))
  const m2 = manifest()
  m2.old.version = '0.3.0'
  assert.ok(validate(m2).errors.some(e => e.includes('4-part')))
})

test('rejects missing artifact fields and a fabricated artifact (no path)', () => {
  const m = manifest()
  delete m.new.artifact.path
  assert.ok(validate(m).errors.some(e => e.includes('artifact.path')))
  const m2 = manifest()
  m2.old.artifact.sha256 = 'zz'
  assert.ok(validate(m2).errors.some(e => e.includes('sha256')))
  const m3 = manifest()
  m3.new.artifact.url = ''
  assert.ok(validate(m3).errors.some(e => e.includes('artifact.url')))
})

test('rejects old == new commit', () => {
  const m = manifest()
  m.new.commit = m.old.commit
  assert.ok(validate(m).errors.some(e => e.includes('no update to prove')))
})

test('artifactPathsExist=true requires the downloaded file on disk', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wbh-'))
  const p = path.join(dir, 'bundle.msixbundle')
  fs.writeFileSync(p, 'x')
  const m = manifest()
  m.old.artifact.path = p
  m.new.artifact.path = path.join(dir, 'new.msixbundle')
  fs.writeFileSync(m.new.artifact.path, 'x')
  assert.ok(validate(m, { artifactPathsExist: true }).ok)
  m.old.artifact.path = path.join(dir, 'missing.msixbundle')
  assert.ok(!validate(m, { artifactPathsExist: true }).ok)
})

test('fourPartNewer: numeric, component-wise, strict', () => {
  assert.equal(fourPartNewer('0.4.0.0', '0.3.9.9'), true)
  assert.equal(fourPartNewer('1.2.3.10', '1.2.3.9'), true)
  assert.equal(fourPartNewer('0.3.0.0', '0.3.0.0'), false)
  assert.equal(fourPartNewer('0.2.9.9', '0.3.0.0'), false)
  assert.equal(fourPartNewer('0.3.0', '0.2.0.0'), false)
})

test('feedLayout stages per-side bundles and one swapped descriptor, rejecting collisions', () => {
  const l = feedLayout('feed', manifest())
  assert.ok(l.oldBundlePath.includes(`${path.sep}old${path.sep}`))
  assert.ok(l.newBundlePath.includes(`${path.sep}new${path.sep}`))
  assert.ok(l.descriptorPath.endsWith(`update.appinstaller`) || l.descriptorPath.endsWith('update.appinstaller'))
  const m = manifest()
  m.new.artifact.path = m.old.artifact.path
  assert.throws(() => feedLayout('feed', m), /collide/)
})

test('descriptorArgs pass the contract shape (empty channel path, optional descriptor filename)', () => {
  assert.deepEqual(descriptorArgs({ baseUrl: 'http://127.0.0.1:9/old', identityName: 'I', version: '0.3.0.0', bundleFilename: 'b.msixbundle' }), {
    baseUrl: 'http://127.0.0.1:9/old', variantChannelPath: '', identityName: 'I', version: '0.3.0.0', bundleFilename: 'b.msixbundle'
  })
  const withD = descriptorArgs({ baseUrl: 'u', identityName: 'I', version: 'v', bundleFilename: 'b', descriptorFilename: 'update.appinstaller' })
  assert.equal(withD.descriptorFilename, 'update.appinstaller')
})
