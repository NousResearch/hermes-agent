import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isPlaceholderAppVersion, launchBuildIdentity, launchMarkerNeedsReprobe } from './launch-build-identity'

const stamp = (
  overrides: Record<string, unknown> = {}
): { commit: string | null; builtAt: string | null; dirty: boolean } => ({
  commit: '357f51c491063f8a1b2c3d4e5f60718293a4b5c6',
  builtAt: '2026-09-28T10:11:12Z',
  dirty: false,
  ...overrides
})

test('launchBuildIdentity keeps a real release version as the whole identity', () => {
  assert.equal(launchBuildIdentity({ appVersion: '0.21.5' }), '0.21.5')
  assert.equal(launchBuildIdentity({ appVersion: '0.21.5', installStamp: stamp() }), '0.21.5')
  assert.equal(launchBuildIdentity({ appVersion: '1.2.3', installStamp: null }), '1.2.3')
})

test('launchBuildIdentity falls back to the install stamp on a source install', () => {
  // A source/`hermes update` install reports 0.0.0 on every build, so the
  // stamp's commit is the only thing that differs between two of them.
  assert.equal(
    launchBuildIdentity({ appVersion: '0.0.0', installStamp: stamp() }),
    '0.0.0+g357f51c49106@2026-09-28T10:11:12Z'
  )

  assert.equal(
    launchBuildIdentity({ appVersion: '0.0.0', installStamp: stamp({ dirty: true }) }),
    '0.0.0+g357f51c49106-dirty@2026-09-28T10:11:12Z'
  )

  // No stamp at all: a dev run, nothing better to compare on.
  assert.equal(launchBuildIdentity({ appVersion: '0.0.0', installStamp: null }), '0.0.0')

  // A commit without a build time still identifies the build.
  assert.equal(
    launchBuildIdentity({ appVersion: '0.0.0', installStamp: stamp({ builtAt: null }) }),
    '0.0.0+g357f51c49106'
  )
})

test('launchBuildIdentity moves between two source builds of the same version', () => {
  const before = launchBuildIdentity({ appVersion: '0.0.0', installStamp: stamp() })

  const after = launchBuildIdentity({
    appVersion: '0.0.0',
    installStamp: stamp({ commit: 'aabbccddeeff00112233445566778899aabbccdd' })
  })

  assert.notEqual(before, after)
})

test('isPlaceholderAppVersion recognizes the values that carry no release identity', () => {
  assert.equal(isPlaceholderAppVersion('0.0.0'), true)
  assert.equal(isPlaceholderAppVersion(''), true)
  assert.equal(isPlaceholderAppVersion('unknown'), true)
  assert.equal(isPlaceholderAppVersion(undefined), true)
  assert.equal(isPlaceholderAppVersion('0.21.5'), false)
})

test('a version change re-probes, as it always has', () => {
  assert.equal(launchMarkerNeedsReprobe({ version: '0.21.5' }, { appVersion: '0.22.0', buildIdentity: '0.22.0' }), true)

  assert.equal(
    launchMarkerNeedsReprobe({ version: '0.21.5' }, { appVersion: '0.21.5', buildIdentity: '0.21.5' }),
    false
  )
})

test('a build change re-probes on a source install whose version never moves', () => {
  const promoted = { version: '0.0.0', build: '0.0.0+gaaaaaaaaaaaa@2026-09-20T00:00:00Z' }

  assert.equal(
    launchMarkerNeedsReprobe(promoted, {
      appVersion: '0.0.0',
      buildIdentity: '0.0.0+gaaaaaaaaaaaa@2026-09-20T00:00:00Z'
    }),
    false
  )

  // The next source build is a different build identity at the same 0.0.0 —
  // this is the case the version check alone could never clear.
  assert.equal(
    launchMarkerNeedsReprobe(promoted, {
      appVersion: '0.0.0',
      buildIdentity: '0.0.0+gbbbbbbbbbbbb@2026-10-02T00:00:00Z'
    }),
    true
  )
})

test('a promoted marker with no build identity is re-probed once on a source install', () => {
  // Marker written by an older build, which recorded only the 0.0.0 version.
  // Nothing can distinguish it from the current build, so it gets exactly one
  // re-probe rather than staying degraded forever.
  assert.equal(launchMarkerNeedsReprobe({ version: '0.0.0' }, { appVersion: '0.0.0' }), true)

  // A real release version keeps the old contract: same version stays sticky.
  assert.equal(launchMarkerNeedsReprobe({ version: '0.21.5' }, { appVersion: '0.21.5' }), false)
})

test('launchMarkerNeedsReprobe is inert without a marker or a version', () => {
  assert.equal(launchMarkerNeedsReprobe(null, { appVersion: '0.0.0' }), false)
  assert.equal(launchMarkerNeedsReprobe({ version: '0.21.5' }, { appVersion: '' }), false)
})
