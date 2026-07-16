/**
 * Tests for electron/mac-rebuilt-app.ts — resolving the rebuilt macOS `.app`
 * bundle for the in-app update swap/relaunch.
 *
 * Why this matters: electron-builder writes the rebuilt bundle under an
 * arch-specific dir. The updater only looked at `mac-arm64` and `mac`, so an
 * Intel rebuild in `release/mac-x64` was missed and the update never installed —
 * it degraded to "Restart Hermes to load the new version" (issue #48160).
 * Resolution must follow the host arch, the same way `scripts/test-desktop.mjs`
 * derives the packaged-app directory.
 */

import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { resolveRebuiltMacApp } from './mac-rebuilt-app'

const ROOT = '/install'
const bundle = dir => path.join(ROOT, 'apps', 'desktop', 'release', dir, 'Hermes.app')
const onlyExists = (...present) => candidate => present.includes(candidate)

test('an Intel/x64 rebuild resolves release/mac-x64', () => {
  const exists = onlyExists(bundle('mac-x64'))

  assert.equal(resolveRebuiltMacApp(ROOT, { arch: 'x64', exists }), bundle('mac-x64'))
})

test('an Apple Silicon rebuild resolves release/mac-arm64', () => {
  const exists = onlyExists(bundle('mac-arm64'))

  assert.equal(resolveRebuiltMacApp(ROOT, { arch: 'arm64', exists }), bundle('mac-arm64'))
})

test('a host-arch (no explicit arch) build falls back to release/mac', () => {
  const exists = onlyExists(bundle('mac'))

  assert.equal(resolveRebuiltMacApp(ROOT, { arch: 'x64', exists }), bundle('mac'))
  assert.equal(resolveRebuiltMacApp(ROOT, { arch: 'arm64', exists }), bundle('mac'))
})

test('an Intel host never selects a stale mac-arm64 bundle', () => {
  // Both arch dirs present (e.g. left by an earlier cross-build): an x64 host
  // must install the x64 bundle, not the arm64 one.
  const exists = onlyExists(bundle('mac-arm64'), bundle('mac-x64'))

  assert.equal(resolveRebuiltMacApp(ROOT, { arch: 'x64', exists }), bundle('mac-x64'))
})

test('a non-x64/non-arm64 arch only considers the generic release/mac dir', () => {
  const exists = onlyExists(bundle('mac-arm64'), bundle('mac-x64'), bundle('mac'))

  assert.equal(resolveRebuiltMacApp(ROOT, { arch: 'ppc64', exists }), bundle('mac'))
})

test('returns undefined when no rebuilt bundle exists', () => {
  assert.equal(resolveRebuiltMacApp(ROOT, { arch: 'x64', exists: () => false }), undefined)
})
