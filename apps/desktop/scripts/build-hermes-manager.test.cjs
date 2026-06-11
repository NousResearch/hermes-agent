'use strict'

/**
 * Tests for build-hermes-manager.cjs release build helpers.
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  buildCargoArgs,
  normalizeTargetPlatform,
  shouldAllowHostBuild
} = require('./build-hermes-manager.cjs')

test('buildCargoArgs builds the release manager cargo arguments', () => {
  assert.deepEqual(buildCargoArgs('D:\\repo'), [
    'build',
    '--release',
    '--manifest-path',
    'D:\\repo\\apps\\hermes-manager\\Cargo.toml'
  ])
})

test('normalizeTargetPlatform maps desktop aliases', () => {
  assert.equal(normalizeTargetPlatform('win'), 'win32')
  assert.equal(normalizeTargetPlatform('mac'), 'darwin')
  assert.equal(normalizeTargetPlatform('linux'), 'linux')
  assert.equal(normalizeTargetPlatform('win32'), 'win32')
})

test('shouldAllowHostBuild allows same-platform release builds', () => {
  assert.equal(shouldAllowHostBuild({ hostPlatform: 'win32', targetPlatform: 'win32' }), true)
  assert.equal(shouldAllowHostBuild({ hostPlatform: 'darwin', targetPlatform: 'mac' }), true)
})

test('shouldAllowHostBuild rejects cross-target release builds', () => {
  assert.equal(shouldAllowHostBuild({ hostPlatform: 'linux', targetPlatform: 'win32' }), false)
  assert.equal(shouldAllowHostBuild({ hostPlatform: 'win32', targetPlatform: 'darwin' }), false)
})

