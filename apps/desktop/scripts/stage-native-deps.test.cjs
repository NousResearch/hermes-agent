'use strict'

/**
 * Tests for stage-native-deps.cjs manager binary staging.
 *
 * These tests isolate the hermes-manager resource copy behavior from the
 * node-pty staging precondition, so stale manager artifacts cannot silently
 * survive when the Rust release binary is absent.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const { normalizeTargetPlatform, stageManagerBinary } = require('./stage-native-deps.cjs')

function withTempLayout(fn) {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-stage-native-deps-'))
  try {
    const repoRoot = path.join(tempRoot, 'repo')
    const appRoot = path.join(repoRoot, 'apps', 'desktop')
    fs.mkdirSync(appRoot, { recursive: true })
    fn({ appRoot, repoRoot })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
}

test('stageManagerBinary clears stale destination when manager source is absent', () => {
  withTempLayout(({ appRoot, repoRoot }) => {
    const destDir = path.join(appRoot, 'build', 'hermes-manager')
    fs.mkdirSync(destDir, { recursive: true })
    fs.writeFileSync(path.join(destDir, 'hermes-manager.exe'), 'stale')

    const logs = []
    stageManagerBinary({ appRoot, log: message => logs.push(message), platform: 'win32', repoRoot })

    assert.equal(fs.existsSync(destDir), false)
    assert.equal(logs.length, 1)
    assert.match(logs[0], /not found; packaged app will use existing Python uninstall fallback/)
  })
})

test('stageManagerBinary replaces stale destination with the prebuilt manager', () => {
  withTempLayout(({ appRoot, repoRoot }) => {
    const sourceDir = path.join(repoRoot, 'apps', 'hermes-manager', 'target', 'release')
    const source = path.join(sourceDir, 'hermes-manager.exe')
    const destDir = path.join(appRoot, 'build', 'hermes-manager')
    fs.mkdirSync(sourceDir, { recursive: true })
    fs.mkdirSync(destDir, { recursive: true })
    fs.writeFileSync(source, 'fresh')
    fs.writeFileSync(path.join(destDir, 'extra-stale-file'), 'stale')

    const logs = []
    stageManagerBinary({
      appRoot,
      hermesVersion: '9.9.9',
      log: message => logs.push(message),
      platform: 'win32',
      repoRoot,
      sourceCommit: 'abc1234'
    })

    assert.equal(fs.readFileSync(path.join(destDir, 'hermes-manager.exe'), 'utf8'), 'fresh')
    assert.equal(fs.existsSync(path.join(destDir, 'extra-stale-file')), false)
    const manifest = JSON.parse(fs.readFileSync(path.join(destDir, 'bundled-manifest.json'), 'utf8'))
    assert.equal(manifest.schema_version, 1)
    assert.equal(manifest.hermes_version, '9.9.9')
    assert.equal(manifest.source_commit, 'abc1234')
    assert.deepEqual(manifest.resources, [
      {
        kind: 'tool',
        path: 'hermes-manager.exe',
        sha256: 'd098ab5e44b9aabb755f76d806598f43573c662b35e4a2eab1e312ec9ad195e2'
      }
    ])
    assert.equal(logs.length, 1)
    assert.match(logs[0], /hermes-manager: copied build[\\/]hermes-manager[\\/]hermes-manager\.exe/)
  })
})

test('stageManagerBinary uses the requested target platform name', () => {
  withTempLayout(({ appRoot, repoRoot }) => {
    const sourceDir = path.join(repoRoot, 'apps', 'hermes-manager', 'target', 'release')
    const source = path.join(sourceDir, 'hermes-manager.exe')
    fs.mkdirSync(sourceDir, { recursive: true })
    fs.writeFileSync(source, 'windows')

    const logs = []
    stageManagerBinary({ appRoot, log: message => logs.push(message), platform: 'win', repoRoot })

    const stagedManager = path.join(appRoot, 'build', 'hermes-manager', 'hermes-manager.exe')
    assert.equal(fs.readFileSync(stagedManager, 'utf8'), 'windows')
    assert.equal(logs.length, 1)
    assert.match(logs[0], /hermes-manager\.exe/)
  })
})

test('normalizeTargetPlatform maps electron-builder aliases', () => {
  assert.equal(normalizeTargetPlatform('win'), 'win32')
  assert.equal(normalizeTargetPlatform('mac'), 'darwin')
  assert.equal(normalizeTargetPlatform('linux'), 'linux')
})
