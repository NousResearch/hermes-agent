import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import { createRendererHealthReporter } from './renderer-health'

function tempRoot(tag: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), `hermes-renderer-health-${tag}-`))
}

function readMarker(markerPath: string): Record<string, unknown> {
  return JSON.parse(fs.readFileSync(markerPath, 'utf8'))
}

test('writes starting then explicit renderer-ready state with updater identity', () => {
  const root = tempRoot('ready')
  const markerPath = path.join(root, 'hermes-desktop-health-test.json')
  const reporter = createRendererHealthReporter({
    markerPath,
    nonce: 'nonce-0123456789abcdef',
    pid: 4242,
    executable: 'C:\\Hermes\\Hermes.exe',
    tempDir: root,
    now: () => 1_000
  })

  assert.equal(reporter.enabled, true)
  assert.deepEqual(readMarker(markerPath), {
    schema: 1,
    nonce: 'nonce-0123456789abcdef',
    pid: 4242,
    executable: 'C:\\Hermes\\Hermes.exe',
    status: 'starting',
    failure_count: 0,
    failure_reason: null,
    updated_at_ms: 1_000
  })

  reporter.ready()
  assert.equal(readMarker(markerPath).status, 'ready')
  assert.equal(readMarker(markerPath).failure_count, 0)
})

test('renderer failure is sticky across a later ready handshake', () => {
  const root = tempRoot('failure')
  const markerPath = path.join(root, 'hermes-desktop-health-test.json')
  const reporter = createRendererHealthReporter({
    markerPath,
    nonce: 'nonce-0123456789abcdef',
    pid: 4242,
    executable: 'C:\\Hermes\\Hermes.exe',
    tempDir: root,
    now: () => 2_000
  })

  reporter.fail('render-process-gone:crashed')
  let marker = readMarker(markerPath)
  assert.equal(marker.status, 'failed')
  assert.equal(marker.failure_count, 1)
  assert.equal(marker.failure_reason, 'render-process-gone:crashed')

  reporter.ready()
  marker = readMarker(markerPath)
  assert.equal(marker.status, 'ready')
  assert.equal(marker.failure_count, 1, 'updater must still reject a recovered crash loop')
  assert.equal(marker.failure_reason, 'render-process-gone:crashed')
})

test('uses the updater-provided temp root when TEMP and TMP differ', () => {
  const root = tempRoot('env-temp-root')
  const markerPath = path.join(root, 'hermes-desktop-health-test.json')
  const previousMarker = process.env.HERMES_DESKTOP_HEALTH_MARKER
  const previousNonce = process.env.HERMES_DESKTOP_HEALTH_NONCE
  const previousTempDir = process.env.HERMES_DESKTOP_HEALTH_TEMP_DIR
  try {
    process.env.HERMES_DESKTOP_HEALTH_MARKER = markerPath
    process.env.HERMES_DESKTOP_HEALTH_NONCE = 'nonce-0123456789abcdef'
    process.env.HERMES_DESKTOP_HEALTH_TEMP_DIR = root
    const reporter = createRendererHealthReporter({ pid: 4242 })
    assert.equal(reporter.enabled, true)
    assert.equal(readMarker(markerPath).status, 'starting')
  } finally {
    if (previousMarker === undefined) {
      delete process.env.HERMES_DESKTOP_HEALTH_MARKER
    } else {
      process.env.HERMES_DESKTOP_HEALTH_MARKER = previousMarker
    }
    if (previousNonce === undefined) {
      delete process.env.HERMES_DESKTOP_HEALTH_NONCE
    } else {
      process.env.HERMES_DESKTOP_HEALTH_NONCE = previousNonce
    }
    if (previousTempDir === undefined) {
      delete process.env.HERMES_DESKTOP_HEALTH_TEMP_DIR
    } else {
      process.env.HERMES_DESKTOP_HEALTH_TEMP_DIR = previousTempDir
    }
  }
})

test('refuses marker paths outside the injected temp directory', () => {
  const root = tempRoot('outside-root')
  const outside = tempRoot('outside-target')
  const markerPath = path.join(outside, 'hermes-desktop-health-test.json')
  const reporter = createRendererHealthReporter({
    markerPath,
    nonce: 'nonce-0123456789abcdef',
    tempDir: root
  })

  assert.equal(reporter.enabled, false)
  reporter.ready()
  assert.equal(fs.existsSync(markerPath), false)
})

test('refuses to follow a dangling symlink marker file', t => {
  const root = tempRoot('dangling-symlink')
  const outside = path.join(root, 'not-created.json')
  const markerPath = path.join(root, 'hermes-desktop-health-test.json')
  try {
    fs.symlinkSync(outside, markerPath, 'file')
  } catch (error) {
    t.skip(`file symlink unavailable: ${String(error)}`)
    return
  }

  const reporter = createRendererHealthReporter({
    markerPath,
    nonce: 'nonce-0123456789abcdef',
    tempDir: root
  })
  assert.equal(reporter.enabled, false)
  reporter.ready()
  assert.equal(fs.existsSync(outside), false)
})

test('refuses to follow a symlink marker file', t => {
  const root = tempRoot('symlink')
  const outside = path.join(root, 'outside.json')
  const markerPath = path.join(root, 'hermes-desktop-health-test.json')
  fs.writeFileSync(outside, 'sentinel', 'utf8')
  try {
    fs.symlinkSync(outside, markerPath, 'file')
  } catch (error) {
    t.skip(`file symlink unavailable: ${String(error)}`)
    return
  }

  const reporter = createRendererHealthReporter({
    markerPath,
    nonce: 'nonce-0123456789abcdef',
    tempDir: root
  })
  assert.equal(reporter.enabled, false)
  reporter.ready()
  assert.equal(fs.readFileSync(outside, 'utf8'), 'sentinel')
})
