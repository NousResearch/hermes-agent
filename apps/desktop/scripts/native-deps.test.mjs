import assert from 'node:assert/strict'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { assertNodePtyNativePayload, findNodePtyNativePayload } from './stage-native-deps.mjs'

function withTempRoot(run) {
  const root = mkdtempSync(join(tmpdir(), 'hermes-node-pty-'))
  try {
    run(root)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

function writePayload(dir, { spawnHelper = false } = {}) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'pty.node'), '')
  if (spawnHelper) writeFileSync(join(dir, 'spawn-helper'), '')
}

test('findNodePtyNativePayload accepts a target prebuild', () => {
  withTempRoot(root => {
    const payload = join(root, 'prebuilds', 'linux-x64')
    writePayload(payload)

    assert.equal(findNodePtyNativePayload({ root, platform: 'linux', arch: 'x64' }), payload)
  })
})

test('findNodePtyNativePayload accepts a local build output', () => {
  withTempRoot(root => {
    const payload = join(root, 'build', 'Release')
    writePayload(payload, { spawnHelper: process.platform === 'darwin' })

    assert.equal(findNodePtyNativePayload({ root, platform: process.platform, arch: process.arch }), payload)
  })
})

test('findNodePtyNativePayload does not reuse a host build for a cross-target package', () => {
  withTempRoot(root => {
    writePayload(join(root, 'build', 'Release'), { spawnHelper: true })
    const targetPlatform = process.platform === 'linux' ? 'win32' : 'linux'

    assert.equal(findNodePtyNativePayload({ root, platform: targetPlatform, arch: process.arch }), undefined)
  })
})

test('assertNodePtyNativePayload fails when pty.node is absent', () => {
  withTempRoot(root => {
    assert.throws(
      () => assertNodePtyNativePayload(root, { platform: 'linux', arch: 'x64' }),
      /no usable native payload/
    )
  })
})

test('assertNodePtyNativePayload accepts a staged build payload', () => {
  withTempRoot(root => {
    const payload = join(root, 'build', 'Release')
    writePayload(payload)

    assert.equal(assertNodePtyNativePayload(root, { platform: 'linux', arch: 'x64' }), payload)
  })
})
