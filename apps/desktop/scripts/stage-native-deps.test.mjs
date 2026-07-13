import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import { stageNodePty } from './stage-native-deps.mjs'

function makeNodePtyFixture() {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-stage-native-deps-'))
  const sourceRoot = path.join(tempRoot, 'node-pty')
  const destRoot = path.join(tempRoot, 'dist', 'node_modules', 'node-pty')

  fs.mkdirSync(path.join(sourceRoot, 'lib'), { recursive: true })
  fs.writeFileSync(path.join(sourceRoot, 'package.json'), '{"main":"lib/index.js"}\n')
  fs.writeFileSync(path.join(sourceRoot, 'lib', 'index.js'), 'export {}\n')

  return { tempRoot, sourceRoot, destRoot }
}

function writeFileWithMode(file, mode) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, 'binary')
  fs.chmodSync(file, mode)
}

function modeBits(file) {
  return fs.statSync(file).mode & 0o777
}

test('stageNodePty makes prebuilt darwin spawn-helper executable', () => {
  const { tempRoot, sourceRoot, destRoot } = makeNodePtyFixture()
  try {
    writeFileWithMode(path.join(sourceRoot, 'prebuilds', 'darwin-arm64', 'pty.node'), 0o644)
    writeFileWithMode(path.join(sourceRoot, 'prebuilds', 'darwin-arm64', 'spawn-helper'), 0o644)

    stageNodePty({ platform: 'darwin', arch: 'arm64', sourceRoot, destRoot })

    const helper = path.join(destRoot, 'prebuilds', 'darwin-arm64', 'spawn-helper')
    const native = path.join(destRoot, 'prebuilds', 'darwin-arm64', 'pty.node')

    assert.equal(modeBits(helper), 0o755)
    assert.equal(modeBits(native), 0o644)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('stageNodePty makes build/Release spawn-helper executable', () => {
  const { tempRoot, sourceRoot, destRoot } = makeNodePtyFixture()
  try {
    writeFileWithMode(path.join(sourceRoot, 'build', 'Release', 'pty.node'), 0o644)
    writeFileWithMode(path.join(sourceRoot, 'build', 'Release', 'spawn-helper'), 0o644)
    writeFileWithMode(path.join(sourceRoot, 'prebuilds', 'linux-x64', 'pty.node'), 0o644)

    stageNodePty({ platform: 'linux', arch: 'x64', sourceRoot, destRoot })

    const helper = path.join(destRoot, 'build', 'Release', 'spawn-helper')
    const native = path.join(destRoot, 'build', 'Release', 'pty.node')

    assert.equal(modeBits(helper), 0o755)
    assert.equal(modeBits(native), 0o644)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})
