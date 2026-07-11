import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import {
  patchUnixTerminalForDarwinShortHelper,
  stageDarwinShortSpawnHelper
} from '../scripts/stage-native-deps.mjs'

test('patchUnixTerminalForDarwinShortHelper injects a darwin short-helper fallback once', () => {
  const source = `const native = loadNativeModule('pty');
let helperPath = native.dir + '/spawn-helper';
helperPath = path.resolve(__dirname, helperPath);
helperPath = helperPath.replace('app.asar', 'app.asar.unpacked');
helperPath = helperPath.replace('node_modules.asar', 'node_modules.asar.unpacked');
const DEFAULT_FILE = 'sh';`

  const patched = patchUnixTerminalForDarwinShortHelper(source)
  assert.match(patched, /process\.platform === 'darwin'/)
  assert.match(patched, /fs\.existsSync\(shortHelperPath\)/)
  assert.match(patched, /\.\.\/\.\.\/\.\.\/bin\/spawn-helper/)
  assert.equal(patchUnixTerminalForDarwinShortHelper(patched), patched)
})

test('stageDarwinShortSpawnHelper copies a short helper and patches unixTerminal.js', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-stage-node-pty-'))
  try {
    const destRoot = path.join(tempRoot, 'dist', 'node_modules', 'node-pty')
    const prebuildDir = path.join(destRoot, 'prebuilds', 'darwin-arm64')
    const libDir = path.join(destRoot, 'lib')
    const shortHelperPath = path.join(tempRoot, 'dist', 'bin', 'spawn-helper')

    fs.mkdirSync(prebuildDir, { recursive: true })
    fs.mkdirSync(libDir, { recursive: true })
    fs.writeFileSync(path.join(prebuildDir, 'spawn-helper'), '#!/bin/sh\nexit 0\n', { mode: 0o755 })
    fs.writeFileSync(
      path.join(libDir, 'unixTerminal.js'),
      `let helperPath = native.dir + '/spawn-helper';
helperPath = path.resolve(__dirname, helperPath);
helperPath = helperPath.replace('app.asar', 'app.asar.unpacked');
helperPath = helperPath.replace('node_modules.asar', 'node_modules.asar.unpacked');
`,
      'utf8'
    )

    const staged = stageDarwinShortSpawnHelper(destRoot, {
      platform: 'darwin',
      arch: 'arm64',
      shortHelperPath
    })

    assert.equal(staged, shortHelperPath)
    assert.equal(fs.existsSync(shortHelperPath), true)
    assert.match(fs.readFileSync(path.join(libDir, 'unixTerminal.js'), 'utf8'), /\.\.\/\.\.\/\.\.\/bin\/spawn-helper/)
    assert.equal(fs.statSync(shortHelperPath).mode & 0o777, 0o755)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})
