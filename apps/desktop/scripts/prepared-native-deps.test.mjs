import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import * as native from './prepared-native-deps.mjs'
import beforePack from './before-pack.mjs'

test('beforePack refuses absent native preparation rather than staging from npm', async () => {
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'native-hook-'))
  try {
    await assert.rejects(beforePack({ appOutDir: '', electronPlatformName: 'linux', arch: 1,
      packager: { projectDir: path.join(source, 'apps/desktop') } }), /run preparation again/)
  } finally {
    fs.rmSync(source, { recursive: true, force: true })
  }
})

test('native consumption copies admitted inputs and rejects a different target or changed binding', () => {
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-native-'))
  try {
    const out = path.join(source, 'native')
    fs.mkdirSync(path.join(out, 'node-pty'), { recursive: true })
    fs.mkdirSync(path.join(source, 'apps/desktop'), { recursive: true })
    fs.writeFileSync(path.join(source, 'package-lock.json'), '{}')
    fs.writeFileSync(path.join(source, 'apps/desktop/package.json'), '{}')
    fs.writeFileSync(path.join(out, 'node-pty/package.json'), '{}')
    const binding = path.join(out, 'node-pty/pty.node')
    fs.writeFileSync(binding, 'native fixture')
    native.recordNativeInputs({ source, out, platform: 'linux', arch: 'x64', nativeToolchain: 'compiler-a' })
    assert.equal(native.readNativeInputs({ source, nativeDeps: out, platform: 'linux', arch: 'x64', nativeToolchain: 'compiler-a' }), out)
    assert.throws(() => native.readNativeInputs({ source, nativeDeps: out, platform: 'linux', arch: 'x64', nativeToolchain: 'compiler-b' }), /run preparation again/)
    const destination = path.join(source, 'product/node_modules')
    native.copyNativeInputs({ source, nativeDeps: out, out: destination, platform: 'linux', arch: 'x64' })
    fs.writeFileSync(path.join(destination, 'node-pty/pty.node'), 'product mutation')
    assert.equal(fs.readFileSync(binding, 'utf8'), 'native fixture')
    assert.throws(() => native.readNativeInputs({ source, nativeDeps: out, platform: 'linux', arch: 'arm64' }), /run preparation again/)
    fs.writeFileSync(binding, 'corrupt')
    assert.throws(() => native.copyNativeInputs({ source, nativeDeps: out, out: destination, platform: 'linux', arch: 'x64' }), /run preparation again/)
    assert.equal(fs.readFileSync(path.join(destination, 'node-pty/pty.node'), 'utf8'), 'product mutation')
  } finally {
    fs.rmSync(source, { recursive: true, force: true })
  }
})
