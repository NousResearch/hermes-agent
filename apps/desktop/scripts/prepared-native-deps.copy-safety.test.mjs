import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test, vi } from 'vitest'
import * as native from './prepared-native-deps.mjs'
import { copyTreeSync, removeTreeSync } from './prepared-native-deps.mjs'

test('removeTreeSync removes nested trees and tolerates missing paths', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'remove-tree-'))
  try {
    fs.mkdirSync(path.join(root, 'a/b/c'), { recursive: true })
    fs.writeFileSync(path.join(root, 'a/b/c/file.txt'), 'x')
    removeTreeSync(path.join(root, 'a'))
    assert.equal(fs.existsSync(path.join(root, 'a')), false)
    removeTreeSync(path.join(root, 'a')) // missing paths are fine
    removeTreeSync(path.join(root, 'missing-file.txt'))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('copyTreeSync copies recursively, preserves modes and honours skip', () => {
  const src = fs.mkdtempSync(path.join(os.tmpdir(), 'copy-src-'))
  const dest = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'copy-dst-')), 'out')
  try {
    fs.mkdirSync(path.join(src, 'nested'), { recursive: true })
    fs.writeFileSync(path.join(src, 'top.txt'), 'top')
    fs.writeFileSync(path.join(src, 'nested/deep.txt'), 'deep')
    fs.writeFileSync(path.join(src, 'skipme.bin'), 'skipped')
    if (process.platform !== 'win32') fs.chmodSync(path.join(src, 'top.txt'), 0o755)
    copyTreeSync(src, dest, src => src.endsWith('skipme.bin'))
    assert.equal(fs.readFileSync(path.join(dest, 'top.txt'), 'utf8'), 'top')
    assert.equal(fs.readFileSync(path.join(dest, 'nested/deep.txt'), 'utf8'), 'deep')
    assert.equal(fs.existsSync(path.join(dest, 'skipme.bin')), false)
    if (process.platform !== 'win32') assert.equal(fs.statSync(path.join(dest, 'top.txt')).mode & 0o777, 0o755)
  } finally {
    fs.rmSync(src, { recursive: true, force: true })
    fs.rmSync(path.dirname(dest), { recursive: true, force: true })
  }
})

test('copyNativeTree fails loudly when a helper silently fails to land (#127420)', () => {
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'native-silent-'))
  try {
    const app = path.join(source, 'apps/desktop')
    const out = path.join(app, 'build/native-deps')
    fs.mkdirSync(path.join(out, 'node-pty'), { recursive: true })
    fs.cpSync(path.join(import.meta.dirname, '../electron/native'), path.join(app, 'electron/native'), { recursive: true })
    fs.writeFileSync(path.join(source, 'package-lock.json'), '{}')
    fs.writeFileSync(path.join(app, 'package.json'), '{}')
    fs.writeFileSync(path.join(out, 'node-pty/package.json'), '{}')
    fs.writeFileSync(path.join(out, 'node-pty/pty.node'), 'native fixture')
    // The Windows helper layout the issue fails on:
    const helper = 'native/win32-x64/hud-modifier-monitor.exe'
    fs.mkdirSync(path.dirname(path.join(out, helper)), { recursive: true })
    fs.writeFileSync(path.join(out, helper), 'helper fixture', { mode: 0o755 })
    const selection = { source, nativeDeps: out, platform: 'win32', arch: 'x64', nativeToolchain: 'compiler-a' }
    native.recordNativeInputs({ ...selection, out })

    // The bug shape: the native (non-libuv) fs rewrite silently no-ops the copy of
    // the one helper on a non-ASCII Windows profile. The copy must detect the
    // missing helper instead of leaving electron-builder to fail with ENOENT.
    const realCopy = fs.copyFileSync
    const spy = vi.spyOn(fs, 'copyFileSync').mockImplementation((src, dest) => {
      if (String(dest).endsWith('hud-modifier-monitor.exe')) return undefined
      return realCopy(src, dest)
    })
    try {
      const destination = path.join(app, 'dist/node_modules')
      assert.throws(
        () => native.copyNativeInputs({ ...selection, out: destination }),
        /incomplete.*hud-modifier-monitor\.exe/)
    } finally {
      spy.mockRestore()
    }
  } finally {
    fs.rmSync(source, { recursive: true, force: true })
  }
})
