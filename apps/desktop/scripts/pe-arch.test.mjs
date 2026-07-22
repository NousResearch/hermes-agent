import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import {
  makeMinimalPeBuffer,
  normalizeCpuArch,
  peArchMatches,
  readPeArch
} from './pe-arch.mjs'
import {
  resolveBuilderTargetArch,
  shouldReuseElectronDist
} from './run-electron-builder-lib.mjs'

test('normalizeCpuArch maps common aliases', () => {
  assert.equal(normalizeCpuArch('AMD64'), 'x64')
  assert.equal(normalizeCpuArch('x86_64'), 'x64')
  assert.equal(normalizeCpuArch('aarch64'), 'arm64')
  assert.equal(normalizeCpuArch('x86'), 'ia32')
  assert.equal(normalizeCpuArch('nope'), null)
})

test('readPeArch returns COFF Machine for minimal PE fixtures', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pe-arch-'))
  try {
    for (const arch of ['x64', 'arm64', 'ia32']) {
      const file = path.join(dir, `${arch}.exe`)
      fs.writeFileSync(file, makeMinimalPeBuffer(arch))
      assert.equal(readPeArch(file), arch)
      assert.equal(peArchMatches(file, arch), true)
      assert.equal(peArchMatches(file, arch === 'x64' ? 'arm64' : 'x64'), false)
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('readPeArch returns null for non-PE and truncated files', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pe-arch-bad-'))
  try {
    const empty = path.join(dir, 'empty.exe')
    fs.writeFileSync(empty, '')
    assert.equal(readPeArch(empty), null)

    const text = path.join(dir, 'text.exe')
    fs.writeFileSync(text, 'not a pe')
    assert.equal(readPeArch(text), null)

    assert.equal(readPeArch(path.join(dir, 'missing.exe')), null)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('resolveBuilderTargetArch prefers explicit flags over host', () => {
  assert.equal(resolveBuilderTargetArch(['--dir', '--arm64'], 'x64'), 'arm64')
  assert.equal(resolveBuilderTargetArch(['--arch=ia32'], 'x64'), 'ia32')
  assert.equal(resolveBuilderTargetArch(['--arch', 'x64'], 'arm64'), 'x64')
  assert.equal(resolveBuilderTargetArch(['--dir'], 'arm64'), 'arm64')
})

test('shouldReuseElectronDist refuses Windows arch-mismatched PE (#69179)', () => {
  const decision = shouldReuseElectronDist({
    platform: 'win32',
    distDir: '/fake/dist',
    binaryPath: '/fake/dist/electron.exe',
    targetArch: 'x64',
    hostArch: 'x64',
    existsSync: () => true,
    peArchReader: () => 'arm64'
  })
  assert.deepEqual(decision, {
    reuse: false,
    reason: 'arch-mismatch',
    got: 'arm64',
    want: 'x64'
  })
})

test('shouldReuseElectronDist reuses matching Windows PE', () => {
  const decision = shouldReuseElectronDist({
    platform: 'win32',
    distDir: '/fake/dist',
    binaryPath: '/fake/dist/electron.exe',
    targetArch: 'x64',
    hostArch: 'x64',
    existsSync: () => true,
    peArchReader: () => 'x64'
  })
  assert.equal(decision.reuse, true)
  assert.equal(decision.reason, 'arch-match')
})

test('shouldReuseElectronDist refuses unreadable Windows PE', () => {
  const decision = shouldReuseElectronDist({
    platform: 'win32',
    distDir: '/fake/dist',
    binaryPath: '/fake/dist/electron.exe',
    targetArch: 'x64',
    existsSync: () => true,
    peArchReader: () => null
  })
  assert.equal(decision.reuse, false)
  assert.equal(decision.reason, 'unreadable-pe')
})

test('shouldReuseElectronDist skips PE gate on non-Windows hosts', () => {
  const decision = shouldReuseElectronDist({
    platform: 'darwin',
    distDir: '/fake/dist',
    binaryPath: '/fake/dist/Electron',
    targetArch: 'arm64',
    existsSync: () => true,
    peArchReader: () => {
      throw new Error('should not read PE on darwin')
    }
  })
  assert.equal(decision.reuse, true)
  assert.equal(decision.reason, 'non-windows')
})
