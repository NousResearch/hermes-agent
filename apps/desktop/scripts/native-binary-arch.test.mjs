import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import {
  archNameFromBuilderValue,
  assertMachOArchitecture,
  classifyMachOArchitectures,
  electronBuilderArchFlag,
  hasExplicitArchFlag
} from './native-binary-arch.mjs'

const CPU_TYPE_X86_64 = 0x01000007
const CPU_TYPE_ARM64 = 0x0100000c

function withTempBinary(buffer, callback) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-macho-'))
  const file = path.join(root, 'Electron')
  try {
    fs.writeFileSync(file, buffer)
    callback(file)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}

function thinMachO(cpuType) {
  const buffer = Buffer.alloc(32)
  buffer.writeUInt32LE(0xfeedfacf, 0)
  buffer.writeUInt32LE(cpuType, 4)
  return buffer
}

function universalMachO(cpuTypes) {
  const buffer = Buffer.alloc(8 + cpuTypes.length * 20)
  buffer.writeUInt32BE(0xcafebabe, 0)
  buffer.writeUInt32BE(cpuTypes.length, 4)
  cpuTypes.forEach((cpuType, index) => buffer.writeUInt32BE(cpuType, 8 + index * 20))
  return buffer
}

test('classifies thin arm64 Mach-O', () => {
  withTempBinary(thinMachO(CPU_TYPE_ARM64), (file) => {
    assert.deepEqual([...classifyMachOArchitectures(file)], ['arm64'])
  })
})

test('classifies thin x64 Mach-O', () => {
  withTempBinary(thinMachO(CPU_TYPE_X86_64), (file) => {
    assert.deepEqual([...classifyMachOArchitectures(file)], ['x64'])
  })
})

test('classifies universal arm64 and x64 Mach-O', () => {
  withTempBinary(universalMachO([CPU_TYPE_X86_64, CPU_TYPE_ARM64]), (file) => {
    assert.deepEqual([...classifyMachOArchitectures(file)], ['x64', 'arm64'])
  })
})

test('maps Electron architecture to builder flag', () => {
  assert.equal(electronBuilderArchFlag(new Set(['arm64'])), '--arm64')
  assert.equal(electronBuilderArchFlag(new Set(['x64'])), '--x64')
  assert.equal(electronBuilderArchFlag(new Set(['x64', 'arm64'])), '--universal')
})

test('preserves explicit builder architecture', () => {
  assert.equal(hasExplicitArchFlag(['--mac', '--arm64']), true)
  assert.equal(hasExplicitArchFlag(['--arch=x64']), true)
  assert.equal(hasExplicitArchFlag(['--dir']), false)
})

test('maps electron-builder numeric architecture values', () => {
  assert.equal(archNameFromBuilderValue(1), 'x64')
  assert.equal(archNameFromBuilderValue(3), 'arm64')
  assert.equal(archNameFromBuilderValue(4), 'universal')
})

test('architecture gate rejects a mismatched packed binary', () => {
  withTempBinary(thinMachO(CPU_TYPE_ARM64), (file) => {
    assert.throws(() => assertMachOArchitecture(file, 'x64'), /expected x64, got arm64/)
    assert.doesNotThrow(() => assertMachOArchitecture(file, 'arm64'))
  })
})
