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

// Big-endian on-disk thin Mach-O (MH_MAGIC_64 read as-is, cpuType big-endian).
function thinMachOBigEndian(cpuType) {
  const buffer = Buffer.alloc(32)
  buffer.writeUInt32BE(0xfeedfacf, 0)
  buffer.writeUInt32BE(cpuType, 4)
  return buffer
}

function universalMachO(cpuTypes) {
  const buffer = Buffer.alloc(8 + cpuTypes.length * 20)
  buffer.writeUInt32BE(0xcafebabe, 0)
  buffer.writeUInt32BE(cpuTypes.length, 4)
  cpuTypes.forEach((cpuType, index) => buffer.writeUInt32BE(cpuType, 8 + index * 20))
  return buffer
}

// Fat Mach-O in an arbitrary encoding, to exercise the byte-swapped and 64-bit
// header variants that only differ in magic, endianness, and entry stride.
function fatMachO({ magic, littleEndian, is64, cpuTypes }) {
  const entrySize = is64 ? 32 : 20
  const buffer = Buffer.alloc(8 + cpuTypes.length * entrySize)
  const writeU32 = (value, offset) =>
    littleEndian ? buffer.writeUInt32LE(value, offset) : buffer.writeUInt32BE(value, offset)
  // The magic is compared after a fixed big-endian read, so write it big-endian.
  buffer.writeUInt32BE(magic, 0)
  writeU32(cpuTypes.length, 4)
  cpuTypes.forEach((cpuType, index) => writeU32(cpuType, 8 + index * entrySize))
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

test('classifies a big-endian thin Mach-O', () => {
  withTempBinary(thinMachOBigEndian(CPU_TYPE_ARM64), (file) => {
    assert.deepEqual([...classifyMachOArchitectures(file)], ['arm64'])
  })
})

test('classifies a byte-swapped fat Mach-O (FAT_CIGAM)', () => {
  const buffer = fatMachO({ magic: 0xbebafeca, littleEndian: true, is64: false, cpuTypes: [CPU_TYPE_X86_64, CPU_TYPE_ARM64] })
  withTempBinary(buffer, (file) => {
    assert.deepEqual([...classifyMachOArchitectures(file)], ['x64', 'arm64'])
  })
})

test('classifies a 64-bit fat Mach-O (FAT_MAGIC_64)', () => {
  const buffer = fatMachO({ magic: 0xcafebabf, littleEndian: false, is64: true, cpuTypes: [CPU_TYPE_ARM64] })
  withTempBinary(buffer, (file) => {
    assert.deepEqual([...classifyMachOArchitectures(file)], ['arm64'])
  })
})

test('classifies a byte-swapped 64-bit fat Mach-O (FAT_CIGAM_64)', () => {
  const buffer = fatMachO({ magic: 0xbfbafeca, littleEndian: true, is64: true, cpuTypes: [CPU_TYPE_X86_64] })
  withTempBinary(buffer, (file) => {
    assert.deepEqual([...classifyMachOArchitectures(file)], ['x64'])
  })
})

test('returns null for a file smaller than the header', () => {
  withTempBinary(Buffer.alloc(4), (file) => {
    assert.equal(classifyMachOArchitectures(file), null)
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
