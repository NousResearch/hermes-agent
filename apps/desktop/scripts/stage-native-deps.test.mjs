import assert from 'node:assert/strict'
import fs, { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import {
  stageNodePtyInto,
  classifyNativeBinary
} from '../scripts/stage-native-deps.mjs'

const { join } = path

// ─── fixtures ──────────────────────────────────────────────────────
//
// Create minimal fake .node files with correct magic bytes so the
// binary classifier and the staging validator exercise real code paths
// without needing actual native modules.

/** Write a fake .node file with the given platform's magic bytes. */
function makeFakeNode(filePath, platform) {
  const headers = {
    linux:   Buffer.from([0x7f, 0x45, 0x4c, 0x46, 0x00, 0x00, 0x00, 0x00]), // ELF
    // On x64/arm64 Darwin, Mach-O binaries are stored little-endian on disk
    // (MH_CIGAM_64 = cffaedfe). This is the form node-pty's prebuilds ship in.
    darwin:  Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 0x00, 0x00, 0x00, 0x00]), // Mach-O 64-bit LE (CIGAM_64)
    win32:   Buffer.from([0x4d, 0x5a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),  // MZ (PE)
  }
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  fs.writeFileSync(filePath, headers[platform] ?? headers.linux)
}

/** Create a minimal fake node-pty source tree in a temp dir. */
function makeFakeNodePty(srcRoot, { prebuildPlatform, prebuildArch } = {}) {
  fs.mkdirSync(srcRoot, { recursive: true })
  fs.writeFileSync(join(srcRoot, 'package.json'), JSON.stringify({ name: 'node-pty', main: 'lib/index.js' }))
  fs.mkdirSync(join(srcRoot, 'lib'), { recursive: true })
  fs.writeFileSync(join(srcRoot, 'lib', 'index.js'), 'module.exports = {};')

  if (prebuildPlatform && prebuildArch) {
    const prebuildDir = join(srcRoot, 'prebuilds', `${prebuildPlatform}-${prebuildArch}`)
    makeFakeNode(join(prebuildDir, 'pty.node'), prebuildPlatform)
  }
}

// ─── classifyNativeBinary tests ─────────────────────────────────────

test('classifyNativeBinary detects ELF as linux', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0x7f, 0x45, 0x4c, 0x46, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), 'linux')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary detects Mach-O 64-bit BE as darwin', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0xfe, 0xed, 0xfa, 0xcf, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), 'darwin')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary detects Mach-O 64-bit LE (CIGAM_64) as darwin', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), 'darwin')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary detects Mach-O 32-bit BE as darwin', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0xfe, 0xed, 0xfa, 0xce, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), 'darwin')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary detects Mach-O 32-bit LE (CIGAM) as darwin', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0xce, 0xfa, 0xed, 0xfe, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), 'darwin')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary detects Fat/Universal BE (cafebabe) as darwin', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0xca, 0xfe, 0xba, 0xbe, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), 'darwin')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary detects Fat/Universal LE (bebafeca / FAT_CIGAM) as darwin', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0xbe, 0xba, 0xfe, 0xca, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), 'darwin')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary detects PE (MZ) as win32', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0x4d, 0x5a, 0x00, 0x00, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), 'win32')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary returns null for unrecognized magic', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const f = join(tmp, 'test.node')
    fs.writeFileSync(f, Buffer.from([0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))
    assert.equal(classifyNativeBinary(f), null)
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('classifyNativeBinary returns null for a missing file', () => {
  assert.equal(classifyNativeBinary('/nonexistent/path/to/thing.node'), null)
})

// ─── cross-target regression tests ──────────────────────────────────
//
// The core bug: stageNodePty receives { platform, arch } from
// electron-builder but unconditionally copies host build/Release, staging
// a host binary for a foreign target. These tests prove the fix:
//
// 1. A host build/Release must NOT be staged for a foreign platform.
// 2. A matching prebuild IS staged for a foreign target.
// 3. A foreign target with no prebuild throws (fail closed).
// 4. A host build/Release IS staged for a matching target.
// 5. Validation rejects a binary whose magic bytes don't match the target.

test('cross-target: host build/Release is NOT staged for a foreign platform', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'node-pty')
    const destRoot = join(tmp, 'dest')

    // Create a node-pty tree with ONLY a host build/Release (no prebuild).
    makeFakeNodePty(srcRoot)
    const buildReleaseDir = join(srcRoot, 'build', 'Release')
    makeFakeNode(join(buildReleaseDir, 'pty.node'), process.platform)

    // Request a foreign platform (different from the host).
    const foreignPlatform = process.platform === 'linux' ? 'darwin' : 'linux'

    assert.throws(
      () => stageNodePtyInto(srcRoot, destRoot, { platform: foreignPlatform, arch: 'x64' }),
      /cannot cross-compile/i
    )

    // build/Release must NOT have been copied to the dest tree.
    assert.equal(
      existsSync(join(destRoot, 'build', 'Release', 'pty.node')),
      false,
      'host build/Release .node must not be staged for a foreign target'
    )
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('cross-target: matching prebuild IS staged for a foreign target', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'node-pty')
    const destRoot = join(tmp, 'dest')

    // Host is (say) darwin. Request linux-x64, which has a prebuild.
    const foreignPlatform = process.platform === 'linux' ? 'darwin' : 'linux'
    makeFakeNodePty(srcRoot, { prebuildPlatform: foreignPlatform, prebuildArch: 'x64' })

    // Also create a host build/Release that should NOT be staged.
    makeFakeNode(join(srcRoot, 'build', 'Release', 'pty.node'), process.platform)

    stageNodePtyInto(srcRoot, destRoot, { platform: foreignPlatform, arch: 'x64' })

    // The foreign prebuild must be staged.
    const stagedPrebuild = join(destRoot, 'prebuilds', `${foreignPlatform}-x64`, 'pty.node')
    assert.equal(existsSync(stagedPrebuild), true, 'foreign prebuild must be staged')

    // The host build/Release must NOT be staged.
    assert.equal(
      existsSync(join(destRoot, 'build', 'Release', 'pty.node')),
      false,
      'host build/Release must not be staged for a foreign target'
    )
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('cross-target: foreign target with no prebuild throws (fail closed)', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'node-pty')
    const destRoot = join(tmp, 'dest')

    // Create a tree with a host build/Release but no foreign prebuild.
    makeFakeNodePty(srcRoot)
    makeFakeNode(join(srcRoot, 'build', 'Release', 'pty.node'), process.platform)

    const foreignPlatform = process.platform === 'linux' ? 'darwin' : 'linux'

    assert.throws(
      () => stageNodePtyInto(srcRoot, destRoot, { platform: foreignPlatform, arch: 'x64' }),
      /cannot cross-compile/i
    )
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('host-target: host build/Release IS staged for a matching target', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'node-pty')
    const destRoot = join(tmp, 'dest')

    makeFakeNodePty(srcRoot)
    makeFakeNode(join(srcRoot, 'build', 'Release', 'pty.node'), process.platform)

    stageNodePtyInto(srcRoot, destRoot, { platform: process.platform, arch: process.arch })

    assert.equal(
      existsSync(join(destRoot, 'build', 'Release', 'pty.node')),
      true,
      'host build/Release must be staged for a matching target'
    )
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('validation rejects a staged binary with the wrong platform magic', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'node-pty')
    const destRoot = join(tmp, 'dest')

    // Create a prebuild dir that claims to be linux-x64 but contains
    // a darwin (Mach-O) binary. This simulates the original bug where
    // a host binary ends up in a foreign target's prebuild slot.
    makeFakeNodePty(srcRoot, { prebuildPlatform: 'linux', prebuildArch: 'x64' })
    // Overwrite the prebuild .node with the WRONG platform magic.
    makeFakeNode(join(srcRoot, 'prebuilds', 'linux-x64', 'pty.node'), 'darwin')

    assert.throws(
      () => stageNodePtyInto(srcRoot, destRoot, { platform: 'linux', arch: 'x64' }),
      /platform mismatch/i
    )
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})
