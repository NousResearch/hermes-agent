import assert from 'node:assert/strict'
import fs, { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { test } from 'vitest'

import {
  stageNodePtyInto,
  classifyNativeBinary,
  copyDirRecursive
} from '../scripts/stage-native-deps.mjs'

const { join } = path
const here = path.dirname(fileURLToPath(import.meta.url))

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

function makeFakeUnixTerminal(srcRoot) {
  fs.writeFileSync(
    join(srcRoot, 'lib', 'unixTerminal.js'),
    [
      "exports.resolveHelper = function (helperPath) {",
      "  helperPath = helperPath.replace('app.asar', 'app.asar.unpacked');",
      "  helperPath = helperPath.replace('node_modules.asar', 'node_modules.asar.unpacked');",
      '  return helperPath;',
      '};'
    ].join('\n')
  )
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

test.skipIf(process.platform === 'win32')(
  'host-target: staged node-pty resolves an already-unpacked helper and preserves executable helpers',
  async () => {
    const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
    try {
      const srcRoot = join(tmp, 'node-pty')
      const destRoot = join(tmp, 'dest')
      const prebuildDir = join(srcRoot, 'prebuilds', `${process.platform}-${process.arch}`)
      const buildReleaseDir = join(srcRoot, 'build', 'Release')

      makeFakeNodePty(srcRoot, {
        prebuildPlatform: process.platform,
        prebuildArch: process.arch
      })
      makeFakeUnixTerminal(srcRoot)
      makeFakeNode(join(buildReleaseDir, 'pty.node'), process.platform)
      fs.writeFileSync(join(prebuildDir, 'spawn-helper'), 'prebuild helper')
      fs.writeFileSync(join(buildReleaseDir, 'spawn-helper'), 'build helper')
      fs.chmodSync(join(prebuildDir, 'spawn-helper'), 0o644)
      fs.chmodSync(join(buildReleaseDir, 'spawn-helper'), 0o644)

      stageNodePtyInto(srcRoot, destRoot, { platform: process.platform, arch: process.arch })

      const stagedUnixTerminalUrl = pathToFileURL(join(destRoot, 'lib', 'unixTerminal.js'))
      stagedUnixTerminalUrl.searchParams.set('t', String(Date.now()))
      const stagedUnixTerminal = await import(stagedUnixTerminalUrl.href)
      const unpackedHelper = join(
        tmp,
        'Hermes.app',
        'Contents',
        'Resources',
        'app.asar.unpacked',
        'dist',
        'node_modules',
        'node-pty',
        'prebuilds',
        `${process.platform}-${process.arch}`,
        'spawn-helper'
      )
      const nodeModulesUnpackedHelper = unpackedHelper.replace(
        `${path.sep}node_modules${path.sep}`,
        `${path.sep}node_modules.asar.unpacked${path.sep}`
      )

      assert.equal(stagedUnixTerminal.resolveHelper(unpackedHelper), unpackedHelper)
      assert.equal(
        stagedUnixTerminal.resolveHelper(nodeModulesUnpackedHelper),
        nodeModulesUnpackedHelper
      )
      assert.equal(
        fs.statSync(join(destRoot, 'prebuilds', `${process.platform}-${process.arch}`, 'spawn-helper')).mode & 0o777,
        0o755
      )
      assert.equal(fs.statSync(join(destRoot, 'build', 'Release', 'spawn-helper')).mode & 0o777, 0o755)
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  }
)

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

// ─── copyDirRecursive tests (Windows 0xC0000409 crash regression) ────
//
// Node 22.x's native recursive cpSync (cpSyncCopyDir) hard-crashes on
// Windows while copying node-pty's conpty payload — a silent 0xC0000409
// that killed the packaged desktop build at the stage step. Directory
// copies must go through the manual per-file walk instead.

/**
 * Returns true if any `cpSync(...)` call in `source` passes a `recursive`
 * option. Walks each call's argument list tracking paren depth, so it catches
 * a call whose `{ recursive: true }` is split across lines — and is not fooled
 * by a `recursive` that belongs to a later mkdirSync/rmSync. Comments are
 * stripped first so the doc comment naming the avoided API doesn't match.
 */
function usesRecursiveCpSync(source) {
  const code = source
    .replace(/\/\*[\s\S]*?\*\//g, '') // block comments
    .replace(/^\s*\/\/.*$/gm, '') // line comments
  const CALL = 'cpSync('
  for (let start = code.indexOf(CALL); start !== -1; start = code.indexOf(CALL, start + 1)) {
    let depth = 0
    let i = start + CALL.length - 1 // points at the opening '('
    for (; i < code.length; i++) {
      if (code[i] === '(') depth++
      else if (code[i] === ')' && --depth === 0) break
    }
    const args = code.slice(start + CALL.length, i)
    if (/\brecursive\b/.test(args)) return true
  }
  return false
}

test('copyDirRecursive reproduces a nested tree with exact bytes', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const src = join(tmp, 'src')
    const dest = join(tmp, 'dest')
    // Mirror node-pty's payload shape: top-level files plus a nested conpty/
    // dir holding a binary blob (the .dll/.exe that trip Windows cpSync).
    fs.mkdirSync(join(src, 'conpty'), { recursive: true })
    fs.writeFileSync(join(src, 'pty.node'), Buffer.from([0, 1, 2, 253, 254, 255]))
    fs.writeFileSync(join(src, 'index.js'), 'module.exports = {}\n', 'utf8')
    fs.writeFileSync(join(src, 'conpty', 'conpty.dll'), Buffer.from([77, 90, 0, 255]))
    fs.writeFileSync(join(src, 'conpty', 'OpenConsole.exe'), Buffer.from([77, 90, 144, 0]))

    copyDirRecursive(src, dest)

    for (const rel of ['pty.node', 'index.js', 'conpty/conpty.dll', 'conpty/OpenConsole.exe']) {
      assert.ok(existsSync(join(dest, rel)), `missing ${rel}`)
      assert.deepEqual(
        fs.readFileSync(join(dest, rel)),
        fs.readFileSync(join(src, rel)),
        `bytes differ for ${rel}`
      )
    }
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('copyDirRecursive creates the destination even for an empty source', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const src = join(tmp, 'empty-src')
    const dest = join(tmp, 'empty-dest')
    fs.mkdirSync(src, { recursive: true })
    copyDirRecursive(src, dest)
    assert.equal(existsSync(dest), true)
    assert.deepEqual(fs.readdirSync(dest), [])
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('stage-native-deps uses no recursive cpSync (Windows 0xC0000409 regression)', () => {
  // Guard against anyone reintroducing `cpSync(.., { recursive: true })`.
  const source = fs.readFileSync(join(here, 'stage-native-deps.mjs'), 'utf8')
  assert.equal(
    usesRecursiveCpSync(source),
    false,
    'stage-native-deps.mjs must not call cpSync with { recursive: true }'
  )
})

test('recursive-cpSync guard is newline-tolerant and precise', () => {
  // A multi-line recursive call must still be caught (the guard's whole point).
  assert.equal(
    usesRecursiveCpSync('cpSync(\n  join(a, b),\n  join(c, d),\n  { recursive: true }\n)\n'),
    true,
    'must catch a cpSync whose { recursive: true } is on a later line'
  )
  // A non-recursive cpSync followed by an unrelated recursive mkdirSync must
  // NOT trip the guard (no false positive from scanning past the call).
  assert.equal(
    usesRecursiveCpSync('cpSync(join(a, b), join(c, d))\nmkdirSync(dest, { recursive: true })\n'),
    false,
    'must not flag a recursive option that belongs to a later call'
  )
})
