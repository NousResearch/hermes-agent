import assert from 'node:assert/strict'
import fs, { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { test } from 'vitest'

import {
  findHalfInstalledGetWindowsDir,
  installGetWindowsNativeBinding,
  stageGetWindows,
  stageGetWindowsInto,
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
    linux: Buffer.from([0x7f, 0x45, 0x4c, 0x46, 0x00, 0x00, 0x00, 0x00]), // ELF
    // On x64/arm64 Darwin, Mach-O binaries are stored little-endian on disk
    // (MH_CIGAM_64 = cffaedfe). This is the form node-pty's prebuilds ship in.
    darwin: Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 0x00, 0x00, 0x00, 0x00]), // Mach-O 64-bit LE (CIGAM_64)
    win32: Buffer.from([0x4d, 0x5a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]) // MZ (PE)
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
      'exports.resolveHelper = function (helperPath) {',
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
      assert.equal(stagedUnixTerminal.resolveHelper(nodeModulesUnpackedHelper), nodeModulesUnpackedHelper)
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

// ─── non-ASCII path regression tests ───────────────────────────────
//
// Node 24's native fs.cpSync/fs.rmSync mishandle non-ASCII Windows paths
// (accented user-profile dirs): recursive copies fail or crash, overwrite
// copies fail with a bogus errno-0 unlink error, and rmSync silently
// deletes nothing. Staging therefore uses libuv-backed primitives only.
// This test stages into an accented src/dest tree — twice, so the
// restage exercises the delete-then-recopy path — to keep it that way.

test('non-ASCII paths: staging into an accented tree works and restages cleanly', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const accented = join(tmp, 'áccentéd ünïcødé-pŕöfílé')
    const srcRoot = join(accented, 'node-pty')
    const destRoot = join(accented, 'dest')

    makeFakeNodePty(srcRoot, {
      prebuildPlatform: process.platform,
      prebuildArch: process.arch
    })
    // Windows prebuilds ship a nested conpty/ payload — cover the
    // recursive-directory branch on every platform.
    const conptyDir = join(srcRoot, 'prebuilds', `${process.platform}-${process.arch}`, 'conpty')
    fs.mkdirSync(conptyDir, { recursive: true })
    fs.writeFileSync(join(conptyDir, 'conpty.dll'), 'fake dll')
    fs.writeFileSync(join(conptyDir, 'OpenConsole.exe'), 'fake exe')

    for (let pass = 1; pass <= 2; pass++) {
      stageNodePtyInto(srcRoot, destRoot, { platform: process.platform, arch: process.arch })

      const stagedPrebuild = join(destRoot, 'prebuilds', `${process.platform}-${process.arch}`)
      assert.equal(existsSync(join(destRoot, 'package.json')), true, `pass ${pass}: package.json staged`)
      assert.equal(existsSync(join(destRoot, 'lib', 'index.js')), true, `pass ${pass}: lib staged`)
      assert.equal(existsSync(join(stagedPrebuild, 'pty.node')), true, `pass ${pass}: prebuild staged`)
      assert.equal(existsSync(join(stagedPrebuild, 'conpty', 'conpty.dll')), true, `pass ${pass}: conpty dir staged`)
      assert.equal(
        existsSync(join(stagedPrebuild, 'conpty', 'OpenConsole.exe')),
        true,
        `pass ${pass}: conpty exe staged`
      )
    }
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

    assert.throws(() => stageNodePtyInto(srcRoot, destRoot, { platform: 'linux', arch: 'x64' }), /platform mismatch/i)
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

// ─── stageGetWindowsInto tests ──────────────────────────────────────

/** Create a minimal fake get-windows source tree in a temp dir. */
function makeFakeGetWindows(srcRoot, { version = '9.3.0', bindings = [] } = {}) {
  fs.mkdirSync(join(srcRoot, 'lib'), { recursive: true })
  fs.writeFileSync(join(srcRoot, 'package.json'), JSON.stringify({ name: 'get-windows', version, main: 'index.js' }))
  fs.writeFileSync(join(srcRoot, 'index.js'), 'export {};')
  fs.writeFileSync(join(srcRoot, 'lib', 'windows.js'), '// upstream pre-gyp loader')
  fs.writeFileSync(join(srcRoot, 'main'), '#!/bin/sh\n')

  for (const { dir, platform } of bindings) {
    makeFakeNode(join(srcRoot, 'lib', 'binding', dir, 'node-get-windows.node'), platform)
  }
}

test('win32 staging skips the darwin binding the tarball bundles on every platform', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')

    // The shape every real Windows build host has: the darwin binding
    // committed into the published tarball PLUS the win32 binding
    // node-pre-gyp downloaded at install time.
    makeFakeGetWindows(srcRoot, {
      bindings: [
        { dir: 'napi-9-darwin-unknown-arm64', platform: 'darwin' },
        { dir: 'napi-9-win32-unknown-x64', platform: 'win32' }
      ]
    })

    stageGetWindowsInto(srcRoot, destRoot, { platform: 'win32', arch: 'x64' })

    assert.ok(existsSync(join(destRoot, 'lib', 'binding', 'napi-9-win32-unknown-x64', 'node-get-windows.node')))
    assert.ok(!existsSync(join(destRoot, 'lib', 'binding', 'napi-9-darwin-unknown-arm64')))
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('win32 staging rejects a binding dir that claims win32 but holds a foreign binary', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')

    makeFakeGetWindows(srcRoot, {
      bindings: [{ dir: 'napi-9-win32-unknown-x64', platform: 'darwin' }]
    })

    assert.throws(
      () => stageGetWindowsInto(srcRoot, destRoot, { platform: 'win32', arch: 'x64' }),
      /expected win32, got darwin/
    )
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('win32-x64 staging degrades to the fail-soft JS surface when only foreign bindings exist', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  const warnings = []
  const origWarn = console.warn
  console.warn = msg => warnings.push(String(msg))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')

    // Half-extracted install (#90829): the package resolves, but lib/binding
    // holds only the darwin dir the tarball bundles and the installer is a no-op.
    makeFakeGetWindows(srcRoot, {
      bindings: [{ dir: 'napi-9-darwin-unknown-arm64', platform: 'darwin' }]
    })

    assert.equal(
      stageGetWindowsInto(srcRoot, destRoot, { platform: 'win32', arch: 'x64', install: () => {} }),
      destRoot
    )
    assert.ok(existsSync(join(destRoot, 'lib', 'windows.js')))
    assert.ok(!existsSync(join(destRoot, 'lib', 'binding')))
    assert.equal(warnings.length, 1)
    assert.match(warnings[0], /native installer produced no win32-x64 binding/)
  } finally {
    console.warn = origWarn
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('win32-arm64 staging omits incompatible bindings and keeps the fail-soft JS surface', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')

    makeFakeGetWindows(srcRoot, {
      bindings: [
        { dir: 'napi-9-darwin-unknown-arm64', platform: 'darwin' },
        { dir: 'napi-9-win32-unknown-x64', platform: 'win32' }
      ]
    })

    stageGetWindowsInto(srcRoot, destRoot, { platform: 'win32', arch: 'arm64' })

    assert.ok(existsSync(join(destRoot, 'lib', 'windows.js')))
    assert.ok(!existsSync(join(destRoot, 'lib', 'binding')))
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('win32 staging self-heals through the native installer when the binding is missing', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')

    // The bricked state a blocked install script leaves behind: the package is
    // present, lib/binding was never populated by node-pre-gyp.
    makeFakeGetWindows(srcRoot, { bindings: [] })

    let calls = 0
    const install = () => {
      calls += 1
      makeFakeNode(join(srcRoot, 'lib', 'binding', 'napi-9-win32-unknown-x64', 'node-get-windows.node'), 'win32')
    }

    stageGetWindowsInto(srcRoot, destRoot, { platform: 'win32', arch: 'x64', install })

    assert.equal(calls, 1)
    assert.ok(existsSync(join(destRoot, 'lib', 'binding', 'napi-9-win32-unknown-x64', 'node-get-windows.node')))
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('darwin staging degrades (not staged) when the helper binary is missing', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  const warnings = []
  const origWarn = console.warn
  console.warn = msg => warnings.push(String(msg))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')

    makeFakeGetWindows(srcRoot)
    fs.rmSync(join(srcRoot, 'main'))

    assert.equal(stageGetWindowsInto(srcRoot, destRoot, { platform: 'darwin' }), undefined)
    assert.ok(!existsSync(destRoot), 'a helper-less module must not ship half-staged')
    assert.equal(warnings.length, 1)
    assert.match(warnings[0], /missing its macOS helper binary/)
  } finally {
    console.warn = origWarn
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('get-windows native install invokes node-pre-gyp directly from the package root', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const installer = join(srcRoot, 'node_modules', '@mapbox', 'node-pre-gyp', 'bin', 'node-pre-gyp')
    fs.mkdirSync(path.dirname(installer), { recursive: true })
    fs.writeFileSync(
      join(srcRoot, 'node_modules', '@mapbox', 'node-pre-gyp', 'package.json'),
      JSON.stringify({ name: '@mapbox/node-pre-gyp', version: '1.0.11' })
    )
    fs.writeFileSync(installer, '')

    const calls = []
    installGetWindowsNativeBinding(srcRoot, {
      spawn: (command, args, options) => {
        calls.push({ command, args, options })
        return { status: 0 }
      }
    })

    assert.deepEqual(calls, [
      {
        command: process.execPath,
        args: [fs.realpathSync(installer), 'install', '--fallback-to-build'],
        options: { cwd: srcRoot, stdio: 'inherit' }
      }
    ])
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('get-windows native install surfaces node-pre-gyp failure', () => {
  assert.throws(
    () =>
      installGetWindowsNativeBinding('C:\\fake\\get-windows', {
        resolveInstaller: () => 'C:\\fake\\node-pre-gyp',
        spawn: () => ({ status: 1 })
      }),
    /native installer exited with 1/
  )
})

test('staging refuses a get-windows version the lib/windows.js rewrite was not verified against', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')

    makeFakeGetWindows(srcRoot, { version: '9.4.0' })

    assert.throws(() => stageGetWindowsInto(srcRoot, destRoot, { platform: 'darwin' }), /verified against 9\.3\.0/)
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('darwin staging ships the Swift helper executable and the rewritten windows.js', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')

    makeFakeGetWindows(srcRoot)

    stageGetWindowsInto(srcRoot, destRoot, { platform: 'darwin' })

    assert.deepEqual(fs.readFileSync(join(destRoot, 'main')), fs.readFileSync(join(srcRoot, 'main')))
    const staged = fs.readFileSync(join(destRoot, 'lib', 'windows.js'), 'utf8')
    assert.match(staged, /Rewritten by stage-native-deps\.mjs/)
    assert.ok(!staged.includes('node-pre-gyp'), 'pre-gyp loader must not survive staging')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

// ─── stageGetWindows (optionalDependency gate) ──────────────────────
//
// get-windows is an optionalDependency: npm omits it when its install script
// fails (no prebuilt for Linux / Windows ARM64) and a Windows in-place update
// can leave it half-extracted when a running Desktop holds files open
// (#90829). Either way only read_window_below is lost, so staging degrades
// with a warning on every platform instead of failing the whole build.

test.runIf(process.platform !== 'win32')('darwin staging preserves executable mode on a POSIX filesystem', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'hermes-stage-'))
  try {
    const srcRoot = join(tmp, 'get-windows')
    const destRoot = join(tmp, 'dest')
    makeFakeGetWindows(srcRoot)
    stageGetWindowsInto(srcRoot, destRoot, { platform: 'darwin' })
    assert.equal(fs.statSync(join(destRoot, 'main')).mode & 0o777, 0o755)
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})

test('staging degrades (never throws) when get-windows is absent on every platform', () => {
  const warnings = []
  const origWarn = console.warn
  console.warn = msg => warnings.push(String(msg))
  try {
    for (const [platform, arch] of [
      ['linux', 'x64'],
      ['darwin', 'arm64'],
      ['win32', 'arm64'],
      ['win32', 'x64']
    ]) {
      assert.equal(
        stageGetWindows({ platform, arch, resolveRoot: () => null, findHalfInstalledDir: () => null }),
        undefined,
        `${platform}-${arch} must degrade`
      )
    }
  } finally {
    console.warn = origWarn
  }
  assert.equal(warnings.length, 4)
  assert.ok(warnings.every(w => w.includes('read_window_below will be unavailable')))
  assert.ok(
    warnings.every(w => !w.includes('half-extracted')),
    'no repair hint without a stale dir'
  )
})

test('a half-installed get-windows dir is found and named in a repair hint', () => {
  const tmp = fs.mkdtempSync(join(os.tmpdir(), 'half-installed-'))
  try {
    // Reporter's state: node_modules/get-windows exists (binding extracted) but
    // package.json never was, so require.resolve fails while the dir is there.
    const app = join(tmp, 'apps', 'desktop')
    const stale = join(tmp, 'node_modules', 'get-windows', 'lib', 'binding')
    fs.mkdirSync(app, { recursive: true })
    fs.mkdirSync(stale, { recursive: true })
    assert.equal(findHalfInstalledGetWindowsDir(app), join(tmp, 'node_modules', 'get-windows'))

    const warnings = []
    const origWarn = console.warn
    console.warn = msg => warnings.push(String(msg))
    try {
      stageGetWindows({
        platform: 'win32',
        arch: 'x64',
        resolveRoot: () => null,
        findHalfInstalledDir: () => findHalfInstalledGetWindowsDir(app)
      })
    } finally {
      console.warn = origWarn
    }
    assert.equal(warnings.length, 1)
    assert.match(warnings[0], /hermes desktop --force-build/)
    assert.ok(warnings[0].includes(join(tmp, 'node_modules', 'get-windows')))

    fs.rmSync(join(tmp, 'node_modules'), { recursive: true, force: true })
    assert.equal(findHalfInstalledGetWindowsDir(app), null)
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
})
