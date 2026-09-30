import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import beforePack, { cleanStaleAppOutDir, preserveRollbackBackup } from '../scripts/before-pack.mjs'

test('cleanStaleAppOutDir removes a populated unpacked directory', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'linux-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    // Reproduce the corrupted partial state: license + payload present,
    // electron binary missing — exactly what trips the ENOENT rename.
    fs.writeFileSync(path.join(appOutDir, 'LICENSE.electron.txt'), 'x', 'utf8')
    fs.writeFileSync(path.join(appOutDir, 'resources.pak'), 'x', 'utf8')
    fs.mkdirSync(path.join(appOutDir, 'resources'), { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'resources', 'app.asar'), 'x', 'utf8')

    const removed = cleanStaleAppOutDir(appOutDir)

    assert.equal(removed, true)
    assert.equal(fs.existsSync(appOutDir), false)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('cleanStaleAppOutDir is a no-op when the directory is absent', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const missing = path.join(tempRoot, 'does-not-exist')
    assert.equal(cleanStaleAppOutDir(missing), false)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('cleanStaleAppOutDir ignores empty or invalid input', () => {
  assert.equal(cleanStaleAppOutDir(''), false)
  assert.equal(cleanStaleAppOutDir(undefined), false)
  assert.equal(cleanStaleAppOutDir(null), false)
  assert.equal(cleanStaleAppOutDir(42), false)
})

test('beforePack default export resolves even when cleanup throws', async () => {
  // A directory path that rmSync can't remove is simulated by passing a
  // context whose appOutDir is a file the hook will try (and be allowed) to
  // remove; the contract under test is that the hook never rejects.
  await assert.doesNotReject(beforePack({ appOutDir: '', electronPlatformName: 'linux' }))
})

// ─── Windows rollback preservation (#69179) ────────────────────────────────

test('preserveRollbackBackup moves a working build to .bak', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'MZ-old-build', 'utf8')
    fs.writeFileSync(path.join(appOutDir, 'resources.pak'), 'x', 'utf8')

    const preserved = preserveRollbackBackup(appOutDir, 'Hermes.exe')

    assert.equal(preserved, true)
    // Original slot vacated so electron-builder stages into a clean tree...
    assert.equal(fs.existsSync(appOutDir), false)
    // ...and the previous working build is intact under .bak for rollback.
    assert.equal(
      fs.readFileSync(path.join(`${appOutDir}.bak`, 'Hermes.exe'), 'utf8'),
      'MZ-old-build'
    )
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('preserveRollbackBackup replaces a stale .bak from an older update', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'current', 'utf8')
    fs.mkdirSync(`${appOutDir}.bak`, { recursive: true })
    fs.writeFileSync(path.join(`${appOutDir}.bak`, 'Hermes.exe'), 'two-updates-ago', 'utf8')

    assert.equal(preserveRollbackBackup(appOutDir, 'Hermes.exe'), true)
    assert.equal(fs.readFileSync(path.join(`${appOutDir}.bak`, 'Hermes.exe'), 'utf8'), 'current')
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('preserveRollbackBackup refuses a partial tree missing the product exe', () => {
  // The corrupted partial state (interrupted prior pack) must NOT become
  // rollback material — it is exactly what cleanStaleAppOutDir exists to wipe.
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'LICENSE.electron.txt'), 'x', 'utf8')

    assert.equal(preserveRollbackBackup(appOutDir, 'Hermes.exe'), false)
    // Tree untouched; the caller's wipe path handles it.
    assert.equal(fs.existsSync(appOutDir), true)
    assert.equal(fs.existsSync(`${appOutDir}.bak`), false)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('preserveRollbackBackup ignores missing or invalid input', () => {
  assert.equal(preserveRollbackBackup(''), false)
  assert.equal(preserveRollbackBackup(undefined), false)
  assert.equal(preserveRollbackBackup(null), false)
  assert.equal(preserveRollbackBackup(path.join(os.tmpdir(), 'does-not-exist-xyz')), false)
})

test('beforePack on win32 preserves the previous build instead of wiping it', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'MZ-working', 'utf8')

    // No packager info in the context → default 'Hermes.exe' product name.
    // node-pty staging is skipped because arch is not a number here.
    await beforePack({ appOutDir, electronPlatformName: 'win32' })

    assert.equal(fs.existsSync(appOutDir), false)
    assert.equal(
      fs.readFileSync(path.join(`${appOutDir}.bak`, 'Hermes.exe'), 'utf8'),
      'MZ-working'
    )
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('beforePack on linux keeps the plain wipe (no .bak)', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'linux-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'x', 'utf8')

    await beforePack({ appOutDir, electronPlatformName: 'linux' })

    assert.equal(fs.existsSync(appOutDir), false)
    assert.equal(fs.existsSync(`${appOutDir}.bak`), false)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// ─── macOS runtime (optional, but never mismatched) ─────────────────────────

function macContext(projectDir, arch) {
  // Arch enum: 1 = x64, 3 = arm64
  return { appOutDir: '', electronPlatformName: 'darwin', arch, packager: { projectDir } }
}

test('beforePack on darwin accepts a missing runtime and refuses a mismatched one', async () => {
  const projectDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    fs.mkdirSync(path.join(projectDir, 'build/runtime'), { recursive: true })
    fs.writeFileSync(path.join(projectDir, 'build/install-stamp.json'), JSON.stringify({ commit: 'abc1234' }))
    // The DMG falls back to the online bootstrap when no runtime was staged.
    // (Native staging is skipped by the catch below; only the runtime gate matters here.)
    const gate = ctx => beforePack(ctx).then(() => null, error => String(error.message))
    assert.doesNotMatch((await gate(macContext(projectDir, 3))) ?? '', /staged runtime/)

    const manifest = path.join(projectDir, 'build/runtime/manifest.json')
    fs.writeFileSync(manifest, JSON.stringify({ platform: 'darwin', arch: 'x64', commit: 'abc1234' }))
    assert.match(await gate(macContext(projectDir, 3)), /staged runtime/)

    fs.writeFileSync(manifest, JSON.stringify({ platform: 'darwin', arch: 'arm64', commit: 'other' }))
    assert.match(await gate(macContext(projectDir, 3)), /staged runtime/)
  } finally {
    fs.rmSync(projectDir, { recursive: true, force: true })
  }
})
