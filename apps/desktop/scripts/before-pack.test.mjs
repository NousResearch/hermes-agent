/**
 * before-pack.test.mjs — tests for before-pack.mjs hooks.
 * Run:  node --test apps/desktop/scripts/before-pack.test.mjs
 */
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import assert from 'node:assert/strict'
import test from 'node:test'
import { cleanStaleAppOutDir, staleBackupPath } from './before-pack.mjs'

test('cleanStaleAppOutDir renames a populated unpacked directory into nested backup', () => {
  const tempRoot = mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'linux-unpacked')
    mkdirSync(appOutDir, { recursive: true })
    writeFileSync(path.join(appOutDir, 'LICENSE.electron.txt'), 'x', 'utf8')
    const { removed, backedUp } = cleanStaleAppOutDir(appOutDir)
    assert.equal(removed, true)
    assert.equal(backedUp, true)
    assert.equal(existsSync(appOutDir), false)
    const backupDir = staleBackupPath(appOutDir)
    assert.equal(existsSync(backupDir), true)
    assert.equal(existsSync(path.join(backupDir, 'LICENSE.electron.txt')), true)
  } finally { rmSync(tempRoot, { recursive: true, force: true }) }
})

test('cleanStaleAppOutDir preserves existing backup when appOutDir exists (retry-safe)', () => {
  const tempRoot = mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'linux-unpacked')
    const backupDir = staleBackupPath(appOutDir)
    mkdirSync(backupDir, { recursive: true })
    writeFileSync(path.join(backupDir, 'HERMES'), 'known-good', 'utf8')
    mkdirSync(appOutDir, { recursive: true })
    writeFileSync(path.join(appOutDir, 'partial-output'), 'from failed retry', 'utf8')
    const { removed, backedUp } = cleanStaleAppOutDir(appOutDir)
    assert.equal(removed, true)
    assert.equal(backedUp, true)
    assert.equal(existsSync(appOutDir), false)
    assert.equal(existsSync(path.join(backupDir, 'HERMES')), true)
    assert.equal(existsSync(path.join(backupDir, 'partial-output')), false)
  } finally { rmSync(tempRoot, { recursive: true, force: true }) }
})

test('cleanStaleAppOutDir is a no-op when the directory is absent', () => {
  const tempRoot = mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const { removed, backedUp } = cleanStaleAppOutDir(path.join(tempRoot, 'does-not-exist'))
    assert.equal(removed, false)
    assert.equal(backedUp, false)
  } finally { rmSync(tempRoot, { recursive: true, force: true }) }
})

test('cleanStaleAppOutDir ignores empty or invalid input', () => {
  for (const bad of ['', undefined, null, 42]) {
    const { removed, backedUp } = cleanStaleAppOutDir(bad)
    assert.equal(removed, false)
    assert.equal(backedUp, false)
  }
})

test('staleBackupPath nests under .rebuild-backup/', () => {
  assert.equal(
    staleBackupPath('/build/release/win-unpacked'),
    path.join('/build/release', '.rebuild-backup', 'win-unpacked')
  )
})

test('beforePack default export resolves even when cleanup throws', async () => {
  const { default: beforePack } = await import('./before-pack.mjs')
  await assert.doesNotReject(beforePack({ appOutDir: '', electronPlatformName: 'linux' }))
})
