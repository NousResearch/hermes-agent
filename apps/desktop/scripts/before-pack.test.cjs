const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { cleanStaleAppOutDir, staleBackupPath } = require('../scripts/before-pack.cjs')

test('cleanStaleAppOutDir renames a populated unpacked directory into nested backup', () => {
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

    const { removed, backedUp } = cleanStaleAppOutDir(appOutDir)

    assert.equal(removed, true)
    assert.equal(backedUp, true)
    // Original directory must be gone.
    assert.equal(fs.existsSync(appOutDir), false)
    // Backup must exist under .rebuild-backup/ with the original files.
    const backupDir = staleBackupPath(appOutDir)
    assert.equal(fs.existsSync(backupDir), true)
    assert.equal(fs.existsSync(path.join(backupDir, 'LICENSE.electron.txt')), true)
    assert.equal(fs.existsSync(path.join(backupDir, 'resources', 'app.asar')), true)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('cleanStaleAppOutDir replaces an existing backup with a fresh rename', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'linux-unpacked')
    const backupDir = staleBackupPath(appOutDir)

    // Create a stale backup from a previous failed build.
    fs.mkdirSync(backupDir, { recursive: true })
    fs.writeFileSync(path.join(backupDir, 'OLD'), 'stale', 'utf8')

    // Create the current build directory.
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'LICENSE.electron.txt'), 'new', 'utf8')

    const { removed, backedUp } = cleanStaleAppOutDir(appOutDir)

    assert.equal(removed, true)
    assert.equal(backedUp, true)
    assert.equal(fs.existsSync(appOutDir), false)
    // The old backup should have been replaced — the stale 'OLD' file must not survive.
    assert.equal(fs.existsSync(path.join(backupDir, 'OLD')), false)
    // The new backup should have the current build's files.
    assert.equal(fs.existsSync(path.join(backupDir, 'LICENSE.electron.txt')), true)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('cleanStaleAppOutDir is a no-op when the directory is absent', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const missing = path.join(tempRoot, 'does-not-exist')
    const { removed, backedUp } = cleanStaleAppOutDir(missing)
    assert.equal(removed, false)
    assert.equal(backedUp, false)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
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
  const { default: beforePack } = require('../scripts/before-pack.cjs')
  // A directory path that rename/rmSync can't handle (empty string)
  // simulates a partial failure; the contract is that the hook never rejects.
  await assert.doesNotReject(beforePack({ appOutDir: '', electronPlatformName: 'linux' }))
})
