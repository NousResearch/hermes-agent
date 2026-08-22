import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import beforePack, {
  ROLLBACK_ACQUISITION_STATUS,
  cleanStaleAppOutDir,
  preserveRollbackBackup
} from '../scripts/before-pack.mjs'

const { BLOCKED, PRESERVED, SAFE_TO_CLEAN } = ROLLBACK_ACQUISITION_STATUS

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

test('beforePack default export resolves for an empty best-effort cleanup target', async () => {
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

    const acquisition = preserveRollbackBackup(appOutDir, 'Hermes.exe')

    assert.equal(acquisition.status, PRESERVED)
    assert.equal(fs.existsSync(appOutDir), false)
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

    const acquisition = preserveRollbackBackup(appOutDir, 'Hermes.exe')

    assert.equal(acquisition.status, PRESERVED)
    assert.equal(fs.readFileSync(path.join(`${appOutDir}.bak`, 'Hermes.exe'), 'utf8'), 'current')
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('preserveRollbackBackup marks a partial tree safe to clean', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'LICENSE.electron.txt'), 'x', 'utf8')

    const acquisition = preserveRollbackBackup(appOutDir, 'Hermes.exe')

    assert.equal(acquisition.status, SAFE_TO_CLEAN)
    assert.equal(fs.existsSync(appOutDir), true)
    assert.equal(fs.existsSync(`${appOutDir}.bak`), false)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('preserveRollbackBackup marks missing or invalid input safe to clean', () => {
  assert.equal(preserveRollbackBackup('').status, SAFE_TO_CLEAN)
  assert.equal(preserveRollbackBackup(undefined).status, SAFE_TO_CLEAN)
  assert.equal(preserveRollbackBackup(null).status, SAFE_TO_CLEAN)
  assert.equal(
    preserveRollbackBackup(path.join(os.tmpdir(), 'does-not-exist-xyz')).status,
    SAFE_TO_CLEAN
  )
})

test('beforePack on win32 preserves the previous build instead of wiping it', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'MZ-working', 'utf8')

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

test('beforePack fails closed when a stale rollback marker cannot be retired', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    const markerPath = `${backupDir}.session`
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'MZ-current-working', 'utf8')
    fs.mkdirSync(backupDir, { recursive: true })
    fs.writeFileSync(path.join(backupDir, 'Hermes.exe'), 'MZ-older-working', 'utf8')
    fs.writeFileSync(markerPath, 'older-session\n', 'utf8')

    await assert.rejects(
      beforePack(
        { appOutDir, electronPlatformName: 'win32' },
        {
          rollbackSessionId: 'new-session',
          rollbackOperations: {
            clearRollbackSession() {
              const error = new Error('simulated locked rollback session marker')
              error.code = 'EPERM'
              throw error
            }
          }
        }
      ),
      error => {
        assert.match(error.message, /refusing destructive Windows package replacement/)
        assert.match(error.message, /rollback-slot-retirement-failed/)
        assert.match(error.message, /simulated locked rollback session marker/)
        return true
      }
    )

    assert.equal(fs.readFileSync(path.join(appOutDir, 'Hermes.exe'), 'utf8'), 'MZ-current-working')
    assert.equal(fs.readFileSync(path.join(backupDir, 'Hermes.exe'), 'utf8'), 'MZ-older-working')
    assert.equal(fs.readFileSync(markerPath, 'utf8'), 'older-session\n')
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('beforePack leaves the current app untouched when marker creation fails', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    let renameCalled = false
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'MZ-current-working', 'utf8')
    fs.writeFileSync(path.join(appOutDir, 'resources.pak'), 'current-resources', 'utf8')

    await assert.rejects(
      beforePack(
        { appOutDir, electronPlatformName: 'win32' },
        {
          rollbackSessionId: 'write-failure-session',
          rollbackOperations: {
            writeRollbackSession() {
              const error = new Error('simulated rollback session write failure')
              error.code = 'EACCES'
              throw error
            },
            renameSync() {
              renameCalled = true
              throw new Error('rename must not run after marker failure')
            }
          }
        }
      ),
      error => {
        assert.match(error.message, /refusing destructive Windows package replacement/)
        assert.match(error.message, /rollback-session-write-failed/)
        assert.match(error.message, /simulated rollback session write failure/)
        return true
      }
    )

    assert.equal(renameCalled, false)
    assert.equal(fs.readFileSync(path.join(appOutDir, 'Hermes.exe'), 'utf8'), 'MZ-current-working')
    assert.equal(
      fs.readFileSync(path.join(appOutDir, 'resources.pak'), 'utf8'),
      'current-resources'
    )
    assert.equal(fs.existsSync(backupDir), false)
    assert.equal(fs.existsSync(`${backupDir}.session`), false)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('beforePack clears the staged marker and leaves the current app when rename fails', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'MZ-current-working', 'utf8')

    await assert.rejects(
      beforePack(
        { appOutDir, electronPlatformName: 'win32' },
        {
          rollbackSessionId: 'rename-failure-session',
          rollbackOperations: {
            renameSync() {
              const error = new Error('simulated package rename failure')
              error.code = 'EPERM'
              throw error
            }
          }
        }
      ),
      error => {
        assert.match(error.message, /current-package-preservation-failed/)
        assert.match(error.message, /simulated package rename failure/)
        return true
      }
    )

    assert.equal(fs.readFileSync(path.join(appOutDir, 'Hermes.exe'), 'utf8'), 'MZ-current-working')
    assert.equal(fs.existsSync(backupDir), false)
    assert.equal(fs.existsSync(`${backupDir}.session`), false)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('preserveRollbackBackup reports blocked when rollback acquisition fails', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-before-pack-'))
  try {
    const appOutDir = path.join(tempRoot, 'win-unpacked')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'Hermes.exe'), 'MZ-current-working', 'utf8')

    const acquisition = preserveRollbackBackup(
      appOutDir,
      'Hermes.exe',
      'blocked-session',
      {
        clearRollbackSession() {
          throw new Error('cannot retire rollback slot')
        }
      }
    )

    assert.equal(acquisition.status, BLOCKED)
    assert.equal(acquisition.reason, 'rollback-slot-retirement-failed')
    assert.equal(fs.readFileSync(path.join(appOutDir, 'Hermes.exe'), 'utf8'), 'MZ-current-working')
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
