import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import { writeFixture, readFixture } from './pe-test-fixture.mjs'

import {
  ROLLBACK_ACQUISITION_STATUS,
  preserveRollbackBackup
} from './before-pack.mjs'
import {
  isWindowsPeExecutable,
  readRollbackSession,
  settleDesktopPack
} from './desktop-pack-transaction.mjs'

const PE_AMD64 = 0x8664

function tempRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pack-transaction-'))
}

function writePe(filePath, marker = 0x42, { machine = PE_AMD64, truncateTo } = {}) {
  const payload = Buffer.alloc(0x400)
  payload[0] = 0x4d
  payload[1] = 0x5a
  payload.writeUInt32LE(0x80, 0x3c)
  payload[0x80] = 0x50
  payload[0x81] = 0x45
  payload[0x82] = 0x00
  payload[0x83] = 0x00
  payload.writeUInt16LE(machine, 0x84)
  payload.writeUInt16LE(1, 0x86)
  payload.writeUInt16LE(0, 0x94)
  payload.writeUInt16LE(0x0002, 0x96)
  payload.writeUInt32LE(0x200, 0xa8)
  payload.writeUInt32LE(0x200, 0xac)
  payload.fill(marker, 0x200)
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  fs.writeFileSync(filePath, truncateTo === undefined ? payload : payload.subarray(0, truncateTo))
}

test('PE verification rejects prefix-only and section-truncated executables', () => {
  const root = tempRoot()
  try {
    const valid = path.join(root, 'valid.exe')
    const truncated = path.join(root, 'truncated.exe')
    const prefixOnly = path.join(root, 'prefix-only.exe')
    writePe(valid)
    writePe(truncated, 0x42, { truncateTo: 0x300 })
    fs.writeFileSync(prefixOnly, 'MZ-not-a-complete-pe', 'utf8')

    assert.equal(isWindowsPeExecutable(valid), true)
    assert.equal(isWindowsPeExecutable(truncated), false)
    assert.equal(isWindowsPeExecutable(prefixOnly), false)
    assert.equal(isWindowsPeExecutable(path.join(root, 'missing.exe')), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('failed builder restores the last valid packaged app over partial output', () => {
  const root = tempRoot()
  try {
    const releaseDir = path.join(root, 'release')
    const appOutDir = path.join(releaseDir, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    writePe(path.join(backupDir, 'Hermes.exe'), 0x11)
    fs.writeFileSync(`${backupDir}.session`, 'session-a\n', 'utf8')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'partial.txt'), 'partial', 'utf8')

    const result = settleDesktopPack({
      releaseDir,
      builderSucceeded: false,
      sessionId: 'session-a'
    })

    assert.equal(result.ok, true)
    assert.deepEqual(result.restored, [appOutDir])
    assert.equal(fs.existsSync(backupDir), false)
    assert.equal(fs.existsSync(`${backupDir}.session`), false)
    assert.equal(isWindowsPeExecutable(path.join(appOutDir, 'Hermes.exe')), true)
    assert.equal(fs.existsSync(path.join(appOutDir, 'partial.txt')), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('successful builder retains rollback for the canonical launchability gate', () => {
  const root = tempRoot()
  try {
    const releaseDir = path.join(root, 'release')
    const appOutDir = path.join(releaseDir, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    writePe(path.join(backupDir, 'Hermes.exe'), 0x11)
    writePe(path.join(appOutDir, 'Hermes.exe'), 0x22)
    fs.writeFileSync(`${backupDir}.session`, 'session-a\n', 'utf8')

    const result = settleDesktopPack({
      releaseDir,
      builderSucceeded: true,
      sessionId: 'session-a'
    })

    assert.equal(result.ok, true)
    assert.deepEqual(result.retained, [backupDir])
    assert.deepEqual(result.discarded, [])
    assert.equal(fs.existsSync(backupDir), true)
    assert.equal(fs.existsSync(`${backupDir}.session`), false)
    assert.equal(isWindowsPeExecutable(path.join(appOutDir, 'Hermes.exe')), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('false builder success restores previous app and becomes a failure', () => {
  const root = tempRoot()
  try {
    const releaseDir = path.join(root, 'release')
    const appOutDir = path.join(releaseDir, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    writePe(path.join(backupDir, 'Hermes.exe'), 0x11)
    fs.writeFileSync(`${backupDir}.session`, 'session-a\n', 'utf8')
    writePe(path.join(appOutDir, 'Hermes.exe'), 0x22, { truncateTo: 0x300 })

    const result = settleDesktopPack({
      releaseDir,
      builderSucceeded: true,
      sessionId: 'session-a'
    })

    assert.equal(result.ok, false)
    assert.deepEqual(result.restored, [appOutDir])
    assert.match(result.failures[0].reason, /exited successfully.*structurally incomplete/)
    assert.equal(isWindowsPeExecutable(path.join(appOutDir, 'Hermes.exe')), true)
    assert.equal(fs.existsSync(backupDir), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})


test('rollback promotion failure restores the failed output path and preserves the backup', () => {
  const root = tempRoot()
  try {
    const releaseDir = path.join(root, 'release')
    const appOutDir = path.join(releaseDir, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    const failedDir = `${appOutDir}.failed`
    writePe(path.join(backupDir, 'Hermes.exe'), 0x11)
    fs.writeFileSync(`${backupDir}.session`, 'session-a\n', 'utf8')
    fs.mkdirSync(appOutDir, { recursive: true })
    fs.writeFileSync(path.join(appOutDir, 'partial.txt'), 'failed-output', 'utf8')

    const result = settleDesktopPack({
      releaseDir,
      builderSucceeded: false,
      sessionId: 'session-a',
      restoreOperations: {
        rename(source, target) {
          if (source === backupDir && target === appOutDir) {
            const error = new Error('simulated rollback promotion failure')
            error.code = 'EPERM'
            throw error
          }
          fs.renameSync(source, target)
        }
      }
    })

    assert.equal(result.ok, false)
    assert.deepEqual(result.restored, [])
    assert.match(result.failures[0].reason, /simulated rollback promotion failure/)
    assert.equal(
      fs.readFileSync(path.join(appOutDir, 'partial.txt'), 'utf8'),
      'failed-output'
    )
    assert.equal(isWindowsPeExecutable(path.join(backupDir, 'Hermes.exe')), true)
    assert.equal(fs.readFileSync(`${backupDir}.session`, 'utf8'), 'session-a\n')
    assert.equal(fs.existsSync(failedDir), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('invalid apparent success cannot hide behind another generation marker', () => {
  const root = tempRoot()
  try {
    const releaseDir = path.join(root, 'release')
    const appOutDir = path.join(releaseDir, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    writePe(path.join(backupDir, 'Hermes.exe'), 0x11)
    fs.writeFileSync(`${backupDir}.session`, 'other-session\n', 'utf8')
    writePe(path.join(appOutDir, 'Hermes.exe'), 0x22, { truncateTo: 0x300 })

    const result = settleDesktopPack({
      releaseDir,
      builderSucceeded: true,
      sessionId: 'current-session'
    })

    assert.equal(result.ok, false)
    assert.deepEqual(result.restored, [])
    assert.match(result.failures[0].reason, /belongs to generation other-session/)
    assert.equal(fs.existsSync(backupDir), true)
    assert.equal(fs.existsSync(appOutDir), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('same electron-builder session cannot overwrite the original rollback generation', () => {
  const root = tempRoot()
  try {
    const appOutDir = path.join(root, 'release', 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    fs.mkdirSync(appOutDir, { recursive: true })
    writeFixture(path.join(appOutDir, 'Hermes.exe'), 'original-generation', 'utf8')

    assert.equal(
      preserveRollbackBackup(appOutDir, 'Hermes.exe', 'session-a').status,
      ROLLBACK_ACQUISITION_STATUS.PRESERVED
    )
    assert.equal(readRollbackSession(backupDir), 'session-a')

    fs.mkdirSync(appOutDir, { recursive: true })
    writeFixture(path.join(appOutDir, 'Hermes.exe'), 'first-target-output', 'utf8')
    assert.equal(
      preserveRollbackBackup(appOutDir, 'Hermes.exe', 'session-a').status,
      ROLLBACK_ACQUISITION_STATUS.PRESERVED
    )

    assert.equal(fs.existsSync(appOutDir), false)
    assert.equal(
      readFixture(path.join(backupDir, 'Hermes.exe'), 'utf8'),
      'original-generation'
    )
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('a new builder session replaces stale rollback material with the current good app', () => {
  const root = tempRoot()
  try {
    const appOutDir = path.join(root, 'release', 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    fs.mkdirSync(backupDir, { recursive: true })
    writeFixture(path.join(backupDir, 'Hermes.exe'), 'older-generation', 'utf8')
    writeFixture(`${backupDir}.session`, 'stale-session\n', 'utf8')
    fs.mkdirSync(appOutDir, { recursive: true })
    writeFixture(path.join(appOutDir, 'Hermes.exe'), 'current-generation', 'utf8')

    // Only a completed, validated builder releases the unfinished marker.
    assert.equal(settleDesktopPack({ releaseDir: path.dirname(appOutDir),
      builderSucceeded: true, sessionId: 'stale-session' }).ok, true)

    assert.equal(
      preserveRollbackBackup(appOutDir, 'Hermes.exe', 'new-session').status,
      ROLLBACK_ACQUISITION_STATUS.PRESERVED
    )
    assert.equal(readRollbackSession(backupDir), 'new-session')
    assert.equal(
      readFixture(path.join(backupDir, 'Hermes.exe'), 'utf8'),
      'current-generation'
    )
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('pack settlement ignores rollback material owned by another generation', () => {
  const root = tempRoot()
  try {
    const releaseDir = path.join(root, 'release')
    const appOutDir = path.join(releaseDir, 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    writePe(path.join(backupDir, 'Hermes.exe'), 0x11)
    writeFixture(`${backupDir}.session`, 'other-session\n', 'utf8')

    const result = settleDesktopPack({
      releaseDir,
      builderSucceeded: false,
      sessionId: 'current-session'
    })

    assert.equal(result.ok, true)
    assert.deepEqual(result.restored, [])
    assert.equal(fs.existsSync(backupDir), true)
    assert.equal(fs.existsSync(appOutDir), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('retry adopts a valid interrupted rollback into its current generation', () => {
  const root = tempRoot()
  try {
    const appOutDir = path.join(root, 'release', 'win-unpacked')
    const backupDir = `${appOutDir}.bak`
    fs.mkdirSync(appOutDir, { recursive: true })
    writeFixture(path.join(appOutDir, 'partial.txt'), 'partial', 'utf8')
    fs.mkdirSync(backupDir, { recursive: true })
    writeFixture(path.join(backupDir, 'Hermes.exe'), 'last-good', 'utf8')
    writeFixture(`${backupDir}.session`, 'interrupted-session\n', 'utf8')

    assert.equal(
      preserveRollbackBackup(appOutDir, 'Hermes.exe', 'retry-session').status,
      ROLLBACK_ACQUISITION_STATUS.SAFE_TO_CLEAN
    )
    assert.equal(readRollbackSession(backupDir), 'retry-session')
    assert.equal(fs.existsSync(appOutDir), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('a structurally complete executable for the wrong target rolls back', () => {
  const root = tempRoot()
  try {
    const output = path.join(root, 'win-unpacked')
    writePe(path.join(output, 'Hermes.exe'), 0x11)
    preserveRollbackBackup(output, 'Hermes.exe', 'arch-session')
    writePe(path.join(output, 'Hermes.exe'), 0x22, { machine: 0xaa64 })
    const result = settleDesktopPack({
      targets: [{ appOutDir: output, productExeName: 'Hermes.exe', expectedMachine: PE_AMD64 }],
      sessionId: 'arch-session', builderSucceeded: true
    })
    assert.equal(result.ok, false)
    assert.equal(fs.readFileSync(path.join(output, 'Hermes.exe'))[0x200], 0x11)
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})

test('an interrupted complete wrong-architecture candidate cannot replace its prior backup', () => {
  const root = tempRoot()
  try {
    const output = path.join(root, 'win-unpacked')
    const backup = `${output}.bak`
    writePe(path.join(backup, 'Hermes.exe'), 0x11)
    fs.writeFileSync(`${backup}.session`, 'interrupted-session')
    writePe(path.join(output, 'Hermes.exe'), 0x22, { machine: 0xaa64 })
    preserveRollbackBackup(output, 'Hermes.exe', 'retry-session')
    assert.equal(fs.readFileSync(path.join(backup, 'Hermes.exe'))[0x200], 0x11)
    const result = settleDesktopPack({ releaseDir: root, builderSucceeded: false, sessionId: 'retry-session' })
    assert.equal(result.ok, true)
    assert.equal(fs.readFileSync(path.join(output, 'Hermes.exe'))[0x200], 0x11)
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})
