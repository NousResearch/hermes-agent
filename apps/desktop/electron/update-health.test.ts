import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, test } from 'vitest'

import { createUpdateHealthManager } from './update-health'

const tempDirs: string[] = []

function setup(
  version: string,
  options: { hashFile?: (filePath: string) => Promise<string>; maxAttempts?: number } = {}
) {
  const cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-health-'))
  tempDirs.push(cacheDir)

  const manager = createUpdateHealthManager({
    cacheDir,
    currentVersion: version,
    enabled: true,
    hashFile: options.hashFile,
    maxAttempts: options.maxAttempts ?? 2,
    now: () => 1_234
  })

  return { cacheDir, manager }
}

function retainInstaller(cacheDir: string, version: string, content = `installer-${version}`) {
  const installerPath = path.join(cacheDir, 'installers', `${version}.exe`)
  fs.mkdirSync(path.dirname(installerPath), { recursive: true })
  fs.writeFileSync(installerPath, content)

  return installerPath
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

describe('managed update startup health', () => {
  test('promotes the first healthy packaged install to hash-verified known-good', async () => {
    const { cacheDir, manager } = setup('0.17.100')
    retainInstaller(cacheDir, '0.17.100')

    assert.deepEqual(manager.beginStartup(), {
      action: 'await-health',
      attempts: 1,
      version: '0.17.100'
    })

    const result = await manager.confirmHealthy()

    assert.equal(result.ok, true)
    assert.equal(result.promoted, true)
    assert.equal(manager.getState().knownGood?.version, '0.17.100')
    assert.match(manager.getState().knownGood?.sha512 ?? '', /^[a-f0-9]{128}$/)
    assert.equal(manager.getState().candidate, undefined)
  })

  test('requests rollback after two unconfirmed candidate launches', async () => {
    const first = setup('0.17.100')
    retainInstaller(first.cacheDir, '0.17.100')
    first.manager.beginStartup()
    await first.manager.confirmHealthy()

    retainInstaller(first.cacheDir, '0.17.101')

    const candidate = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 2,
      now: () => 2_000
    })

    assert.equal(candidate.beginStartup().action, 'await-health')

    const retry = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 2,
      now: () => 3_000
    })

    const decision = retry.beginStartup()

    assert.deepEqual(decision, {
      action: 'rollback',
      failedVersion: '0.17.101',
      installerPath: path.join(first.cacheDir, 'installers', '0.17.100.exe'),
      reason: 'attempt-limit',
      targetVersion: '0.17.100'
    })
    assert.equal((await retry.verifyRollbackInstaller(decision)).ok, true)
  })

  test('does not count a clean early user quit as a failed startup', async () => {
    const first = setup('0.17.100')
    retainInstaller(first.cacheDir, '0.17.100')
    first.manager.beginStartup()
    await first.manager.confirmHealthy()

    retainInstaller(first.cacheDir, '0.17.101')

    const candidate = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 2
    })

    candidate.beginStartup()
    candidate.recordCleanExit()

    const retry = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 2
    })

    assert.deepEqual(retry.beginStartup(), {
      action: 'await-health',
      attempts: 1,
      version: '0.17.101'
    })
  })

  test('refuses rollback when the retained known-good installer was tampered with', async () => {
    const first = setup('0.17.100')
    const installerPath = retainInstaller(first.cacheDir, '0.17.100')
    first.manager.beginStartup()
    await first.manager.confirmHealthy()
    fs.appendFileSync(installerPath, '-tampered')

    retainInstaller(first.cacheDir, '0.17.101')

    const candidate = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 1
    })

    const decision = candidate.beginStartup()

    assert.equal(decision.action, 'rollback')
    assert.deepEqual(await candidate.verifyRollbackInstaller(decision), {
      error: 'Known-good installer integrity verification failed.',
      ok: false
    })
  })

  test('recognizes a completed rollback without treating the restored version as a candidate', async () => {
    const first = setup('0.17.100')
    retainInstaller(first.cacheDir, '0.17.100')
    first.manager.beginStartup()
    await first.manager.confirmHealthy()

    retainInstaller(first.cacheDir, '0.17.101')

    const candidate = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 1
    })

    const decision = candidate.beginStartup()
    assert.equal(decision.action, 'rollback')
    candidate.recordRollbackStarted(decision)

    const restored = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.100',
      enabled: true
    })

    assert.deepEqual(restored.beginStartup(), {
      action: 'known-good',
      version: '0.17.100'
    })
    assert.equal(restored.getState().candidate, undefined)
    assert.equal(restored.getState().rollback, undefined)
    assert.equal(restored.isVersionRejected('0.17.101'), true)
    assert.equal(restored.isVersionRejected('0.17.102'), false)

    const rejectedRetry = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true
    })

    assert.deepEqual(rejectedRetry.beginStartup(), {
      action: 'rollback',
      failedVersion: '0.17.101',
      installerPath: path.join(first.cacheDir, 'installers', '0.17.100.exe'),
      reason: 'rejected-version',
      targetVersion: '0.17.100'
    })
  })

  test('persists a terminal state after the rollback handoff limit is reached', async () => {
    const first = setup('0.17.100')
    retainInstaller(first.cacheDir, '0.17.100')
    first.manager.beginStartup()
    await first.manager.confirmHealthy()
    retainInstaller(first.cacheDir, '0.17.101')

    const candidate = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 1,
      maxRollbackHandoffs: 1,
      now: () => 2_000
    })

    const rollback = candidate.beginStartup()
    assert.equal(rollback.action, 'rollback')
    candidate.recordRollbackStarted(rollback)

    const exhausted = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 1,
      maxRollbackHandoffs: 1,
      now: () => 3_000
    })

    assert.deepEqual(exhausted.beginStartup(), {
      action: 'rollback-exhausted',
      failedVersion: '0.17.101',
      handoffs: 1,
      targetVersion: '0.17.100'
    })
    assert.deepEqual(await exhausted.confirmHealthy(), {
      error: 'Automatic startup rollback handoff limit is exhausted.',
      ok: false
    })
    assert.equal(exhausted.timeoutDecision(), null)
    assert.equal(exhausted.getState().rollback?.handoffs, 1)
  })

  test('serializes candidate promotion before a concurrent timeout can claim rollback', async () => {
    const first = setup('0.17.100')
    retainInstaller(first.cacheDir, '0.17.100')
    first.manager.beginStartup()
    await first.manager.confirmHealthy()
    retainInstaller(first.cacheDir, '0.17.101')

    let releaseHash!: () => void
    let signalHashStarted!: () => void
    const hashReleased = new Promise<void>(resolve => void (releaseHash = resolve))
    const hashStarted = new Promise<void>(resolve => void (signalHashStarted = resolve))

    const candidate = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      hashFile: async () => {
        signalHashStarted()
        await hashReleased

        return 'a'.repeat(128)
      },
      maxAttempts: 2
    })

    candidate.beginStartup()
    const confirmation = candidate.confirmHealthy()
    await hashStarted

    assert.equal(candidate.timeoutDecision(), null)
    releaseHash()
    assert.deepEqual(await confirmation, { ok: true, promoted: true })
    assert.equal(candidate.getState().knownGood?.version, '0.17.101')
  })

  test('releases failed confirmation so the timeout can still roll back', async () => {
    const first = setup('0.17.100')
    retainInstaller(first.cacheDir, '0.17.100')
    first.manager.beginStartup()
    await first.manager.confirmHealthy()
    retainInstaller(first.cacheDir, '0.17.101')

    const candidate = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      hashFile: async () => {
        throw new Error('disk read failed')
      }
    })

    candidate.beginStartup()
    assert.equal((await candidate.confirmHealthy()).ok, false)
    assert.equal(candidate.isAwaitingHealth(), true)
    assert.equal(candidate.timeoutDecision()?.action, 'rollback')
  })

  test('refuses candidate promotion after the timeout has claimed rollback', async () => {
    const first = setup('0.17.100')
    retainInstaller(first.cacheDir, '0.17.100')
    first.manager.beginStartup()
    await first.manager.confirmHealthy()
    retainInstaller(first.cacheDir, '0.17.101')

    const candidate = createUpdateHealthManager({
      cacheDir: first.cacheDir,
      currentVersion: '0.17.101',
      enabled: true,
      maxAttempts: 2
    })

    candidate.beginStartup()
    assert.equal(candidate.timeoutDecision()?.action, 'rollback')
    assert.deepEqual(await candidate.confirmHealthy(), {
      error: 'Startup rollback is already in progress.',
      ok: false
    })
    assert.equal(candidate.getState().knownGood?.version, '0.17.100')
  })
})
