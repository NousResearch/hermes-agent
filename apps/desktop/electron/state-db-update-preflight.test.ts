import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, test, vi } from 'vitest'

import type { PreUpdateBackupPolicy } from './pre-update-backup-policy'
import {
  backupDeadlineMs,
  inspectStateDbHeader,
  planEmergencyStateDbBackup,
  requiredFreeBytes,
  runStateDbUpdatePreflight,
  selectEmergencyBackupPrune,
  type StateDbPreflightDeps
} from './state-db-update-preflight'

const tempDirs: string[] = []

const QUICK_POLICY: PreUpdateBackupPolicy = {
  backupKeep: 5,
  mode: 'quick',
  quickKeep: 1,
  quickMaxFileSize: 1024
}

function tempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-state-preflight-'))
  tempDirs.push(dir)

  return dir
}

function validHeaderFile(filePath: string, size = 101): void {
  const content = Buffer.alloc(size)
  Buffer.from('SQLite format 3\0').copy(content)
  fs.writeFileSync(filePath, content)
}

function validInspection(size = 101) {
  return { headerHex: Buffer.from('SQLite format 3\0').toString('hex'), size, status: 'valid' as const }
}

function successfulDeps(overrides: Partial<StateDbPreflightDeps> = {}): StateDbPreflightDeps {
  return {
    availableBytes: () => 10_000_000_000n,
    createBackup: async (_source, destination) => {
      fs.writeFileSync(destination, 'snapshot')

      return { durationMs: 5, size: 8, verification: 'integrity-check' }
    },
    inspectFootprint: () => ({ estimatedBackupBytes: 512, logicalBytes: 512, mainBytes: 101, walBytes: 0 }),
    inspectHeader: () => validInspection(),
    listFiles: directory => fs.readdirSync(directory),
    nonce: () => 'nonce',
    now: () => new Date('2026-09-03T12:00:00.000Z'),
    removeFile: filePath => fs.rmSync(filePath, { force: true }),
    ...overrides
  }
}

afterEach(() => {
  vi.restoreAllMocks()

  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

describe('inspectStateDbHeader', () => {
  test('allows only ENOENT as a missing database', () => {
    assert.deepEqual(inspectStateDbHeader(path.join(tempDir(), 'missing.db')), { status: 'missing' })
  })

  test('requires a regular file, sufficient size, and an exact 16-byte header', () => {
    const dir = tempDir()
    const valid = path.join(dir, 'valid.db')
    validHeaderFile(valid)

    assert.equal(inspectStateDbHeader(valid).status, 'valid')
    assert.throws(() => inspectStateDbHeader(dir), /regular file/)

    const short = path.join(dir, 'short.db')
    fs.writeFileSync(short, 'SQLite format 3\0')
    assert.throws(() => inspectStateDbHeader(short), /too small/)

    const wrong = path.join(dir, 'wrong.db')
    fs.writeFileSync(wrong, Buffer.alloc(101, 0xff))
    assert.throws(() => inspectStateDbHeader(wrong), /invalid SQLite header/)
  })

  test('fails closed on stat and open errors other than ENOENT', () => {
    const source = path.join(tempDir(), 'state.db')
    validHeaderFile(source)
    const denied = Object.assign(new Error('denied'), { code: 'EACCES' })

    vi.spyOn(fs, 'statSync').mockImplementationOnce(() => {
      throw denied
    })
    assert.throws(() => inspectStateDbHeader(source), /denied/)

    vi.spyOn(fs, 'openSync').mockImplementationOnce(() => {
      throw denied
    })
    assert.throws(() => inspectStateDbHeader(source), /denied/)
  })

  test('rejects a short read and always closes the descriptor after read failure', () => {
    const source = path.join(tempDir(), 'state.db')
    validHeaderFile(source)

    vi.spyOn(fs, 'readSync').mockReturnValueOnce(15)
    assert.throws(() => inspectStateDbHeader(source), /header read was short/)

    const close = vi.spyOn(fs, 'closeSync')
    vi.spyOn(fs, 'readSync').mockImplementationOnce(() => {
      throw new Error('read failed')
    })
    assert.throws(() => inspectStateDbHeader(source), /read failed/)
    assert.equal(close.mock.calls.length, 1)
  })
})

describe('backup planning', () => {
  test('off always skips before an artifact plan exists', () => {
    assert.deepEqual(planEmergencyStateDbBackup({ ...QUICK_POLICY, mode: 'off' }, 99_999), {
      reason: 'config-disabled',
      status: 'skipped'
    })
  })

  test('quick backs up at the canonical boundary and skips boundary plus one', () => {
    assert.deepEqual(planEmergencyStateDbBackup(QUICK_POLICY, 1024), {
      keep: 1,
      status: 'backup'
    })
    assert.deepEqual(planEmergencyStateDbBackup(QUICK_POLICY, 1025), {
      reason: 'quick-size-cap',
      status: 'skipped'
    })
  })

  test('full uses the normalized configured retention above the quick boundary', () => {
    assert.deepEqual(planEmergencyStateDbBackup({ ...QUICK_POLICY, backupKeep: 4, mode: 'full' }, 10_000), {
      keep: 4,
      status: 'backup'
    })
  })

  test('free-space bound and deadline are conservative and bounded', () => {
    assert.equal(requiredFreeBytes(100n), 1024n * 1024n * 1024n + 100n)
    assert.equal(requiredFreeBytes(20n * 1024n * 1024n * 1024n), 22n * 1024n * 1024n * 1024n)
    assert.equal(backupDeadlineMs(1), 120_000)
    assert.equal(backupDeadlineMs(40 * 1024 * 1024 * 1024), 30 * 60_000)
  })

  test('retention counts the newly published artifact', () => {
    const current = 'state.db.pre-update-emergency-2026-09-03T12-00-00-000Z-nonce.bak'

    const files = [
      current,
      'state.db.pre-update-emergency-2026-09-02T12-00-00-000Z-a.bak',
      'state.db.pre-update-emergency-2026-09-01T12-00-00-000Z-b.bak',
      'state.db.pre-update-emergency-2026-08-31T12-00-00-000Z-c.bak',
      'state.db.pre-update-emergency-junk.bak.partial',
      'state.db'
    ]

    assert.deepEqual(selectEmergencyBackupPrune(files, current, 1), files.slice(1, 4))
    assert.deepEqual(selectEmergencyBackupPrune(files, current, 3), [files[3]])
  })
})

describe('runStateDbUpdatePreflight', () => {
  test('missing state.db exits before policy/Python discovery or any artifact work', async () => {
    const calls: string[] = []

    const deps = successfulDeps({
      inspectHeader: () => ({ status: 'missing' })
    })

    const result = await runStateDbUpdatePreflight(
      {
        hermesHome: tempDir(),
        policy: async () => {
          calls.push('policy')
          throw new Error('must not run')
        },
        rememberLog: () => {}
      },
      deps
    )

    assert.deepEqual(result, { status: 'missing' })
    assert.deepEqual(calls, [])
  })

  test('config off with an oversized database performs no footprint, space, enumeration, write, or delete', async () => {
    const calls: string[] = []

    const deps = successfulDeps({
      availableBytes: () => {
        calls.push('space')

        return 0n
      },
      createBackup: async () => {
        calls.push('backup')
        throw new Error('must not run')
      },
      inspectFootprint: () => {
        calls.push('footprint')
        throw new Error('must not run')
      },
      inspectHeader: () => validInspection(16 * 1024 * 1024 * 1024),
      listFiles: () => {
        calls.push('list')

        return []
      },
      removeFile: () => calls.push('delete')
    })

    const logs: string[] = []

    const result = await runStateDbUpdatePreflight(
      { hermesHome: tempDir(), policy: { ...QUICK_POLICY, mode: 'off' }, rememberLog: line => logs.push(line) },
      deps
    )

    assert.deepEqual(result, { reason: 'config-disabled', status: 'skipped' })
    assert.deepEqual(calls, [])
    assert.ok(logs.some(line => line.includes('disabled by updates.pre_update_backup')))
  })

  test('validates the header before resolving policy and fails closed on resolver errors', async () => {
    const calls: string[] = []

    const deps = successfulDeps({
      availableBytes: () => {
        calls.push('space')

        return 0n
      },
      inspectFootprint: () => {
        calls.push('footprint')
        throw new Error('must not run')
      }
    })

    await assert.rejects(
      runStateDbUpdatePreflight(
        {
          hermesHome: tempDir(),
          policy: async () => {
            calls.push('policy')
            throw new Error('policy process timed out')
          },
          rememberLog: () => {}
        },
        deps
      ),
      /policy process timed out/
    )

    assert.deepEqual(calls, ['policy'])
  })

  test('does not resolve policy when the existing database header is invalid', async () => {
    let policyCalls = 0

    const deps = successfulDeps({
      inspectHeader: () => {
        throw new Error('invalid SQLite header')
      }
    })

    await assert.rejects(
      runStateDbUpdatePreflight(
        {
          hermesHome: tempDir(),
          policy: async () => {
            policyCalls += 1

            return QUICK_POLICY
          },
          rememberLog: () => {}
        },
        deps
      ),
      /invalid SQLite header/
    )
    assert.equal(policyCalls, 0)
  })

  test('quick skips a large logical/WAL footprint before free-space or artifact work', async () => {
    const calls: string[] = []

    const deps = successfulDeps({
      availableBytes: () => {
        calls.push('space')

        return 0n
      },
      createBackup: async () => {
        calls.push('backup')
        throw new Error('must not run')
      },
      inspectFootprint: () => ({ estimatedBackupBytes: 1025, logicalBytes: 1000, mainBytes: 101, walBytes: 1024 }),
      listFiles: () => {
        calls.push('list')

        return []
      }
    })

    const result = await runStateDbUpdatePreflight(
      { hermesHome: tempDir(), policy: QUICK_POLICY, rememberLog: () => {} },
      deps
    )

    assert.deepEqual(result, { reason: 'quick-size-cap', status: 'skipped' })
    assert.deepEqual(calls, [])
  })

  test('insufficient or unmeasurable free space aborts before enumeration, writes, or deletes', async () => {
    for (const availableBytes of [
      () => 0n,
      () => {
        throw new Error('probe failed')
      }
    ]) {
      const calls: string[] = []

      const deps = successfulDeps({
        availableBytes,
        createBackup: async () => {
          calls.push('backup')
          throw new Error('must not run')
        },
        listFiles: () => {
          calls.push('list')

          return []
        },
        removeFile: () => calls.push('delete')
      })

      await assert.rejects(
        runStateDbUpdatePreflight(
          { hermesHome: tempDir(), policy: { ...QUICK_POLICY, mode: 'full' }, rememberLog: () => {} },
          deps
        ),
        /free space/
      )
      assert.deepEqual(calls, [])
    }
  })

  test('free-space boundary rejects one byte below and allows the exact required amount', async () => {
    const required = requiredFreeBytes(512n)
    let backupCalls = 0

    await assert.rejects(
      runStateDbUpdatePreflight(
        { hermesHome: tempDir(), policy: { ...QUICK_POLICY, mode: 'full' }, rememberLog: () => {} },
        successfulDeps({
          availableBytes: () => required - 1n,
          createBackup: async () => {
            backupCalls += 1
            throw new Error('must not run')
          }
        })
      ),
      /insufficient free space/
    )
    assert.equal(backupCalls, 0)

    const result = await runStateDbUpdatePreflight(
      { hermesHome: tempDir(), policy: { ...QUICK_POLICY, mode: 'full' }, rememberLog: () => {} },
      successfulDeps({ availableBytes: () => required })
    )

    assert.equal(result.status, 'backed-up')
  })

  test('cleans stale partials only after the space gate, publishes, then applies total-count retention', async () => {
    const home = tempDir()
    const old = 'state.db.pre-update-emergency-2026-09-01T00-00-00-000Z-old.bak'
    const stale = 'state.db.pre-update-emergency-2026-09-02T00-00-00-000Z-stale.bak.partial'
    fs.writeFileSync(path.join(home, old), 'old')
    fs.writeFileSync(path.join(home, stale), 'partial')
    const events: string[] = []

    const deps = successfulDeps({
      availableBytes: () => {
        events.push('space')

        return 10_000_000_000n
      },
      createBackup: async (_source, destination) => {
        events.push('backup')
        assert.equal(fs.existsSync(path.join(home, stale)), false)
        fs.writeFileSync(destination, 'snapshot')

        return { durationMs: 5, size: 8, verification: 'integrity-check' }
      },
      removeFile: filePath => {
        events.push(`delete:${path.basename(filePath)}`)
        fs.rmSync(filePath, { force: true })
      }
    })

    const result = await runStateDbUpdatePreflight(
      { hermesHome: home, policy: QUICK_POLICY, rememberLog: () => {} },
      deps
    )

    assert.equal(result.status, 'backed-up')
    assert.equal(events[0], 'space')
    assert.equal(events[1], `delete:${stale}`)
    assert.equal(events[2], 'backup')
    assert.ok(events.includes(`delete:${old}`))
  })

  test('backup or verification failure performs no published-backup retention', async () => {
    const home = tempDir()
    const old = 'state.db.pre-update-emergency-2026-09-01T00-00-00-000Z-old.bak'
    fs.writeFileSync(path.join(home, old), 'preserve me')
    const deleted: string[] = []

    const deps = successfulDeps({
      createBackup: async () => {
        throw new Error('verification failed')
      },
      removeFile: filePath => {
        deleted.push(path.basename(filePath))
        fs.rmSync(filePath, { force: true })
      }
    })

    await assert.rejects(
      runStateDbUpdatePreflight(
        { hermesHome: home, policy: { ...QUICK_POLICY, mode: 'full' }, rememberLog: () => {} },
        deps
      ),
      /verification failed/
    )
    assert.equal(fs.readFileSync(path.join(home, old), 'utf8'), 'preserve me')
    assert.deepEqual(deleted, [])
  })
})
