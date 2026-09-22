import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  JournalCorruptionError,
  JournalError,
  ManagedRolloutJournal,
  type JournalEventInput,
  type JournalFs,
  type JournalSnapshot,
  type UnresolvedFence
} from './managed-rollout-journal'

const ROLLOUT_A = '11111111-1111-4111-8111-111111111111'
const ROLLOUT_B = '22222222-2222-4222-8222-222222222222'
const ROLLOUT_C = '33333333-3333-4333-8333-333333333333'
const INSTALL_A = 'install-a'

function snapshot(id: string = ROLLOUT_A, overrides: Partial<JournalSnapshot> = {}): JournalSnapshot {
  return {
    schemaVersion: 1,
    id,
    revision: 0,
    createdAt: '2026-09-21T00:00:00.000Z',
    updatedAt: '2026-09-21T00:00:00.000Z',
    phase: 'running',
    finishedAt: null,
    archivedAt: null,
    attempts: [],
    eventCount: 0,
    ...overrides
  }
}

function event(kind: string, installId: string | null = null): JournalEventInput {
  return {
    kind,
    actor: 'system',
    installId,
    reason: null,
    evidenceDigest: null
  }
}

function fence(
  rolloutId: string = ROLLOUT_A,
  installId: string = INSTALL_A,
  overrides: Partial<UnresolvedFence> = {}
): UnresolvedFence {
  return {
    key: `${rolloutId}:${installId}`,
    rolloutId,
    installId,
    correlationId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
    reason: 'authorized launch has no conclusive settlement',
    recordedAt: '2026-09-21T00:00:00.000Z',
    ...overrides
  }
}

function withTempDirectory(run: (directory: string) => void): void {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-managed-rollout-journal-'))

  try {
    run(directory)
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

function realFs(overrides: Partial<JournalFs> = {}): JournalFs {
  return {
    mkdirSync: (filePath, options) => fs.mkdirSync(filePath, options),
    readdirSync: filePath => fs.readdirSync(filePath, { encoding: 'utf8' }),
    lstatSync: filePath => fs.lstatSync(filePath),
    readFileSync: filePath => fs.readFileSync(filePath, 'utf8'),
    writeFileSync: (filePath, data, options) => fs.writeFileSync(filePath, data, options),
    renameSync: (from, to) => fs.renameSync(from, to),
    unlinkSync: filePath => fs.unlinkSync(filePath),
    ...overrides
  }
}

function journal(
  directory: string,
  overrides: Omit<ConstructorParameters<typeof ManagedRolloutJournal>[0], 'directory'> = {}
): ManagedRolloutJournal {
  return new ManagedRolloutJournal({ directory, ...overrides })
}

test('writes a record with CAS revision and reopens it in a fresh instance', () => {
  withTempDirectory(directory => {
    const first = journal(directory)
    const created = first.create(snapshot(), { events: [event('created')] })

    assert.equal(created.revision, 1)
    assert.deepEqual(first.read(ROLLOUT_A).snapshot.revision, 1)
    assert.equal(first.events(ROLLOUT_A, { limit: 10 }).items[0].sequence, 1)

    const updated = first.record({
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-update-1',
      payload: { action: 'pause', expectedRevision: 1 },
      snapshot: snapshot(ROLLOUT_A, { phase: 'paused' }),
      events: [event('paused')]
    })

    assert.equal(updated.revision, 2)
    assert.equal(first.read(ROLLOUT_A).snapshot.phase, 'paused')

    const reopened = journal(directory)
    assert.equal(reopened.read(ROLLOUT_A).snapshot.revision, 2)
    assert.equal(reopened.read(ROLLOUT_A).snapshot.phase, 'paused')
    assert.deepEqual(
      reopened.events(ROLLOUT_A, { limit: 10 }).items.map(item => item.kind),
      ['created', 'paused']
    )
  })
})

test('a failed atomic write leaves the last valid record and does not reserve the request', () => {
  withTempDirectory(directory => {
    let failWrites = false
    const injected = realFs({
      writeFileSync: (filePath, data, options) => {
        if (failWrites && String(filePath).includes('.tmp-')) {
          const error = new Error('No space left on device') as NodeJS.ErrnoException
          error.code = 'ENOSPC'
          throw error
        }
        fs.writeFileSync(filePath, data, options)
      }
    })
    const instance = journal(directory, { fs: injected })
    instance.create(snapshot(), { events: [event('created')] })
    failWrites = true

    assert.throws(
      () =>
        instance.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-disk-full',
          payload: { action: 'pause' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'paused' }),
          events: [event('paused')]
        }),
      (error: unknown) => error instanceof JournalError && error.code === 'write-failed'
    )

    assert.equal(instance.read(ROLLOUT_A).snapshot.revision, 1)
    assert.equal(instance.events(ROLLOUT_A, { limit: 10 }).items.length, 1)

    failWrites = false
    const retry = instance.record({
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-disk-full',
      payload: { action: 'pause' },
      snapshot: snapshot(ROLLOUT_A, { phase: 'paused' }),
      events: [event('paused')]
    })
    assert.equal(retry.revision, 2)
  })
})

test('a durability acknowledgement failure leaves the prior revision and fence state intact', () => {
  withTempDirectory(directory => {
    let failSync = false
    const injected = realFs({
      syncFileSync: filePath => {
        if (failSync && String(filePath).includes('.tmp-')) throw new Error('flush failed')
      }
    })
    const instance = journal(directory, { fs: injected })
    instance.create(snapshot())
    failSync = true

    assert.throws(
      () =>
        instance.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-flush-failure',
          payload: { action: 'pause' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'paused' })
        }),
      (error: unknown) => error instanceof JournalError && error.code === 'write-failed'
    )
    assert.equal(instance.read(ROLLOUT_A).snapshot.revision, 1)
  })
})

test('a failed final rename leaves the destination unchanged', () => {
  withTempDirectory(directory => {
    let failRename = false
    const recordPath = path.join(directory, `${ROLLOUT_A}.json`)
    const injected = realFs({
      renameSync: (from, to) => {
        if (failRename && to === recordPath) {
          const error = new Error('destination is locked') as NodeJS.ErrnoException
          error.code = 'EPERM'
          throw error
        }
        fs.renameSync(from, to)
      }
    })
    const instance = journal(directory, { fs: injected })
    instance.create(snapshot(), { events: [event('created')] })
    failRename = true

    assert.throws(
      () =>
        instance.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-rename-failure',
          payload: { action: 'resume' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'running' }),
          events: [event('resumed')]
        }),
      (error: unknown) => error instanceof JournalError && error.code === 'write-failed'
    )

    assert.equal(instance.read(ROLLOUT_A).snapshot.revision, 1)
    assert.deepEqual(
      instance.events(ROLLOUT_A, { limit: 10 }).items.map(item => item.kind),
      ['created']
    )
    assert.equal(journal(directory).read(ROLLOUT_A).snapshot.revision, 1)
  })
})

test('rejects truncated and schema-invalid records before dispatch can reopen them', () => {
  withTempDirectory(directory => {
    journal(directory).create(snapshot(), { events: [event('created')] })
    fs.writeFileSync(path.join(directory, `${ROLLOUT_A}.json`), '{', 'utf8')

    assert.throws(
      () => journal(directory),
      (error: unknown) => error instanceof JournalCorruptionError && error.code === 'corrupt-journal'
    )
  })

  withTempDirectory(directory => {
    fs.mkdirSync(directory, { recursive: true })
    fs.writeFileSync(
      path.join(directory, `${ROLLOUT_A}.json`),
      JSON.stringify({ schemaVersion: 99, id: ROLLOUT_A }),
      'utf8'
    )

    assert.throws(
      () => journal(directory),
      (error: unknown) => error instanceof JournalCorruptionError && error.code === 'corrupt-journal'
    )
  })
})

test('rejects symlink record paths through the injected filesystem seam', () => {
  withTempDirectory(directory => {
    let reportSymlink = false
    const recordPath = path.join(directory, `${ROLLOUT_A}.json`)
    const injected = realFs({
      lstatSync: filePath => {
        if (reportSymlink && filePath === recordPath) {
          return {
            isFile: () => false,
            isDirectory: () => false,
            isSymbolicLink: () => true
          } as fs.Stats
        }
        return fs.lstatSync(filePath)
      }
    })
    const instance = journal(directory, { fs: injected })
    instance.create(snapshot())
    reportSymlink = true

    assert.throws(
      () =>
        instance.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-symlink',
          payload: { action: 'pause' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'paused' })
        }),
      (error: unknown) =>
        error instanceof JournalError && (error.code === 'unsafe-path' || error.code === 'corrupt-journal')
    )
    assert.equal(instance.read(ROLLOUT_A).snapshot.revision, 1)
  })
})

test('dedupes the same request durably and rejects a mismatched payload before CAS', () => {
  withTempDirectory(directory => {
    const first = journal(directory)
    first.create(snapshot())
    const request = {
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-dedupe-1',
      payload: { action: 'pause', reason: 'operator' },
      snapshot: snapshot(ROLLOUT_A, { phase: 'paused' }),
      events: [event('paused')]
    }
    const accepted = first.record(request)
    const duplicate = first.record({ ...request, expectedRevision: 0 })

    assert.equal(duplicate.duplicate, true)
    assert.equal(duplicate.acceptanceId, accepted.acceptanceId)
    assert.equal(first.read(ROLLOUT_A).snapshot.revision, 2)
    assert.equal(first.events(ROLLOUT_A, { limit: 10 }).items.length, 1)

    const reopened = journal(directory)
    const durableDuplicate = reopened.record({ ...request, expectedRevision: 1 })
    assert.equal(durableDuplicate.duplicate, true)
    assert.equal(durableDuplicate.acceptanceId, accepted.acceptanceId)

    assert.throws(
      () => reopened.record({ ...request, payload: { action: 'resume' } }),
      (error: unknown) => error instanceof JournalError && error.code === 'request-payload-mismatch'
    )
    assert.equal(reopened.read(ROLLOUT_A).snapshot.revision, 2)
    assert.equal(reopened.events(ROLLOUT_A, { limit: 10 }).items.length, 1)
  })
})

test('checks stale revision after dedupe and does not reserve a rejected request', () => {
  withTempDirectory(directory => {
    const instance = journal(directory)
    instance.create(snapshot())
    instance.record({
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-revision-1',
      payload: { action: 'pause' },
      snapshot: snapshot(ROLLOUT_A, { phase: 'paused' }),
      events: [event('paused')]
    })

    assert.throws(
      () =>
        instance.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-stale',
          payload: { action: 'resume' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'running' }),
          events: [event('resumed')]
        }),
      (error: unknown) => error instanceof JournalError && error.code === 'stale-revision'
    )

    const accepted = instance.record({
      id: ROLLOUT_A,
      expectedRevision: 2,
      requestId: 'request-stale',
      payload: { action: 'resume' },
      snapshot: snapshot(ROLLOUT_A, { phase: 'running' }),
      events: [event('resumed')]
    })
    assert.equal(accepted.revision, 3)
  })
})

test('refreshes from disk before CAS so a stale service instance cannot overwrite a newer revision', () => {
  withTempDirectory(directory => {
    const creator = journal(directory)
    creator.create(snapshot())
    const stale = journal(directory)
    const writer = journal(directory)
    writer.record({
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-writer-1',
      payload: { action: 'pause' },
      snapshot: snapshot(ROLLOUT_A, { phase: 'paused' }),
      events: [event('paused')]
    })

    assert.throws(
      () =>
        stale.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-stale-service',
          payload: { action: 'resume' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'running' }),
          events: [event('resumed')]
        }),
      (error: unknown) => error instanceof JournalError && error.code === 'stale-revision'
    )
    assert.equal(journal(directory).read(ROLLOUT_A).snapshot.phase, 'paused')
  })
})

test('removing a resolved fence updates the durable unresolved index', () => {
  withTempDirectory(directory => {
    const instance = journal(directory)
    const unresolved = fence()
    instance.create(snapshot(ROLLOUT_A, { phase: 'stopped' }), { unresolved: [unresolved] })
    assert.equal(instance.hasUnresolvedInstall(INSTALL_A), true)

    instance.record({
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-resolve-fence',
      payload: { action: 'reconciled', key: unresolved.key },
      snapshot: snapshot(ROLLOUT_A, { phase: 'completed' }),
      events: [event('reconciled', INSTALL_A)],
      facts: [
        {
          kind: 'settlement-validated',
          rolloutId: ROLLOUT_A,
          correlationId: unresolved.correlationId,
          installId: INSTALL_A,
          observedAt: '2026-09-21T00:00:01.000Z',
          basis: 'validated terminal receipt and restored scope'
        }
      ],
      unresolved: { remove: [unresolved.key] }
    })

    assert.equal(instance.hasUnresolvedInstall(INSTALL_A), false)
    assert.equal(journal(directory).hasUnresolvedInstall(INSTALL_A), false)
  })
})

test('refuses fence release without a matching validated settlement fact', () => {
  withTempDirectory(directory => {
    const instance = journal(directory)
    const unresolved = fence()
    instance.create(snapshot(ROLLOUT_A, { phase: 'running' }), { unresolved: [unresolved] })

    assert.throws(
      () =>
        instance.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-unproven-release',
          payload: { action: 'reconciled' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'completed' }),
          unresolved: { remove: [unresolved.key] }
        }),
      (error: unknown) => error instanceof JournalError && error.code === 'fence-release-unproven'
    )
    assert.equal(instance.hasUnresolvedInstall(INSTALL_A), true)
  })
})

test('refuses a second writer while the canonical journal owner marker exists', () => {
  withTempDirectory(directory => {
    const instance = journal(directory)
    instance.create(snapshot())
    fs.writeFileSync(path.join(directory, '.owner'), '{"pid":1,"ownerId":"held"}', 'utf8')

    assert.throws(
      () =>
        instance.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-owner-held',
          payload: { action: 'pause' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'paused' })
        }),
      (error: unknown) => error instanceof JournalError && error.code === 'owner-unavailable'
    )
    assert.equal(instance.read(ROLLOUT_A).snapshot.revision, 1)
  })
})
test('a released stale lease cannot remove a replacement owner marker', () => {
  withTempDirectory(directory => {
    const instance = journal(directory)
    const first = instance.acquireOwner()
    const ownerPath = path.join(directory, '.owner')

    fs.unlinkSync(ownerPath)
    fs.writeFileSync(ownerPath, '{"pid":2,"ownerId":"replacement"}', 'utf8')
    first.release()

    assert.equal(fs.existsSync(ownerPath), true)
  })
})

test('keeps event cursors stable when later events are inserted', () => {
  withTempDirectory(directory => {
    const instance = journal(directory)
    instance.create(snapshot(), { events: [event('one'), event('two'), event('three'), event('four')] })

    const firstPage = instance.events(ROLLOUT_A, { limit: 2 })
    assert.deepEqual(
      firstPage.items.map(item => item.kind),
      ['one', 'two']
    )
    assert.ok(firstPage.nextCursor)

    instance.record({
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-events-1',
      payload: { action: 'append' },
      snapshot: snapshot(ROLLOUT_A),
      events: [event('five'), event('six')]
    })

    const secondPage = instance.events(ROLLOUT_A, { cursor: firstPage.nextCursor!, limit: 2 })
    assert.deepEqual(
      secondPage.items.map(item => item.kind),
      ['three', 'four']
    )
    const thirdPage = instance.events(ROLLOUT_A, { cursor: secondPage.nextCursor!, limit: 10 })
    assert.deepEqual(
      thirdPage.items.map(item => item.kind),
      ['five', 'six']
    )
  })
})

test('persists summaries and archive metadata, including after reopen', () => {
  withTempDirectory(directory => {
    const instance = journal(directory, { clock: () => '2026-09-21T01:00:00.000Z' })
    instance.create(snapshot(ROLLOUT_A, { phase: 'stopped' }))
    const archived = instance.archive({
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-archive-1',
      payload: { action: 'archive', reason: 'operator stopped' },
      actor: 'local-operator',
      reason: 'operator stopped'
    })

    assert.equal(archived.revision, 2)
    const summary = instance.history({ limit: 10 }).items[0]
    assert.equal(summary.id, ROLLOUT_A)
    assert.equal(summary.archive?.actor, 'local-operator')
    assert.equal(summary.archive?.reason, 'operator stopped')
    assert.equal(summary.archived, true)
    assert.equal(instance.read(ROLLOUT_A).archive?.at, '2026-09-21T01:00:00.000Z')

    const reopened = journal(directory)
    assert.equal(reopened.history({ limit: 10 }).items[0].archive?.reason, 'operator stopped')
  })
})

test('prunes old settled records but retains an unresolved tombstone and its fence', () => {
  withTempDirectory(directory => {
    const instance = journal(directory, { retentionLimit: 1 })
    instance.create(snapshot(ROLLOUT_A, { phase: 'stopped' }), { unresolved: [fence()] })
    instance.archive({
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-archive-a',
      payload: { action: 'archive', id: ROLLOUT_A },
      actor: 'local-operator',
      reason: 'stopped with unknown outcome'
    })
    instance.create(snapshot(ROLLOUT_B, { phase: 'completed' }))
    instance.archive({
      id: ROLLOUT_B,
      expectedRevision: 1,
      requestId: 'request-archive-b',
      payload: { action: 'archive', id: ROLLOUT_B },
      actor: 'local-operator',
      reason: 'settled'
    })
    instance.create(snapshot(ROLLOUT_C, { phase: 'completed' }))
    instance.archive({
      id: ROLLOUT_C,
      expectedRevision: 1,
      requestId: 'request-archive-c',
      payload: { action: 'archive', id: ROLLOUT_C },
      actor: 'local-operator',
      reason: 'settled'
    })

    const result = instance.prune()
    assert.ok(result.prunedRecordIds.includes(ROLLOUT_A))
    assert.ok(!fs.existsSync(path.join(directory, `${ROLLOUT_A}.json`)))
    assert.equal(instance.hasUnresolvedInstall(INSTALL_A), true)
    const tombstone = instance.unresolvedIndex().find(item => item.key === `${ROLLOUT_A}:${INSTALL_A}`)
    assert.equal(tombstone?.tombstone, true)
    assert.equal(tombstone?.rolloutId, ROLLOUT_A)

    const reopened = journal(directory, { retentionLimit: 1 })
    assert.equal(reopened.hasUnresolvedInstall(INSTALL_A), true)
    assert.equal(reopened.unresolvedIndex().find(item => item.key === `${ROLLOUT_A}:${INSTALL_A}`)?.tombstone, true)
    assert.equal(
      reopened.history({ limit: 10 }).items.some(item => item.id === ROLLOUT_A),
      true
    )
  })
})

test('refuses replay against a compacted tombstone', () => {
  withTempDirectory(directory => {
    const instance = journal(directory, { retentionLimit: 1 })
    instance.create(snapshot(ROLLOUT_A, { phase: 'completed' }))
    instance.create(snapshot(ROLLOUT_B, { phase: 'completed' }))
    instance.create(snapshot(ROLLOUT_C, { phase: 'completed' }))
    instance.prune()

    assert.throws(
      () =>
        instance.record({
          id: ROLLOUT_A,
          expectedRevision: 1,
          requestId: 'request-after-compaction',
          payload: { action: 'resume' },
          snapshot: snapshot(ROLLOUT_A, { phase: 'running' })
        }),
      (error: unknown) => error instanceof JournalError && error.code === 'replay-expired'
    )
  })
})

test('a crash after the record rename is recoverable and dedupe prevents replay', () => {
  withTempDirectory(directory => {
    let crashAfterRename = false
    const recordPath = path.join(directory, `${ROLLOUT_A}.json`)
    const injected = realFs({
      renameSync: (from, to) => {
        fs.renameSync(from, to)
        if (crashAfterRename && to === recordPath) {
          crashAfterRename = false
          throw new Error('simulated process crash after durable record rename')
        }
      }
    })
    const instance = journal(directory, { fs: injected })
    instance.create(snapshot())
    crashAfterRename = true

    const request = {
      id: ROLLOUT_A,
      expectedRevision: 1,
      requestId: 'request-crash-boundary',
      payload: { action: 'pause' },
      snapshot: snapshot(ROLLOUT_A, { phase: 'paused' }),
      events: [event('paused')]
    }
    assert.throws(() => instance.record(request), /simulated process crash/)

    const reopened = journal(directory)
    assert.equal(reopened.read(ROLLOUT_A).snapshot.phase, 'paused')
    const duplicate = reopened.record({ ...request, expectedRevision: 1 })
    assert.equal(duplicate.duplicate, true)
    assert.equal(reopened.events(ROLLOUT_A, { limit: 10 }).items.length, 1)
  })
})
