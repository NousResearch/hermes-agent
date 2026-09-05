import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { preflightStateDb, STATE_DB_BACKUP_FREE_RESERVE_BYTES } from './state-db-preflight'

const roots: string[] = []
const SQLITE_HEADER = Buffer.concat([Buffer.from('SQLite format 3\0'), Buffer.alloc(128)])

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-state-db-preflight-'))
  roots.push(root)
  fs.writeFileSync(path.join(root, 'state.db'), SQLITE_HEADER)

  return root
}

function backup(root: string, stamp: string, body = 'prior') {
  const file = path.join(root, `state.db.pre-update-emergency-${stamp}.bak`)
  fs.writeFileSync(file, body)

  return file
}

afterEach(() => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('prunes stale emergency copies before allocating a clone backup', () => {
  const root = fixture()
  const oldest = backup(root, '2026-08-01T00-00-00-000Z')
  const middle = backup(root, '2026-08-02T00-00-00-000Z')
  const newest = backup(root, '2026-08-03T00-00-00-000Z')
  const order: string[] = []

  const result = preflightStateDb(root, () => {}, {
    now: () => new Date('2026-09-03T20:00:00.000Z'),
    statfsSync: () => ({
      bavail: SQLITE_HEADER.length + STATE_DB_BACKUP_FREE_RESERVE_BYTES,
      bsize: 1
    }),
    unlinkSync: file => {
      order.push(`unlink:${path.basename(file)}`)
      fs.unlinkSync(file)
    },
    cloneFile: (source, destination) => {
      order.push('clone')
      fs.copyFileSync(source, destination)
    }
  })

  assert.equal(result.status, 'created')
  assert.equal(result.method, process.platform === 'darwin' ? 'clone-or-physical' : 'clone')
  assert.equal(fs.existsSync(oldest), false)
  assert.equal(fs.existsSync(middle), false)
  assert.equal(fs.existsSync(newest), true)
  assert.equal(fs.existsSync(result.path), true)
  assert.deepEqual(order.slice(0, 2), [
    'unlink:state.db.pre-update-emergency-2026-08-02T00-00-00-000Z.bak',
    'unlink:state.db.pre-update-emergency-2026-08-01T00-00-00-000Z.bak'
  ])
  assert.equal(order[2], 'clone')
})

test('low disk refuses physical fallback and removes a partial reflink destination', () => {
  const root = fixture()
  const newest = backup(root, '2026-08-03T00-00-00-000Z')
  let physicalCopies = 0
  let spaceChecks = 0

  const result = preflightStateDb(root, () => {}, {
    now: () => new Date('2026-09-03T20:00:00.000Z'),
    statfsSync: () => {
      spaceChecks += 1

      return {
        bavail:
          SQLITE_HEADER.length +
          STATE_DB_BACKUP_FREE_RESERVE_BYTES -
          (process.platform === 'darwin' && spaceChecks === 1 ? 0 : 1),
        bsize: 1
      }
    },
    cloneFile: (_source, destination) => {
      fs.writeFileSync(destination, 'partial')
      throw new Error('clone unsupported')
    },
    copyFileSync: (source, destination) => {
      physicalCopies += 1
      fs.copyFileSync(source, destination)
    }
  })

  assert.equal(result.status, 'skipped-insufficient-space')
  assert.equal(physicalCopies, 0)
  assert.equal(fs.existsSync(newest), true)
  assert.equal(fs.existsSync(result.path), false)
})

test.runIf(process.platform === 'darwin')(
  'low disk refuses the macOS clone path before cp can fall back to a physical copy',
  () => {
    const root = fixture()
    let cloneAttempts = 0

    const result = preflightStateDb(root, () => {}, {
      now: () => new Date('2026-09-03T20:00:00.000Z'),
      statfsSync: () => ({
        bavail: SQLITE_HEADER.length + STATE_DB_BACKUP_FREE_RESERVE_BYTES - 1,
        bsize: 1
      }),
      cloneFile: () => {
        cloneAttempts += 1
      }
    })

    assert.equal(result.status, 'skipped-insufficient-space')
    assert.equal(cloneAttempts, 0)
    assert.equal(fs.existsSync(result.path), false)
  }
)

test.runIf(process.platform === 'darwin')(
  'unverified headroom refuses the macOS clone path before cp can fall back to a physical copy',
  () => {
    const root = fixture()
    let cloneAttempts = 0

    const result = preflightStateDb(root, () => {}, {
      now: () => new Date('2026-09-03T20:00:00.000Z'),
      statfsSync: () => {
        throw new Error('statfs unavailable')
      },
      cloneFile: () => {
        cloneAttempts += 1
      }
    })

    assert.equal(result.status, 'skipped-unverified-space')
    assert.equal(cloneAttempts, 0)
    assert.equal(fs.existsSync(result.path), false)
  }
)

test('falls back to a physical copy only when verified headroom remains', () => {
  const root = fixture()
  let physicalCopies = 0

  const result = preflightStateDb(root, () => {}, {
    now: () => new Date('2026-09-03T20:00:00.000Z'),
    statfsSync: () => ({
      bavail: SQLITE_HEADER.length + STATE_DB_BACKUP_FREE_RESERVE_BYTES,
      bsize: 1
    }),
    cloneFile: () => {
      throw new Error('clone unsupported')
    },
    copyFileSync: (source, destination) => {
      physicalCopies += 1
      fs.copyFileSync(source, destination)
    }
  })

  assert.equal(result.status, 'created')
  assert.equal(result.method, 'physical')
  assert.equal(physicalCopies, 1)
  assert.deepEqual(fs.readFileSync(result.path), SQLITE_HEADER)
})

test('refreshes a growing source size before allocating a physical backup', () => {
  const root = fixture()
  const source = path.join(root, 'state.db')
  let physicalCopies = 0

  const result = preflightStateDb(root, () => {}, {
    cloneFile: null,
    now: () => new Date('2026-09-03T20:00:00.000Z'),
    statfsSync: () => {
      fs.appendFileSync(source, 'growth')

      return {
        bavail: SQLITE_HEADER.length + STATE_DB_BACKUP_FREE_RESERVE_BYTES,
        bsize: 1
      }
    },
    copyFileSync: (from, destination) => {
      physicalCopies += 1
      fs.copyFileSync(from, destination)
    }
  })

  assert.equal(result.status, 'skipped-insufficient-space')
  assert.equal(physicalCopies, 0)
  assert.equal(fs.existsSync(result.path), false)
})

test('accepts a completed valid snapshot when the source grows during copying', () => {
  const root = fixture()
  const source = path.join(root, 'state.db')

  const result = preflightStateDb(root, () => {}, {
    cloneFile: null,
    now: () => new Date('2026-09-03T20:00:00.000Z'),
    statfsSync: () => ({
      bavail: SQLITE_HEADER.length + STATE_DB_BACKUP_FREE_RESERVE_BYTES + 1024,
      bsize: 1
    }),
    copyFileSync: (from, destination) => {
      fs.appendFileSync(source, 'growth')
      fs.copyFileSync(from, destination)
    }
  })

  assert.equal(result.status, 'created')
  assert.equal(result.method, 'physical')
  assert.deepEqual(fs.readFileSync(result.path), Buffer.concat([SQLITE_HEADER, Buffer.from('growth')]))
})

test('rejects a same-size snapshot without a SQLite header', () => {
  const root = fixture()

  const result = preflightStateDb(root, () => {}, {
    cloneFile: (_source, destination) => {
      fs.writeFileSync(destination, Buffer.alloc(SQLITE_HEADER.length))
    },
    now: () => new Date('2026-09-03T20:00:00.000Z'),
    statfsSync: () => ({
      bavail: SQLITE_HEADER.length + STATE_DB_BACKUP_FREE_RESERVE_BYTES,
      bsize: 1
    }),
    copyFileSync: (_source, destination) => {
      fs.writeFileSync(destination, Buffer.alloc(SQLITE_HEADER.length))
    }
  })

  assert.equal(result.status, 'failed')
  assert.equal(fs.existsSync(result.path), false)
})

test('removes a partial physical backup when fallback copying fails', () => {
  const root = fixture()

  const result = preflightStateDb(root, () => {}, {
    cloneFile: null,
    statfsSync: () => ({
      bavail: SQLITE_HEADER.length + STATE_DB_BACKUP_FREE_RESERVE_BYTES,
      bsize: 1
    }),
    copyFileSync: (_source, destination) => {
      fs.writeFileSync(destination, 'partial')
      throw new Error('ENOSPC')
    },
    now: () => new Date('2026-09-03T20:00:00.000Z')
  })

  assert.equal(result.status, 'failed')
  assert.equal(fs.existsSync(result.path), false)
})

test.runIf(process.platform === 'darwin')('creates a guarded cp -c backup for a large sparse database', () => {
  const root = fixture()
  const source = path.join(root, 'state.db')

  fs.truncateSync(source, 64 * 1024 * 1024)

  const result = preflightStateDb(root, () => {}, {
    now: () => new Date('2026-09-03T20:00:00.000Z'),
    statfsSync: () => ({
      bavail: fs.statSync(source).size + STATE_DB_BACKUP_FREE_RESERVE_BYTES,
      bsize: 1
    })
  })

  assert.equal(result.status, 'created')
  assert.equal(result.method, 'clone-or-physical')
  assert.equal(fs.statSync(result.path).size, fs.statSync(source).size)
})
