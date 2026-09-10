// Retention policy for pre-update emergency state.db backups (#91229).
//
// The reported symptom was 2.5-3 GB of `state.db.pre-update-emergency-*.bak`
// files that "are never cleaned up". A prune did exist, but it was nested
// inside the `try` whose first statement wrote the new backup, so it only ran
// when that write succeeded. These tests pin the selection rule; the
// unconditional-call half is asserted structurally at the bottom.

import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import {
  EMERGENCY_BACKUP_PREFIX,
  EMERGENCY_BACKUP_RETENTION,
  EMERGENCY_BACKUP_SUFFIX,
  isEmergencyBackup,
  selectEmergencyBackupsToDelete,
} from './emergency-backup-retention'

/** Build the exact filename `preflightStateDb` writes for a given instant. */
function backupName(iso: string): string {
  return `${EMERGENCY_BACKUP_PREFIX}${iso.replace(/[:.]/g, '-')}${EMERGENCY_BACKUP_SUFFIX}`
}

const OLDEST = backupName('2026-08-15T03:11:52.004Z')
const OLDER = backupName('2026-08-17T09:02:00.900Z')
const NEWER = backupName('2026-08-20T22:45:10.120Z')
const NEWEST = backupName('2026-08-21T01:30:00.000Z')

describe('isEmergencyBackup', () => {
  it('accepts exactly the filenames preflightStateDb writes', () => {
    expect(isEmergencyBackup(NEWEST)).toBe(true)
  })

  it('leaves the live database and its sidecars alone', () => {
    // The single most important negative: this sweep runs in HERMES_HOME, so
    // a loose predicate deletes the database the backups exist to protect.
    expect(isEmergencyBackup('state.db')).toBe(false)
    expect(isEmergencyBackup('state.db-wal')).toBe(false)
    expect(isEmergencyBackup('state.db-shm')).toBe(false)
  })

  it('leaves other .bak files alone', () => {
    // The Python-level snapshot is a different mechanism with a different
    // lifetime; this policy does not own it.
    expect(isEmergencyBackup('state.db.bak')).toBe(false)
    expect(isEmergencyBackup('config.yaml.bak')).toBe(false)
    expect(isEmergencyBackup(`${EMERGENCY_BACKUP_PREFIX}2026-08-21.tmp`)).toBe(false)
  })

  it('does not match a prefixed name embedded mid-filename', () => {
    expect(isEmergencyBackup(`copy-of-${NEWEST}`)).toBe(false)
  })
})

describe('selectEmergencyBackupsToDelete', () => {
  it('keeps the newest RETENTION backups and returns the rest', () => {
    const doomed = selectEmergencyBackupsToDelete([OLDER, NEWEST, OLDEST, NEWER])

    expect(EMERGENCY_BACKUP_RETENTION).toBe(3)
    expect(doomed).toEqual([OLDEST])
  })

  it('returns nothing while at or under the budget', () => {
    expect(selectEmergencyBackupsToDelete([])).toEqual([])
    expect(selectEmergencyBackupsToDelete([NEWEST])).toEqual([])
    expect(selectEmergencyBackupsToDelete([NEWEST, NEWER, OLDER])).toEqual([])
  })

  it('counts the just-written backup toward the budget', () => {
    // The old code excluded the new file before slicing, so three survived
    // while the comment said two. Whatever the number is, the newest file has
    // to be inside it or the budget is off by one forever.
    const listing = [OLDEST, OLDER, NEWER, NEWEST]
    const kept = listing.filter(f => !selectEmergencyBackupsToDelete(listing).includes(f))

    expect(kept).toHaveLength(EMERGENCY_BACKUP_RETENTION)
    expect(kept).toContain(NEWEST)
  })

  it('orders by name, not by array order or mtime', () => {
    // Sorting is lexicographic because the ISO timestamp is fixed-width; a
    // shuffled listing (readdir gives no ordering guarantee) must not change
    // which files survive.
    const shuffled = [NEWER, OLDEST, NEWEST, OLDER]

    expect(selectEmergencyBackupsToDelete(shuffled)).toEqual([OLDEST])
  })

  it('ignores unrelated files when choosing what to delete', () => {
    const doomed = selectEmergencyBackupsToDelete([
      'state.db',
      'state.db-wal',
      'config.yaml',
      OLDEST,
      OLDER,
      NEWER,
      NEWEST,
    ])

    expect(doomed).toEqual([OLDEST])
  })

  it('deletes everything when told to retain none', () => {
    expect(selectEmergencyBackupsToDelete([NEWEST, OLDEST], 0)).toEqual([NEWEST, OLDEST])
  })

  it('treats a nonsensical retention as retain-none rather than deleting at random', () => {
    expect(selectEmergencyBackupsToDelete([NEWEST, OLDEST], -1)).toEqual([NEWEST, OLDEST])
    expect(selectEmergencyBackupsToDelete([NEWEST, OLDEST], Number.NaN)).toEqual([NEWEST, OLDEST])
  })

  it('scales past the budget without dropping any candidate', () => {
    const many = Array.from({ length: 40 }, (_, i) =>
      backupName(`2026-08-21T0${Math.floor(i / 10)}:${String(i % 10).padStart(2, '0')}:00.000Z`)
    )

    const doomed = selectEmergencyBackupsToDelete(many)

    expect(doomed).toHaveLength(many.length - EMERGENCY_BACKUP_RETENTION)
    // Nothing kept is also deleted, and nothing is listed twice.
    expect(new Set(doomed).size).toBe(doomed.length)
  })
})

describe('preflightStateDb wiring', () => {
  // The selection rule above is only half the fix. The bug was WHERE the
  // sweep was called from, and that is a property of main.ts, which cannot be
  // imported here (it boots Electron). So assert it against the source, the
  // same way the repo pins other cross-module invariants.
  const main = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
  const body = main.slice(main.indexOf('function preflightStateDb('))

  it('sweeps before the first early return, not after the copy succeeds', () => {
    const call = body.indexOf('pruneEmergencyStateDbBackups(hermesHome, rememberLog)')
    // Anchor on the guard itself, not on the word "return": the surrounding
    // comments say "early return" and would satisfy a naive search.
    const firstGuard = body.indexOf('if (!fileExists(stateDbPath))')
    const copy = body.indexOf('fs.copyFileSync(stateDbPath, emergencyPath)')

    expect(call).toBeGreaterThan(-1)
    expect(firstGuard).toBeGreaterThan(-1)
    // Before the first guard => a missing or too-small state.db still reclaims.
    expect(call).toBeLessThan(firstGuard)
    // Before the copy => ENOSPC and EBUSY still reclaim. This is the assertion
    // that fails if anyone re-nests the sweep inside the copy's try block.
    expect(call).toBeLessThan(copy)
  })

  it('no longer carries the inline prune that was coupled to the copy', () => {
    expect(body).not.toContain("f.startsWith('state.db.pre-update-emergency-')")
  })
})
