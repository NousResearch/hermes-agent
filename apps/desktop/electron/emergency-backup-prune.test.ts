import { describe, expect, it } from 'vitest'

import { selectEmergencyBackupsToDelete } from './emergency-backup-prune'

const NEW = 'state.db.pre-update-emergency-2026-09-15T09-34-46-808Z.bak'
const MID = 'state.db.pre-update-emergency-2026-09-15T05-35-10-317Z.bak'
const OLD = 'state.db.pre-update-emergency-2026-09-15T00-57-19-753Z.bak'
const OLDEST = 'state.db.pre-update-emergency-2026-09-09T05-12-27-903Z.bak'

describe('selectEmergencyBackupsToDelete', () => {
  it('counts the just-written backup toward the retention limit', () => {
    // The regression: with keep=2 and the fresh backup on disk, three files must not survive.
    const deleted = selectEmergencyBackupsToDelete([OLD, MID, NEW], NEW, 2)

    expect(deleted).toEqual([OLD])
  })

  it('leaves exactly `keep` backups behind', () => {
    const entries = [OLDEST, OLD, MID, NEW]

    const deleted = selectEmergencyBackupsToDelete(entries, NEW, 2)
    const survivors = entries.filter(f => !deleted.includes(f))

    expect(survivors).toHaveLength(2)
    expect(survivors).toEqual([MID, NEW])
  })

  it('keeps the newest backups and deletes the oldest', () => {
    const deleted = selectEmergencyBackupsToDelete([OLDEST, OLD, MID, NEW], NEW, 2)

    expect(deleted).toEqual([OLD, OLDEST])
  })

  it('deletes nothing when the limit is not yet reached', () => {
    expect(selectEmergencyBackupsToDelete([NEW], NEW, 2)).toEqual([])
    expect(selectEmergencyBackupsToDelete([MID, NEW], NEW, 2)).toEqual([])
  })

  it('never deletes the backup this update just wrote', () => {
    // Even at keep=1 the fresh copy is the survivor: it is the only one matching the
    // binary the user is updating away from.
    expect(selectEmergencyBackupsToDelete([OLD, MID, NEW], NEW, 1)).toEqual([MID, OLD])
  })

  it('counts the fresh backup even when the directory listing predates it', () => {
    // readdir raced the copy: NEW is absent from entries but already exists on disk.
    const deleted = selectEmergencyBackupsToDelete([OLD, MID], NEW, 2)

    expect(deleted).toEqual([OLD])
    expect(deleted).not.toContain(NEW)
  })

  it('ignores unrelated files in the hermes home directory', () => {
    const entries = [
      'state.db',
      'state.db-wal',
      'state.db-shm',
      'state.db.pre-update-emergency-2026-09-15T09-34-46-808Z.bak-wal',
      'state.db.auto-maintenance.lock',
      OLD,
      MID,
      NEW
    ]

    const deleted = selectEmergencyBackupsToDelete(entries, NEW, 2)

    expect(deleted).toEqual([OLD])
  })

  it('supports keep=0 by deleting every backup including the fresh one', () => {
    expect(selectEmergencyBackupsToDelete([OLD, MID, NEW], NEW, 0)).toEqual([NEW, MID, OLD])
  })

  it('rejects a negative limit', () => {
    expect(() => selectEmergencyBackupsToDelete([NEW], NEW, -1)).toThrow(RangeError)
  })
})
