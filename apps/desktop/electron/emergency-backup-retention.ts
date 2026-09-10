// Retention policy for the pre-update emergency copies of `state.db`.
//
// Before an update, `preflightStateDb` copies `state.db` to
// `state.db.pre-update-emergency-<timestamp>.bak` so a torn update can be
// recovered from. Those copies are the size of the live database, which on a
// busy install is comfortably several hundred MB, so they have to be reclaimed
// or they eat the disk (issue #91229 reported 2.5-3 GB of accumulated .bak
// files and ~6.5 GB of week-over-week C: growth).
//
// The sweep used to live inline in `preflightStateDb`, nested inside the `try`
// whose first statement was the `copyFileSync`. That coupled reclaiming old
// backups to successfully writing a new one, so every failure mode that
// *prevents* the copy also skipped the cleanup:
//
//   * the copy throws ENOSPC because the disk is full -- and it is full partly
//     because these backups were never reclaimed;
//   * the copy throws EBUSY/EPERM because another process holds `state.db`,
//     which on Windows is the ordinary state during a failed self-update and
//     the exact scenario #91229 is filed about;
//   * `state.db` is missing or too small to be a real database, both of which
//     `return` before the copy is ever attempted.
//
// In other words the janitor only ran on the days nothing needed cleaning.
// Splitting the selection out here lets `preflightStateDb` run it
// unconditionally, and lets it be tested without Electron or a real disk.

/** Filename prefix written by `preflightStateDb`. */
export const EMERGENCY_BACKUP_PREFIX = 'state.db.pre-update-emergency-'

/** Filename suffix written by `preflightStateDb`. */
export const EMERGENCY_BACKUP_SUFFIX = '.bak'

/**
 * How many emergency backups to keep, newest first, counting the one just
 * written.
 *
 * This preserves the effective behaviour of the previous implementation
 * rather than the behaviour its comment claimed. That comment said "Prune to
 * the 2 most recent", but the filter excluded the backup it had just created
 * before slicing, so three files survived: the new one plus two older. At the
 * reported ~650-750 MB apiece that is the difference between ~1.4 GB and
 * ~2.1 GB retained, so it is not a rounding error, and picking the smaller
 * number would delete recovery data users currently have. Deciding which
 * number is actually wanted is a maintainer call; this change only makes the
 * number explicit and honest.
 */
export const EMERGENCY_BACKUP_RETENTION = 3

/** True when *name* is one of the emergency backups this module manages. */
export function isEmergencyBackup(name: string): boolean {
  return (
    typeof name === 'string' &&
    name.startsWith(EMERGENCY_BACKUP_PREFIX) &&
    name.endsWith(EMERGENCY_BACKUP_SUFFIX)
  )
}

/**
 * Given every filename in the Hermes home directory, return the emergency
 * backups that should be deleted -- everything past the newest *retain*.
 *
 * Ordering is lexicographic and deliberately so: the timestamp is
 * `new Date().toISOString()` with `:` and `.` replaced by `-`, which is
 * fixed-width and zero-padded, so byte order and chronological order agree.
 * That keeps the sweep independent of file mtimes, which a backup/restore or
 * a sync client can rewrite.
 *
 * Non-backup files are ignored, so this is safe to hand a whole directory
 * listing. Returns names, not paths; the caller owns the directory.
 */
export function selectEmergencyBackupsToDelete(
  names: readonly string[],
  retain: number = EMERGENCY_BACKUP_RETENTION
): string[] {
  const keep = Number.isFinite(retain) && retain > 0 ? Math.floor(retain) : 0

  return names.filter(isEmergencyBackup).sort().reverse().slice(keep)
}
