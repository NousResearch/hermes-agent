/**
 * Which pre-update emergency `state.db` backups to delete.
 *
 * The updater copies `state.db` to `state.db.pre-update-emergency-<ts>.bak` before every
 * update, then prunes so only the newest `keep` copies survive. On a multi-GB `state.db`
 * an off-by-one here costs gigabytes per update, silently, so the selection is a pure
 * function that can be tested without touching the filesystem.
 *
 * `justWritten` is the backup this update produced. It is already on disk by the time the
 * prune runs, so it MUST be part of the retention count — excluding it from the candidate
 * list keeps one more file than `keep` asks for.
 */
export function selectEmergencyBackupsToDelete(
  entries: readonly string[],
  justWritten: string,
  keep: number
): string[] {
  if (keep < 0) {
    throw new RangeError(`keep must be >= 0, got ${keep}`)
  }

  const backups = entries
    .filter(f => f.startsWith('state.db.pre-update-emergency-') && f.endsWith('.bak'))
    .slice()
    .sort()
    .reverse()

  // The just-written backup is the newest; keep it at the front even when the caller's
  // directory listing predates it, so a stale listing cannot delete the fresh copy.
  const ordered = backups.includes(justWritten)
    ? backups
    : [justWritten, ...backups]

  return ordered.slice(Math.max(keep, 0))
}
