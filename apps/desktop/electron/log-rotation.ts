// Any log the shell keeps across launches needs a size bound — desktop.log has
// been seen at ~326 GB, which exhausts the disk and then breaks update/install
// (no room for git/venv/npm temp files).
//
// Mirror the Python logs (hermes_logging.py RotatingFileHandler, maxBytes x
// backupCount): cascade live -> .1 -> .2 -> .3, drop the oldest. Steady-state
// stays bounded at ~(backupCount + 1) x cap however hard the app loops.
//
// Bounding alone never RECLAIMS an already-huge file: a plain rotation just
// renames the monster to .1 and strands it for a cycle a healthy app may never
// reach. A multi-GB boot-loop transcript has no diagnostic value, so anything
// past the discard ceiling is deleted outright — the updated app self-heals a
// disk a stale build filled, on the next launch.

export const LOG_MAX_BYTES = 10 * 1024 * 1024
export const LOG_BACKUP_COUNT = 3
export const LOG_DISCARD_BYTES = LOG_MAX_BYTES * 4

export const logBackupPath = (base: string, n: number): string => `${base}.${n}`

export type LogRotationOp = ['rm', string] | ['mv', string, string]

// Pure planner: ordered fs ops to bound the live log at `base`. [] = nothing.
// Each step is ['rm', path] or ['mv', src, dst]; executed best-effort so a
// missing chain link never aborts the rest.
export function planLogRotation(size: number, base: string): LogRotationOp[] {
  if (size < LOG_MAX_BYTES) {
    return []
  }

  const backups = (n: number) => Array.from({ length: n }, (_, i) => logBackupPath(base, i + 1))

  // Pathological boot-loop log: reclaim live + every backup outright.
  if (size > LOG_DISCARD_BYTES) {
    return [base, ...backups(LOG_BACKUP_COUNT)].map(p => ['rm', p] as LogRotationOp)
  }

  // Cascade: drop oldest, shift each up, live -> .1.
  const ops: LogRotationOp[] = [['rm', logBackupPath(base, LOG_BACKUP_COUNT)]]

  for (let i = LOG_BACKUP_COUNT - 1; i >= 1; i--) {
    ops.push(['mv', logBackupPath(base, i), logBackupPath(base, i + 1)])
  }

  ops.push(['mv', base, logBackupPath(base, 1)])

  return ops
}
