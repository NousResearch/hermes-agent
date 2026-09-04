// Stale git lock-file recovery for the desktop update-check path.
//
// A crashed or killed `git fetch` on a shallow clone can leave
// `.git/shallow.lock` behind. Every later fetch then fails with
// "fatal: Unable to create '.git/shallow.lock': File exists", so the desktop
// update check reports 'fetch-failed' forever — git never self-heals these
// lock files. Mirrors hermes_cli/gitlock.py: a lock is removed only when it
// is older than the min age AND no git process is currently running.

import { execFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

// Lock files younger than this are presumed live (a fetch is in flight) and
// are never removed. git lock files live for seconds under normal operation;
// anything older than 10 minutes is abandoned.
export const STALE_LOCK_MIN_AGE_MS = 10 * 60 * 1000

// Same self-healable lock set as hermes_cli/gitlock.py.
export const LOCK_NAMES = ['shallow.lock', 'index.lock', 'HEAD.lock', 'MERGE_HEAD.lock']

function gitProcessRunning(): Promise<boolean> {
  return new Promise(resolve => {
    const [cmd, args] =
      process.platform === 'win32'
        ? ['tasklist', ['/FI', 'IMAGENAME eq git.exe', '/FO', 'CSV']]
        : ['pgrep', ['-x', 'git']]

    // windowsHide: on Windows this tasklist probe runs from the GUI-subsystem
    // Electron main process, so without it every update-check poll allocates
    // (and flashes) a visible console window for the child (#96694).
    execFile(cmd, args, { timeout: 10_000, windowsHide: true }, (error, stdout) => {
      if (process.platform === 'win32') {
        // tasklist exits 0 either way; presence is signaled in stdout.
        resolve(Boolean(stdout && stdout.toLowerCase().includes('git.exe')))

        return
      }

      // pgrep: exit 0 = at least one match; 1 = none; other = probe failure.
      // On probe failure stay conservative: report "running" so no lock is
      // touched when we cannot tell.
      if (error && (error as any).code === 1) {
        resolve(false)

        return
      }

      resolve(true)
    })
  })
}

// Remove abandoned .git lock files under repoRoot. Returns removed paths.
// Never throws: a lock we cannot stat or unlink is skipped.
export async function clearStaleGitLocks(
  repoRoot: string,
  {
    minAgeMs = STALE_LOCK_MIN_AGE_MS,
    isGitRunning = gitProcessRunning
  }: {
    minAgeMs?: number
    isGitRunning?: () => Promise<boolean>
  } = {}
): Promise<string[]> {
  const gitDir = path.join(repoRoot, '.git')
  const removed: string[] = []

  try {
    if (!fs.statSync(gitDir).isDirectory()) {
      return removed
    }
  } catch {
    return removed
  }

  if (await isGitRunning()) {
    return removed
  }

  const cutoff = Date.now() - minAgeMs

  for (const name of LOCK_NAMES) {
    const lockPath = path.join(gitDir, name)

    try {
      const st = fs.statSync(lockPath)

      if (st.isFile() && st.mtimeMs < cutoff) {
        fs.unlinkSync(lockPath)
        removed.push(lockPath)
      }
    } catch {
      // Missing or concurrently removed — skipping is always safe.
    }
  }

  return removed
}
