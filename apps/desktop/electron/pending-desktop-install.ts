import fs from 'node:fs'
import path from 'node:path'

/** The durable record an updater writes when a running bundle made it skip an
 *  install (#123737): `<HERMES_HOME>/pending_desktop_install.json`, written by
 *  `hermes_cli/main_desktop.py::record_pending_desktop_install` and completed
 *  by `hermes desktop finish-update` (or the next successful install pass). */

export interface PendingDesktopInstall {
  app: string
  rebuilt_app: string
  asar_hash: string
  recorded_at?: string
}

export function pendingDesktopInstallPath(hermesHome: string): string {
  return path.join(hermesHome, 'pending_desktop_install.json')
}

/**
 * Read the pending-install record, or null when absent/unreadable. Fail-quiet:
 * a corrupt or missing record must never gate the version IPC or the boot path.
 */
export function readPendingDesktopInstall(
  hermesHome: string,
  readFile: (p: string) => string = (p: string): string => fs.readFileSync(p, 'utf8'),
  exists: (p: string) => boolean = (p: string): boolean => fs.existsSync(p)
): PendingDesktopInstall | null {
  const recordPath = pendingDesktopInstallPath(hermesHome)

  if (!exists(recordPath)) {
    return null
  }

  try {
    const parsed = JSON.parse(readFile(recordPath)) as Partial<PendingDesktopInstall>

    // Only a record whose staged bundle can still be named is actionable; the
    // CLI clears unactionable records itself, so anything odd here is a torn
    // write — treat it as absent.
    if (typeof parsed.app !== 'string' || typeof parsed.rebuilt_app !== 'string') {
      return null
    }

    return { app: parsed.app, rebuilt_app: parsed.rebuilt_app, asar_hash: String(parsed.asar_hash ?? '') }
  } catch {
    return null
  }
}
