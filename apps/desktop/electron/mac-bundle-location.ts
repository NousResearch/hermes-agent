/**
 * Where a packaged macOS app bundle lives, as far as macOS can take it away. The Applications move
 * prompt (mac-applications-guard.ts) and bundled CLI provisioning (cli-provision.ts) both read it;
 * the prompt layers its own Downloads, Desktop and dismissal rules on top.
 */
import fs from 'node:fs'
import path from 'node:path'

export type MacEphemeralBundleLocation = 'app-translocation' | 'read-only-volume'

/** The running app's .app bundle: the executable is <App>.app/Contents/MacOS/<exe>. */
export function resolveMacAppBundle(exePath: string, platform: NodeJS.Platform = process.platform): string | null {
  if (platform !== 'darwin' || !exePath) {
    return null
  }

  let dir = path.posix.dirname(exePath)

  for (let i = 0; i < 2; i++) {
    dir = path.posix.dirname(dir)
  }

  return dir.endsWith('.app') ? dir : null
}

/** A quarantined app opened in place runs from a read-only copy under a fresh AppTranslocation folder. */
export function isAppTranslocated(target: string): boolean {
  return path.posix.resolve(target).includes('/AppTranslocation/')
}

// A bundle macOS can take away: a translocated copy is dropped on restart, a read-only volume on eject.
// EACCES (a root-owned folder) is left to Squirrel's admin prompt; only EROFS means a read-only mount.
export function ephemeralMacBundleLocation(
  target: string,
  access: (file: string, mode: number) => void = fs.accessSync
): MacEphemeralBundleLocation | null {
  if (isAppTranslocated(target)) {
    return 'app-translocation'
  }

  const normalized = path.posix.resolve(target)

  for (const candidate of [path.posix.dirname(normalized), normalized]) {
    try {
      access(candidate, fs.constants.W_OK)
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EROFS') {
        return 'read-only-volume'
      }
    }
  }

  return null
}
