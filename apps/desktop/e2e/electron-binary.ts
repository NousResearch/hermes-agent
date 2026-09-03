/**
 * Locating the dev Electron binary for the e2e fixtures.
 *
 * Kept in its own module so the resolution rules can be unit-tested without
 * importing the Playwright runner (fixtures.ts pulls in `_electron`, the mock
 * server and the error-banner guard).
 *
 * Three rules the previous single-path probe got wrong:
 *
 *  1. The binary is not always under the REPO ROOT. This is an npm workspaces
 *     repo, and npm only hoists a dependency to the root when nothing conflicts
 *     — otherwise `electron` installs into `apps/desktop/node_modules`. Both
 *     layouts are normal, so both have to be searched, nearest package first.
 *  2. The binary is `electron.exe` on Windows. A bare `electron` never exists
 *     there, so the probe could only ever miss.
 *  3. `which` is not a command on Windows. The PATH fallback spawned it
 *     unconditionally, so on Windows the fallback failed for the wrong reason
 *     and the error message blamed a missing `npm install`.
 */

import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import { createRequire } from 'node:module'
import * as path from 'node:path'

/** The dist file name: `electron.exe` on Windows, `electron` elsewhere. */
export function electronBinaryName(platform: NodeJS.Platform = process.platform): string {
  return platform === 'win32' ? 'electron.exe' : 'electron'
}

/**
 * Where an npm install can leave the binary, in probe order: nearest package
 * first, so a workspace-local install wins over a stale hoisted one.
 */
export function electronDistCandidates(roots: string[], platform: NodeJS.Platform = process.platform): string[] {
  return roots.map((root) => path.join(root, 'node_modules', 'electron', 'dist', electronBinaryName(platform)))
}

/** The PATH-lookup command for this platform. Windows has `where`, not `which`. */
export function pathLookupCommand(platform: NodeJS.Platform = process.platform): string {
  return platform === 'win32' ? 'where' : 'which'
}

/**
 * Ask the installed `electron` package where its own binary is.
 *
 * Its main export IS the absolute executable path, resolved from `path.txt`
 * and honouring `ELECTRON_OVERRIDE_DIST_PATH`, so this covers layouts and
 * overrides a hand-built path cannot know about. Returns null when the package
 * is not resolvable from `from`, or when it does not hand back a path (the
 * export is the Electron API object, not a path, when required from inside
 * Electron itself).
 */
export function electronPackagePath(from: string): null | string {
  try {
    const resolved = createRequire(path.join(from, 'package.json'))('electron') as unknown

    return typeof resolved === 'string' && resolved ? resolved : null
  } catch {
    return null
  }
}

/**
 * Resolve the Electron binary, or throw with the layouts that were searched.
 *
 * `roots` are searched in order; pass the desktop package before the repo root.
 */
export function resolveElectronBinary(roots: string[]): string {
  for (const root of roots) {
    const declared = electronPackagePath(root)

    if (declared && fs.existsSync(declared)) {
      return declared
    }
  }

  for (const candidate of electronDistCandidates(roots)) {
    if (fs.existsSync(candidate)) {
      return candidate
    }
  }

  // Nix devshells put `electron` on PATH with no node_modules copy at all.
  const lookup = spawnSync(pathLookupCommand(), ['electron'], { encoding: 'utf8' })

  if (lookup.status === 0 && lookup.stdout.trim()) {
    // `where` reports every match, one per line; take the first.
    const first = lookup.stdout.trim().split(/\r?\n/)[0].trim()

    if (first) {
      return first
    }
  }

  throw new Error(
    `Electron binary not found. Searched ${electronDistCandidates(roots).join(', ')} and PATH. ` +
      'Run "npm install" from the repo root to install devDependencies.',
  )
}
