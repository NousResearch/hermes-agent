/**
 * windows-short-path.ts
 *
 * Convert a Windows path to its 8.3 short-path form.
 *
 * simple-git's customBinaryPlugin validates the binary path with a regex that
 * does not allow spaces. Git-for-Windows is installed under "C:\Program Files"
 * by default, so the resolved absolute path must be converted to its short form
 * before being passed to simple-git.
 */

import { execFileSync } from 'node:child_process'

export interface ToShortPathDeps {
  execFileSync?: typeof execFileSync
}

/**
 * Convert a Windows path to its 8.3 short-path form.
 *
 * Uses cmd.exe's for-variable expansion (`%~sA`) so the result contains no
 * spaces and passes simple-git's isBadArgument regex. Falls back to the
 * original path if the lookup fails or returns the same path.
 */
export function toShortPath(filePath: string, deps: ToShortPathDeps = {}): string {
  const run = deps.execFileSync ?? execFileSync

  try {
    const result = String(
      run(
        'cmd.exe',
        ['/c', `for %A in ("${filePath}") do @echo %~sA`],
        { timeout: 5000, windowsHide: true, encoding: 'utf8' }
      )
    ).trim()

    if (result && result !== filePath) {
      return result
    }
  } catch {
    // fall through to the original path
  }

  return filePath
}
