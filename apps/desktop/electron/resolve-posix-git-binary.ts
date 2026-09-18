import path from 'node:path'

export interface ResolvePosixGitBinaryOptions {
  pathEnv: string | undefined
  fileExists: (filePath: string) => boolean
  canExecute: (filePath: string) => boolean
}

/**
 * Pick the `git` to use on non-Windows platforms.
 *
 * resolveGitBinary()'s non-Windows branch used to be `findOnPath('git') ||
 * 'git'`, which returns the first PATH entry where a `git` file merely
 * *exists* — not the same as one that can run. After a macOS OS upgrade
 * without Rosetta, a leftover Intel `git` earlier on PATH than a working
 * native one still exists, so it wins, and fails at spawn time with
 * "Bad CPU type in executable" (Darwin errno 86). Every runGit() call then
 * reports the update server as unreachable instead of naming the real cause
 * (#114718).
 *
 * Prefer the first PATH entry that exists AND runs; fall back to the first
 * that merely exists so behaviour is unchanged when the probe itself cannot
 * run (e.g. a sandboxed spawn).
 */
export function resolvePosixGitBinary(opts: ResolvePosixGitBinaryOptions): string | null {
  const { pathEnv, fileExists, canExecute } = opts

  const candidates = String(pathEnv || '')
    .split(':')
    .filter(Boolean)
    .map(dir => path.posix.join(dir, 'git'))
    .filter(fileExists)

  return candidates.find(canExecute) || candidates[0] || null
}
