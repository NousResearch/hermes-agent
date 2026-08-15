import path from 'node:path'

export interface GitBinaryOptions {
  isWindows: boolean
  env: Record<string, string | undefined>
  fileExists: (filePath: string) => boolean
  /** Probe: does this candidate actually execute (`git --version`)? */
  binaryRuns: (filePath: string) => boolean
  findOnPath?: (command: string) => string | null
}

/**
 * Choose which git executable to use.
 *
 * On Windows a candidate can exist on disk and still be unusable: install.ps1
 * lays down `%LOCALAPPDATA%\hermes\git\cmd\git.exe`, and if the PortableGit
 * payload (`mingw64\`) is missing or half-extracted, spawning it fails with
 *
 *   error launching git: The system cannot find the path specified.
 *
 * which surfaces as "Couldn't check for updates. Check your connection and
 * try again." — blaming the network on a machine with a perfectly good Git
 * for Windows one entry further down the list. So prefer a candidate that
 * both exists *and* runs.
 *
 * Fall back to the first candidate that merely exists when no probe
 * succeeds, so behaviour is unchanged where the probe itself cannot run
 * (locked-down execution policy, AV interposing on spawn) rather than
 * skipping a git that would have worked.
 *
 * Resolution order (first match wins):
 *   1. a candidate that exists and runs
 *   2. a candidate that merely exists
 *   3. git on PATH
 *   4. bare 'git'
 */
export function selectGitBinary(opts: GitBinaryOptions): string {
  const { isWindows, env, fileExists, binaryRuns, findOnPath } = opts

  if (!isWindows) {
    return (findOnPath ? findOnPath('git') : null) || 'git'
  }

  const localAppData = env.LOCALAPPDATA || ''
  const candidates: string[] = []

  // Candidate paths are Windows paths regardless of host platform (tests run
  // on POSIX CI hosts too), so join with win32 semantics explicitly.
  const joinWin = path.win32.join

  if (localAppData) {
    candidates.push(joinWin(localAppData, 'hermes', 'git', 'cmd', 'git.exe'))
    candidates.push(joinWin(localAppData, 'hermes', 'git', 'bin', 'git.exe'))
  }

  candidates.push(joinWin(env['ProgramFiles'] || 'C:\\Program Files', 'Git', 'cmd', 'git.exe'))
  candidates.push(joinWin(env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'cmd', 'git.exe'))

  if (localAppData) {
    candidates.push(joinWin(localAppData, 'Programs', 'Git', 'cmd', 'git.exe'))
  }

  return (
    candidates.find(c => fileExists(c) && binaryRuns(c)) ||
    candidates.find(fileExists) ||
    (findOnPath ? findOnPath('git') : null) ||
    'git'
  )
}
