import path from 'node:path'

export interface GitBashOptions {
  isWindows: boolean
  env: Record<string, string | undefined>
  fileExists: (filePath: string) => boolean
  findOnPath?: (command: string) => string | null
  /**
   * The executable to locate (e.g. 'bash.exe' or 'git.exe'). Controls
   * which override env var is consulted and which subdirectories are
   * probed under Git installations. Defaults to 'bash.exe' for backward
   * compatibility.
   */
  executable?: string
}

/**
 * Locate bash.exe (or git.exe when executable:'git.exe') on Windows (and bash/git on POSIX).
 * This unifies discovery for both HERMES_GIT_BASH_PATH and HERMES_GIT_EXE_PATH.
 *
 * Resolution order (first match wins):
 *   1. The appropriate HERMES_GIT_*_PATH env var override (BASH for bash.exe, EXE for git.exe)
 *   2. PortableGit under %LOCALAPPDATA%\hermes\git\ (install.ps1)
 *   3. Standard Git for Windows install locations
 *   4. %LOCALAPPDATA%\Programs\Git\ (user-scoped)
 *   5. <name> on PATH (via injected findOnPath)
 */
export function findGitBash(opts: GitBashOptions): string | null {
  const { isWindows, env, fileExists, findOnPath } = opts
  const exe = opts.executable || 'bash.exe'
  const name = exe.replace(/\.exe$/i, '') // 'bash' or 'git' for PATH and docs

  if (!isWindows) {
    return findOnPath ? findOnPath(name) : null
  }

  // Respect the matching override env var (mirrors tools/environments/local.py:_find_bash for bash;
  // HERMES_GIT_EXE_PATH follows the exact same pattern for git.exe).
  const overrideVar = name === 'git' ? 'HERMES_GIT_EXE_PATH' : 'HERMES_GIT_BASH_PATH'
  const custom = env[overrideVar]

  if (custom && fileExists(custom)) {
    return custom
  }

  const localAppData = env.LOCALAPPDATA || ''
  const candidates: string[] = []

  // Candidate paths are Windows paths regardless of host platform (tests run
  // on POSIX CI hosts too), so join with win32 semantics explicitly.
  const joinWin = path.win32.join

  const programFiles = env['ProgramFiles'] || 'C:\\Program Files'
  const programFilesX86 = env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)'

  if (localAppData) {
    // hermes-managed PortableGit layout differs by binary:
    //   git.exe lives under cmd/ (and bin/ for some layouts)
    //   bash.exe lives under bin/ (and usr/bin/ for MinGit)
    candidates.push(joinWin(localAppData, 'hermes', 'git', 'bin', exe))
    if (name === 'git') {
      candidates.push(joinWin(localAppData, 'hermes', 'git', 'cmd', exe))
    } else {
      candidates.push(joinWin(localAppData, 'hermes', 'git', 'usr', 'bin', exe))
    }
  }

  candidates.push(joinWin(programFiles, 'Git', 'bin', exe))
  candidates.push(joinWin(programFiles, 'Git', 'cmd', exe))
  candidates.push(joinWin(programFilesX86, 'Git', 'bin', exe))
  candidates.push(joinWin(programFilesX86, 'Git', 'cmd', exe))

  if (localAppData) {
    candidates.push(joinWin(localAppData, 'Programs', 'Git', 'bin', exe))
    candidates.push(joinWin(localAppData, 'Programs', 'Git', 'cmd', exe))
  }

  for (const candidate of candidates) {
    if (fileExists(candidate)) {
      return candidate
    }
  }

  if (findOnPath) {
    const onPath = findOnPath(name)

    if (onPath) {
      return onPath
    }
  }

  return null
}
