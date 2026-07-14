import path from 'node:path'

export interface GitBashOptions {
  isWindows: boolean
  env: Record<string, string | undefined>
  fileExists: (filePath: string) => boolean
  findOnPath?: (command: string) => string | null
}

/**
 * Locate bash.exe on Windows.
 * Resolution order (first match wins):
 *   1. HERMES_GIT_BASH_PATH env var override
 *   2. PortableGit under %LOCALAPPDATA%\hermes\git\ (install.ps1)
 *   3. Standard Git for Windows install locations
 *   4. %LOCALAPPDATA%\Programs\Git\ (user-scoped)
 *   5. bash on PATH
 */
export function findGitBash(opts: GitBashOptions): string | null {
  const { isWindows, env, fileExists, findOnPath } = opts

  if (!isWindows) {
    return findOnPath ? findOnPath('bash') : null
  }

  // Respect HERMES_GIT_BASH_PATH if set (mirrors tools/environments/local.py:_find_bash).
  const gitBashPath = env.HERMES_GIT_BASH_PATH
  if (gitBashPath && fileExists(gitBashPath)) return gitBashPath

  const localAppData = env.LOCALAPPDATA || ''
  const candidates: string[] = []

  if (localAppData) {
    candidates.push(path.join(localAppData, 'hermes', 'git', 'bin', 'bash.exe'))
    candidates.push(path.join(localAppData, 'hermes', 'git', 'usr', 'bin', 'bash.exe'))
  }

  candidates.push(path.join(env['ProgramFiles'] || 'C:\\Program Files', 'Git', 'bin', 'bash.exe'))
  candidates.push(path.join(env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'bin', 'bash.exe'))

  if (localAppData) {
    candidates.push(path.join(localAppData, 'Programs', 'Git', 'bin', 'bash.exe'))
  }

  for (const candidate of candidates) {
    if (fileExists(candidate)) return candidate
  }

  if (findOnPath) {
    const onPath = findOnPath('bash')
    if (onPath) return onPath
  }

  return null
}
