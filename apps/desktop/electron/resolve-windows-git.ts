import { readFileSync } from 'node:fs'
import path from 'node:path'

/** Read PM's selected entry; never guess from inactive or partially staged tools. */
function managedGitCandidates(localAppData: string): string[] {
  const store = path.join(localAppData, 'hermes', 'tools')
  try {
    const facts = JSON.parse(readFileSync(path.join(store, 'facts.json'), 'utf8'))
    const entry = facts?.packages?.git?.entry
    if (
      facts?.schema !== 1 ||
      typeof entry !== 'string' ||
      !entry ||
      entry === '.' ||
      entry === '..' ||
      /[\\/\\\\:]/.test(entry)
    ) {
      return []
    }
    return ['cmd', 'bin'].map(folder => path.join(store, entry, folder, 'git.exe'))
  } catch {
    // Missing, unreadable or corrupt facts must not hide a working system Git.
    return []
  }
}

export interface WindowsGitOptions {
  env: Record<string, string | undefined>
  fileExists: (candidate: string) => boolean
  findOnPath: (command: string) => string | null
}

/** Windows git discovery shared by update checks and the desktop Git IPCs. */
export function resolveWindowsGit({ env, fileExists, findOnPath }: WindowsGitOptions): string {
  const localAppData = env.LOCALAPPDATA || ''
  const candidates: string[] = []

  if (localAppData) {
    candidates.push(path.join(localAppData, 'hermes', 'git', 'cmd', 'git.exe'))
    candidates.push(path.join(localAppData, 'hermes', 'git', 'bin', 'git.exe'))
    candidates.push(...managedGitCandidates(localAppData))
  }

  candidates.push(path.join(env['ProgramFiles'] || 'C:\\Program Files', 'Git', 'cmd', 'git.exe'))
  candidates.push(path.join(env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'cmd', 'git.exe'))

  if (localAppData) {
    candidates.push(path.join(localAppData, 'Programs', 'Git', 'cmd', 'git.exe'))
  }

  return candidates.find(fileExists) || findOnPath('git') || 'git'
}
