import path from 'node:path'

/** The filesystem surface the candidate builders need — injectable for tests. */
export interface GitCandidateFs {
  existsSync: (candidate: string) => boolean
  readdirSync: (dir: string) => string[]
}

/** Windows env slice resolveGitBinary reads (injectable for tests). */
export interface WindowsGitEnv {
  localAppData: string
  programFiles: string
  programFilesX86: string
}

/** The dir prefix a UGit (https://github.com/ugit/UGit) install lives under. */
const UGIT_DIR = 'UGit'
/** UGit's Electron app dirs are versioned (`app-5.50.1`, …); git ships inside. */
const UGIT_APP_PREFIX = 'app-'
/** Where the UGit-bundled Git-for-Windows puts git.exe inside an app dir. */
const UGIT_GIT_REL = path.join('resources', 'app', 'git', 'cmd', 'git.exe')
/** PM-managed Git tool directories under the Hermes local application root. */
const HERMES_TOOLS_DIR = path.join('hermes', 'tools')
const MANAGED_GIT_PREFIX = 'git-'

/**
 * Every `%LOCALAPPDATA%\UGit\app-*\resources\app\git\cmd\git.exe` on disk,
 * sorted newest-first by version (descending dir name).
 *
 * UGit bundles its own Git-for-Windows copy under a versioned app dir, so the
 * exact path moves with every update — no fixed candidate can name it. The
 * path IS usually on the user's PATH, but an Electron process launched from
 * Explorer inherits the login-time environment block, which can lack entries
 * added later by the UGit installer, so the update check's `git` spawn
 * ENOENTs and "Check for updates" fails (#61494).
 *
 * Dirs whose bundled git.exe is missing are skipped (an app dir mid-update);
 * a missing UGit dir returns [] — the glob is best-effort, never fatal.
 */
export function ugitGitBinaries(localAppData: string, fs: GitCandidateFs): string[] {
  const ugitRoot = path.join(localAppData, UGIT_DIR)

  let entries: string[]

  try {
    entries = fs.readdirSync(ugitRoot)
  } catch {
    return []
  }

  return entries
    .filter(entry => entry.startsWith(UGIT_APP_PREFIX))
    .sort((a, b) => b.localeCompare(a, undefined, { numeric: true }))
    .map(entry => path.join(ugitRoot, entry, UGIT_GIT_REL))
    .filter(fs.existsSync)
}


/**
 * Find Git installed by Hermes PM under `%LOCALAPPDATA%\hermes\tools`.
 * PM versions the tool directory, so resolve the newest available executable
 * rather than assuming a particular Git release.
 */
export function managedGitBinaries(localAppData: string, fs: GitCandidateFs): string[] {
  const toolsRoot = path.join(localAppData, HERMES_TOOLS_DIR)

  let entries: string[]

  try {
    entries = fs.readdirSync(toolsRoot)
  } catch {
    return []
  }

  return entries
    .filter(entry => entry.startsWith(MANAGED_GIT_PREFIX))
    .sort((a, b) => b.localeCompare(a, undefined, { numeric: true }))
    .map(entry => path.join(toolsRoot, entry, 'cmd', 'git.exe'))
    .filter(fs.existsSync)
}

/**
 * resolveGitBinary's fixed Windows candidate list, in preference order:
 * the Hermes-bundled PortableGit first, then UGit's bundled copies, then the
 * standard Git-for-Windows locations.
 */
export function windowsGitCandidates(env: WindowsGitEnv, fs: GitCandidateFs): string[] {
  const candidates: string[] = []

  if (env.localAppData) {
    candidates.push(...managedGitBinaries(env.localAppData, fs))
    candidates.push(path.join(env.localAppData, 'hermes', 'git', 'cmd', 'git.exe'))
    candidates.push(path.join(env.localAppData, 'hermes', 'git', 'bin', 'git.exe'))
    candidates.push(...ugitGitBinaries(env.localAppData, fs))
  }

  candidates.push(path.join(env.programFiles, 'Git', 'cmd', 'git.exe'))
  candidates.push(path.join(env.programFilesX86, 'Git', 'cmd', 'git.exe'))

  if (env.localAppData) {
    candidates.push(path.join(env.localAppData, 'Programs', 'Git', 'cmd', 'git.exe'))
  }

  return candidates
}
