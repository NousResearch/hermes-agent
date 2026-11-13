import { execFileSync } from 'node:child_process'
import path from 'node:path'

import { selectRunnableBinary } from './select-runnable-binary'

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

/**
 * A binary can exist on disk and still be unlaunchable — on macOS an
 * Intel-only build ahead on PATH (e.g. a pre-Rosetta-removal Homebrew)
 * fails at spawn time with errno -86 (EBADARCH), which callers then report
 * as an update-server/network problem. On Windows the same class of failure
 * arrives as ERROR_ACCESS_DENIED when an update left the PM-pinned tool
 * unlaunchable. Probing `git --version` before committing to a candidate skips
 * such entries; the existence-only fallback keeps behaviour unchanged where
 * the probe itself cannot run.
 */
export function binaryRuns(candidate: string): boolean {
  try {
    execFileSync(candidate, ['--version'], {
      stdio: 'ignore',
      timeout: 5000,
      windowsHide: true
    })

    return true
  } catch {
    return false
  }
}

/** Every PATH entry that holds `command`, in PATH order. */
export function findPathCandidates(
  command: string,
  pathEntries: string,
  fileExists: (candidate: string) => boolean
): string[] {
  const candidates: string[] = []

  for (const entry of pathEntries.split(path.delimiter).filter(Boolean)) {
    const candidate = path.join(entry, command)

    if (fileExists(candidate)) {
      candidates.push(candidate)
    }
  }

  return candidates
}

/**
 * The Windows git to spawn: the first candidate that exists AND runs, else the
 * first that merely exists, else null.
 *
 * The PM-pinned git leads the list, and an update can leave it present but
 * unlaunchable — extraction replaces the tool tree's inherited ACLs with the
 * archive's own metadata, so the interactive user's token cannot spawn it
 * (ERROR_ACCESS_DENIED) while a system Git further down the list works.
 * Selecting by existence alone commits to that broken copy, and the update
 * check then answers `head-unavailable`, which the UI reports as an unreachable
 * update server.
 */
export function selectWindowsGitBinary(opts: {
  env: WindowsGitEnv
  fs: GitCandidateFs
  /** Probe: does this candidate actually execute (`<binary> --version`)? */
  binaryRuns: (candidate: string) => boolean
}): string | null {
  return selectRunnableBinary({
    candidates: windowsGitCandidates(opts.env, opts.fs),
    fileExists: opts.fs.existsSync,
    binaryRuns: opts.binaryRuns
  })
}
