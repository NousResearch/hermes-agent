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

/** PM publishes pinned tool packages under ``<hermes data root>/tools/``. */
const PM_STORE_DIR = 'tools'
/** PM's entry dir for the pinned git carries its version (`git-2.53.0+3-win32-x64`). */
const PM_GIT_ENTRY = /^git-\d/
/** ``binary_rel`` for pm's win32 git package (`pm/packages.py::Git`). */
const PM_GIT_REL = path.join('cmd', 'git.exe')

/** The dir prefix a UGit (https://github.com/ugit/UGit) install lives under. */
const UGIT_DIR = 'UGit'
/** UGit's Electron app dirs are versioned (`app-5.50.1`, …); git ships inside. */
const UGIT_APP_PREFIX = 'app-'
/** Where the UGit-bundled Git-for-Windows puts git.exe inside an app dir. */
const UGIT_GIT_REL = path.join('resources', 'app', 'git', 'cmd', 'git.exe')

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
 * Every `%LOCALAPPDATA%\hermes\tools\git-<version>-<target>\cmd\git.exe` on disk,
 * newest-first by version.
 *
 * `scripts/install.ps1`'s `Get-PinnedGit` provisions PortableGit into the PM store, and
 * `Ensure-Git` prepends it to the INSTALLER process's PATH only — deliberately, so a run
 * never inherits an unpinned system Git. Nothing is persisted to the machine or user PATH,
 * and the entry dir carries the pinned version, so no fixed candidate can name it:
 * enumerate the store, as the UGit glob does.
 *
 * This is the layout the Python side already resolves through pm
 * (`hermes_cli/boot_bootstrap.py::_git_binary` → `pm.installed_package("git")`). The list
 * below still named only the pre-PM location (`<data root>\git\{cmd,bin}`), so on a host
 * whose ONLY git is the pinned one every spawn ENOENTed, and the update check reported
 * "Could not read the installed revision" with `head: null` and `origin: ""` — no git had
 * run at all (#134600).
 *
 * Any published entry is a working git; newest-first picks what a current install selected.
 * A missing store returns [] — like the UGit glob, best-effort and never fatal.
 */
export function pmPinnedGitBinaries(localAppData: string, fs: GitCandidateFs): string[] {
  const store = path.join(localAppData, 'hermes', PM_STORE_DIR)

  let entries: string[]

  try {
    entries = fs.readdirSync(store)
  } catch {
    return []
  }

  return entries
    .filter(entry => PM_GIT_ENTRY.test(entry))
    .sort((a, b) => b.localeCompare(a, undefined, { numeric: true }))
    .map(entry => path.join(store, entry, PM_GIT_REL))
    .filter(fs.existsSync)
}

/**
 * resolveGitBinary's fixed Windows candidate list, in preference order: the
 * Hermes-pinned PortableGit first — the PM store slot, then the pre-PM
 * `hermes\git` location — then UGit's bundled copies, then the standard
 * Git-for-Windows locations.
 */
export function windowsGitCandidates(env: WindowsGitEnv, fs: GitCandidateFs): string[] {
  const candidates: string[] = []

  if (env.localAppData) {
    candidates.push(...pmPinnedGitBinaries(env.localAppData, fs))
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
