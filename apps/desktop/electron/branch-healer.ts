/**
 * Pure helpers for self-healing the desktop updater's tracked branch (#105042).
 *
 * If origin published a branch (e.g. a PR branch or shared feature branch) that was
 * merged into main and deleted on the remote, the desktop updater self-heals by
 * falling back to main and persisting that choice.
 *
 * However, exit code 2 from `git ls-remote --exit-code --heads <remote> <branch>`
 * only indicates that the ref is absent on the remote. That is true for a branch
 * deleted after merge, but it is equally true for a purely local branch that was
 * never pushed to origin.
 *
 * To avoid silently re-pinning a developer's local checkout and abandoning their
 * unmerged code:
 * 1. Verify that the branch was actually tracked remotely (has a remote-tracking
 *    ref like `refs/remotes/origin/<branch>` or an `@upstream` configured).
 *    If no remote tracking ref exists, it is a local-only branch: keep the pin.
 * 2. Verify that the branch carries no unmerged commits relative to origin/main
 *    (or main). If the branch carries local commits not yet in main, do not re-pin.
 *
 * Extracted so the healing logic can be unit-tested without booting Electron.
 */

export interface RunGitResult {
  code: number
  stdout: string
  stderr: string
}

export interface BranchHealerDeps {
  runGit: (args: string[], options?: { cwd?: string }) => Promise<RunGitResult>
  getOriginUrl: (updateRoot: string) => Promise<string>
  isOfficialSshRemote: (url: string) => boolean
  officialRepoHttpsUrl?: string
  rememberLog?: (msg: string) => void
  readDesktopUpdateConfig?: () => { branch: string }
  writeDesktopUpdateConfig?: (config: { branch: string }) => void
}

export async function resolveHealedBranch(
  deps: BranchHealerDeps,
  updateRoot: string,
  branch: string
): Promise<string> {
  if (!branch || branch === 'main') {
    return branch || 'main'
  }

  const {
    runGit,
    getOriginUrl,
    isOfficialSshRemote,
    officialRepoHttpsUrl = 'https://github.com/NousResearch/hermes-agent.git',
    rememberLog = () => {},
    readDesktopUpdateConfig = () => ({ branch: 'main' }),
    writeDesktopUpdateConfig = () => {}
  } = deps

  const originUrl = await getOriginUrl(updateRoot)
  const remote = isOfficialSshRemote(originUrl) ? officialRepoHttpsUrl : 'origin'
  const probe = await runGit(['ls-remote', '--exit-code', '--heads', remote, branch], { cwd: updateRoot })

  // Exit 2 means no matching ref on the remote. Any other non-zero code is an error
  // (e.g. transient network failure), which must never strand the user on the wrong branch.
  if (probe.code !== 2) {
    return branch
  }

  // Probe returned 2: the ref does not exist on remote.
  // Distinguish "deleted upstream after merge" from "never pushed / purely local branch":
  // Check if a remote-tracking ref or upstream exists for this branch.
  const remoteTracking = await runGit(['rev-parse', '--verify', '--quiet', `refs/remotes/origin/${branch}`], {
    cwd: updateRoot
  })
  const upstream = await runGit(['rev-parse', '--verify', '--quiet', `${branch}@{upstream}`], { cwd: updateRoot })
  const hasRemoteRef = remoteTracking.code === 0 || upstream.code === 0

  if (!hasRemoteRef) {
    rememberLog(`[updates] branch '${branch}' has no remote-tracking ref (never pushed); keeping branch pin`)
    return branch
  }

  // Branch was once tracked, but verify it carries no unmerged commits before re-pinning to main.
  let unmergedCheck = await runGit(['rev-list', '--count', `origin/main..${branch}`], { cwd: updateRoot })
  if (unmergedCheck.code !== 0) {
    unmergedCheck = await runGit(['rev-list', '--count', `main..${branch}`], { cwd: updateRoot })
  }

  if (unmergedCheck.code === 0) {
    const count = parseInt(unmergedCheck.stdout.trim(), 10)
    if (!isNaN(count) && count > 0) {
      rememberLog(`[updates] origin/${branch} is gone, but branch carries ${count} unmerged commit(s); keeping branch pin`)
      return branch
    }
  }

  rememberLog(`[updates] origin/${branch} is gone (merged?); falling back to main`)
  const config = readDesktopUpdateConfig()

  if (config.branch !== 'main') {
    writeDesktopUpdateConfig({ ...config, branch: 'main' })
  }

  return 'main'
}
