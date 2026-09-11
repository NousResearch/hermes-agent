import * as fs from 'node:fs'
import * as path from 'node:path'

import type { InstallStamp } from '../install-stamp'
import { branchTipApiUrl, cacheIsFresh, compareApiUrl, githubRepoSlug, parseCompare } from '../update-api-check'
import { classifyUpdateRoot } from '../update-root-policy'

import type { UpdaterStatusWire } from './index'

export interface CheckoutCheckDeps {
  writeFileAtomic: (filePath: string, contents: string) => void
  updateCheckCachePath: string
  isGitCheckout: (root: string) => boolean
  readCanonicalInstallStamp: () => { updateMechanism?: InstallStamp['updateMechanism'] } | null
  readDesktopUpdateConfig: () => { branch: string }
  resolveUpdateRoot: () => string
  resolveHealedBranch: (root: string, branch: string) => Promise<string>
  getOriginUrl: (root: string) => Promise<string>
  runGit: (args: string[], options?: { cwd?: string }) => Promise<{ code: number; stdout: string; stderr: string }>
  fetchGitHubApi: (url: string, accept?: string) => Promise<unknown>
  rememberLog: (chunk: unknown) => void
}

interface CachedCheckoutCheck {
  fetchedAt: number
  currentSha: string
  branch: string
  originUrl: string
  updateRoot: string
  status: UpdaterStatusWire & Record<string, unknown>
}

function readCache(filePath: string): CachedCheckoutCheck | null {
  try {
    const cached: CachedCheckoutCheck = JSON.parse(fs.readFileSync(filePath, 'utf8'))

    return cached?.status?.supported === true ? cached : null
  } catch {
    return null
  }
}

async function checkApi(
  deps: CheckoutCheckDeps,
  slug: string,
  branch: string,
  currentSha: string
): Promise<Partial<UpdaterStatusWire>> {
  let targetSha: string

  try {
    targetSha = String(await deps.fetchGitHubApi(branchTipApiUrl(slug, branch), 'application/vnd.github.sha')).trim()
  } catch (error: unknown) {
    return { error: 'fetch-failed', message: `GitHub API: ${error instanceof Error ? error.message : String(error)}` }
  }

  if (!/^[0-9a-f]{40}$/i.test(targetSha)) {
    return { error: 'fetch-failed', message: 'GitHub API returned no tip SHA.' }
  }

  if (targetSha === currentSha) {
    return { behind: 0, updateAvailable: false, targetSha, commits: [] }
  }

  const compared = await deps
    .fetchGitHubApi(compareApiUrl(slug, currentSha, targetSha))
    .then(parseCompare)
    .catch((): null => null)

  return {
    behind: compared?.behind ?? null,
    updateAvailable: compared?.behind !== 0,
    targetSha,
    commits: compared?.behind === 0 ? [] : (compared?.commits ?? [])
  }
}

async function checkLsRemote(
  deps: CheckoutCheckDeps,
  updateRoot: string,
  branch: string,
  currentSha: string
): Promise<Partial<UpdaterStatusWire>> {
  const target = await deps.runGit(['ls-remote', 'origin', `refs/heads/${branch}`], { cwd: updateRoot })
  const targetSha = target.stdout.trim().split(/\s+/)[0] || ''

  if (target.code !== 0 || !targetSha) {
    return { error: 'fetch-failed', message: target.stderr.split('\n')[0] || 'git ls-remote failed.' }
  }

  if (targetSha === currentSha) {
    return { behind: 0, updateAvailable: false, targetSha, commits: [] }
  }

  const known = (await deps.runGit(['cat-file', '-e', `${targetSha}^{commit}`], { cwd: updateRoot })).code === 0

  const isAncestor =
    known && (await deps.runGit(['merge-base', '--is-ancestor', targetSha, 'HEAD'], { cwd: updateRoot })).code === 0

  return { behind: isAncestor ? 0 : null, updateAvailable: !isAncestor, targetSha, commits: [] }
}

/** Passive checks must not fetch packs or mutate a steward-owned checkout. */
export async function checkCheckoutUpdates(
  deps: CheckoutCheckDeps,
  { force = false }: { force?: boolean } = {}
): Promise<UpdaterStatusWire> {
  const updateRoot = deps.resolveUpdateRoot()
  let { branch } = deps.readDesktopUpdateConfig()

  const policy = classifyUpdateRoot({
    isGitTree: deps.isGitCheckout(updateRoot),
    updateMechanism: deps.readCanonicalInstallStamp()?.updateMechanism ?? null
  })

  if (!policy.updatable) {
    return {
      supported: false,
      reason: policy.verdict === 'not-a-checkout' ? 'not-a-git-checkout' : `update-root-${policy.verdict}`,
      message: policy.message,
      advice: policy.advice,
      hermesRoot: updateRoot,
      branch
    }
  }

  const git = async (args: string[]): Promise<string> => (await deps.runGit(args, { cwd: updateRoot })).stdout.trim()

  const [currentSha, dirty, currentBranch, originUrl] = await Promise.all([
    git(['rev-parse', 'HEAD']),
    git(['status', '--porcelain']),
    git(['rev-parse', '--abbrev-ref', 'HEAD']),
    deps.getOriginUrl(updateRoot)
  ])

  const cached = readCache(deps.updateCheckCachePath)
  const now = Date.now()

  if (
    !force &&
    cached?.updateRoot === updateRoot &&
    cached.originUrl === originUrl &&
    cacheIsFresh(cached, { branch, currentSha, now })
  ) {
    return { ...cached.status, dirty: dirty.length > 0, currentBranch }
  }

  branch = await deps.resolveHealedBranch(updateRoot, branch)
  const slug = githubRepoSlug(originUrl)

  const status = slug
    ? await checkApi(deps, slug, branch, currentSha)
    : await checkLsRemote(deps, updateRoot, branch, currentSha)

  const result: UpdaterStatusWire & Record<string, unknown> = {
    supported: true,
    branch,
    currentBranch,
    currentSha,
    dirty: dirty.length > 0,
    hermesRoot: updateRoot,
    fetchedAt: now,
    ...status
  }

  try {
    fs.mkdirSync(path.dirname(deps.updateCheckCachePath), { recursive: true })
    const entry: CachedCheckoutCheck = { fetchedAt: now, currentSha, branch, originUrl, updateRoot, status: result }
    deps.writeFileAtomic(deps.updateCheckCachePath, JSON.stringify(entry))
  } catch (error: unknown) {
    deps.rememberLog(
      `[updates] could not persist check cache: ${error instanceof Error ? error.message : String(error)}`
    )
  }

  return result
}
