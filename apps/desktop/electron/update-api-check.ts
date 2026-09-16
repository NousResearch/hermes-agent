/**
 * Passive update checks against the GitHub REST API instead of git.
 *
 * Every desktop client used to run `git fetch origin <branch>` (or `ls-remote`)
 * twice every 30 minutes, plus on each window focus. Multiplied across the
 * install base that is tens of millions of pack negotiations a day against one
 * repo — GitHub flagged it. A passive check only needs two facts the API gives
 * for free: the remote tip SHA (`GET /repos/{repo}/commits/{branch}` with the
 * `application/vnd.github.sha` media type — a 40-byte body) and, when the tips
 * differ, the compare endpoint's `ahead_by` + `commits[]`. `git fetch` now
 * runs only when the user actually applies an update.
 *
 * Pure helpers here (URL builders, cache policy, payload mapping) so they are
 * unit-testable without booting Electron; the bounded network call is injected.
 */

import { canonicalGitHubRemote } from './update-remote'

export const UPDATE_CHECK_TTL_MS = 24 * 60 * 60 * 1000
// A failed check (offline, 403 rate-limit) is retried sooner than a good one,
// but never on every poller tick.
export const UPDATE_CHECK_FAILURE_TTL_MS = 60 * 60 * 1000

export interface CachedUpdateCheck {
  fetchedAt: number
  currentSha: string
  branch: string
  status: Record<string, unknown> & { error?: string }
}

/** `owner/repo` for any GitHub remote form; null for non-GitHub origins. */
export function githubRepoSlug(originUrl: string): string | null {
  const canonical = canonicalGitHubRemote(originUrl)
  const match = /^github\.com\/([^/]+\/[^/]+)$/.exec(canonical)

  return match ? match[1] : null
}

export function branchTipApiUrl(slug: string, branch: string): string {
  return `https://api.github.com/repos/${slug}/commits/${encodeURIComponent(branch)}`
}

export function compareApiUrl(slug: string, currentSha: string, targetSha: string): string {
  return `https://api.github.com/repos/${slug}/compare/${currentSha}...${targetSha}`
}

/**
 * Whether a cached result still answers a passive check. The cache is keyed on
 * the local HEAD and branch: applying an update or switching branches changes
 * HEAD and invalidates it immediately, so a 24h TTL never shows a stale
 * "update available" after the user just updated.
 */
export function cacheIsFresh(
  cached: CachedUpdateCheck | null | undefined,
  { branch, currentSha, now }: { branch: string; currentSha: string; now: number }
): boolean {
  if (!cached || cached.branch !== branch || cached.currentSha !== currentSha) {
    return false
  }

  const ttl = cached.status.error ? UPDATE_CHECK_FAILURE_TTL_MS : UPDATE_CHECK_TTL_MS

  return now - cached.fetchedAt < ttl
}

export interface CompareCommit {
  sha: string
  summary: string
  author: string
  at: number
}

/**
 * Map the compare payload to the shape the update overlay renders. `ahead_by`
 * is how far the remote tip is ahead of local HEAD, i.e. the behind count; 0
 * with differing tips means local carries commits on top of origin (not
 * behind). Any shape surprise returns null so callers keep the honest
 * "update available, count unknown" state instead of trusting a partial answer.
 */
export function parseCompare(payload: unknown): { behind: number; commits: CompareCommit[] } | null {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const ahead = (payload as { ahead_by?: unknown }).ahead_by

  if (typeof ahead !== 'number' || !Number.isInteger(ahead) || ahead < 0) {
    return null
  }

  const raw = (payload as { commits?: unknown }).commits

  const commits: CompareCommit[] = Array.isArray(raw)
    ? raw
        .map(entry => {
          const sha = typeof entry?.sha === 'string' ? entry.sha : ''
          const message = typeof entry?.commit?.message === 'string' ? entry.commit.message : ''
          const author = typeof entry?.commit?.author?.name === 'string' ? entry.commit.author.name : ''

          const date =
            typeof entry?.commit?.committer?.date === 'string' ? Date.parse(entry.commit.committer.date) : NaN

          return { sha, summary: message.split('\n')[0], author, at: Number.isFinite(date) ? date : 0 }
        })
        .filter(commit => commit.sha)
        // The overlay lists newest first; compare returns oldest first.
        .reverse()
    : []

  return { behind: ahead, commits }
}

export interface GitHubTokenSources {
  /** Environment to read PAT vars from; injected so tests never touch the real one. */
  env?: NodeJS.ProcessEnv
  /** Resolves a token via the gh CLI; injected so tests never spawn a process. */
  ghAuthToken?: () => Promise<string | null>
}

/**
 * Resolve a GitHub API token for passive update checks, mirroring the skills
 * hub's priority order: GITHUB_TOKEN / GH_TOKEN, then `gh auth token`, then
 * anonymous. Anonymous requests share one 60 req/hr per-IP bucket, which shared
 * and datacenter egress IPs exhaust constantly, so every check behind them 403s
 * (#108804); an authenticated check gets the per-user 5,000 req/hr budget. A
 * token must never surface in logs or error copy: failure to resolve one simply
 * means the request goes out anonymous, exactly as before.
 */
export async function resolveGitHubApiToken({
  env = process.env,
  ghAuthToken
}: GitHubTokenSources = {}): Promise<string | null> {
  const pat = (env.GITHUB_TOKEN || env.GH_TOKEN || '').trim()

  if (pat) {
    return pat
  }

  try {
    const fromGh = ((await ghAuthToken?.()) ?? '').trim()

    return fromGh || null
  } catch {
    return null
  }
}
