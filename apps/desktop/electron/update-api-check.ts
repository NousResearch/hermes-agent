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

/** A git command runner — main.ts injects its runGit; tests inject a fake. */
export type GitRunner = (args: string[], options?: { cwd?: string }) => Promise<{
  code: number
  stdout: string
  stderr: string
}>

/**
 * Answer "how far behind is HEAD?" from the LOCAL object database when the
 * compare endpoint can't — its 404 on a local-only HEAD (a patched checkout's
 * merge/rebase commits exist nowhere upstream) or a rate limit. Mirrors the
 * ls-remote path's ancestry guard and the backend's banner._tips_behind: a
 * tip reachable from HEAD is a local commit AHEAD, not an update; anything
 * else counts HEAD..tip. Returns null when the tip isn't in the local
 * database (a truly stale checkout) so callers keep the honest
 * "update available, count unknown" state instead of a fabricated number.
 */
export async function resolveBehindLocally(
  runGit: GitRunner,
  cwd: string,
  currentSha: string,
  targetSha: string
): Promise<number | null> {
  const known = await runGit(['cat-file', '-e', `${targetSha}^{commit}`], { cwd })

  if (known.code !== 0) {
    return null
  }

  if ((await runGit(['merge-base', '--is-ancestor', targetSha, currentSha], { cwd })).code === 0) {
    return 0
  }

  const count = await runGit(['rev-list', '--count', `${currentSha}..${targetSha}`], { cwd })
  const parsed = Number.parseInt(count.stdout.trim(), 10)

  return count.code === 0 && Number.isInteger(parsed) && parsed >= 0 ? parsed : null
}

/**
 * Newest-first commit list for resolveBehindLocally's count, in the same
 * CompareCommit shape as parseCompare so the overlay's "what's changed" list
 * renders the local answer too.
 */
export async function listLocalCommits(
  runGit: GitRunner,
  cwd: string,
  currentSha: string,
  targetSha: string
): Promise<CompareCommit[]> {
  const log = await runGit(
    ['log', '--reverse', '--date=iso-strict', '--pretty=%H%x1f%an%x1f%ad%x1f%s', `${currentSha}..${targetSha}`],
    { cwd }
  )

  if (log.code !== 0) {
    return []
  }

  return log.stdout
    .split('\n')
    .map(line => line.split('\x1f'))
    .filter(parts => parts.length === 4 && parts[0])
    .map(([sha, author, date, summary]) => ({
      sha,
      author,
      summary,
      at: Number.isFinite(Date.parse(date)) ? Date.parse(date) : 0
    }))
    .reverse()
}
