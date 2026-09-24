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
  /** When the release-anchor lookup last ran (release mode only). */
  releaseTagFetchedAt?: number
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
 * Release-anchored checks (`updateRefAnchor: 'release'`) compare the local HEAD
 * to the latest *tagged release* instead of the branch tip. Upstream main moves
 * hundreds of commits a day, so a tip-anchored badge is effectively always on
 * for fork installs; a release-anchored badge lights only when a new release
 * exists that HEAD lacks, and clears again once an update merges it.
 *
 * The anchor (which release is latest) must itself stay fresh even while the
 * 24h status cache would otherwise hold — otherwise a release published mid-day
 * stays invisible until HEAD moves. So in release mode the cache also expires
 * RELEASE_TAG_TTL_MS after the last anchor lookup.
 */
export const RELEASE_TAG_TTL_MS = 2 * 60 * 60 * 1000

export function latestReleaseApiUrl(slug: string): string {
  return `https://api.github.com/repos/${slug}/releases/latest`
}

/**
 * Latest release payload → its tag name, or null. Only version-shaped names
 * ("v2026.9.21", "v0.21.4", "2026.9.21") anchor a check; anything else (or a
 * missing/preview payload) falls the caller back to the branch tip.
 */
export function parseReleaseTagName(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const tag = (payload as { tag_name?: unknown }).tag_name

  if (typeof tag !== 'string' || tag.length === 0) {
    return null
  }

  const name = tag.replace(/^refs\/tags\//, '')

  return /^v?\d+(\.\d+)*$/.test(name) ? name : null
}

/**
 * Whether a cached result still answers a passive check. The cache is keyed on
 * the local HEAD and branch: applying an update or switching branches changes
 * HEAD and invalidates it immediately, so a 24h TTL never shows a stale
 * "update available" after the user just updated.
 */
export function cacheIsFresh(
  cached: CachedUpdateCheck | null | undefined,
  {
    branch,
    currentSha,
    now,
    releaseAnchor
  }: { branch: string; currentSha: string; now: number; releaseAnchor?: 'branch' | 'release' }
): boolean {
  if (!cached || cached.branch !== branch || cached.currentSha !== currentSha) {
    return false
  }

  const ttl = cached.status.error ? UPDATE_CHECK_FAILURE_TTL_MS : UPDATE_CHECK_TTL_MS

  if (now - cached.fetchedAt >= ttl) {
    return false
  }

  // Release-anchored mode must notice a release published after the last check
  // even while the 24h status cache would otherwise hold — expire once the
  // anchor lookup ages past RELEASE_TAG_TTL_MS so the next check re-resolves it.
  if (releaseAnchor === 'release' && now - (cached.releaseTagFetchedAt ?? 0) >= RELEASE_TAG_TTL_MS) {
    return false
  }

  return true
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

/** GitHub's rate-limit headers off a failed response, when it carried them. */
export function rateLimitFromHeaders(headers: Record<string, string | string[] | undefined>): {
  rateLimitRemaining: number | null
  rateLimitReset: number | null
} {
  const num = (name: string) => {
    const raw = headers[name]
    const value = Number.parseInt(Array.isArray(raw) ? raw[0] : String(raw ?? ''), 10)

    return Number.isFinite(value) ? value : null
  }

  return { rateLimitRemaining: num('x-ratelimit-remaining'), rateLimitReset: num('x-ratelimit-reset') }
}

export interface UpdateCheckFailure {
  statusCode?: number
  code?: string
  message?: string
  /** `x-ratelimit-remaining` on the failed response; 0 marks a genuine rate limit. */
  rateLimitRemaining?: number | null
  /** `x-ratelimit-reset` (unix seconds) on the failed response. */
  rateLimitReset?: number | null
  /** Whether the failed request carried a GITHUB_TOKEN / GH_TOKEN credential. */
  authenticated?: boolean
}

/**
 * One line a user can act on (or paste into a bug report) instead of the
 * generic "couldn't reach the update server": which host, which failure.
 * #105855 was a run of GitHub outages that read as a Hermes bug because the
 * UI hid the cause.
 *
 * A 403 with `x-ratelimit-remaining: 0` is the anonymous per-IP budget spent
 * by every client behind one exit (office NAT, VPN, proxy) — waiting an hour
 * does not help when the neighbours refill it, so the line names the fix the
 * user actually has (a GITHUB_TOKEN in the environment) and the real reset
 * time. Any other 403/429 is reported as what it is rather than as a rate
 * limit the user did not cause.
 */
export function describeUpdateCheckFailure(error: UpdateCheckFailure | null | undefined, now = Date.now()): string {
  const status = error?.statusCode
  const code = error?.code

  if ((status === 403 || status === 429) && error?.rateLimitRemaining === 0) {
    const resetMs = typeof error.rateLimitReset === 'number' ? error.rateLimitReset * 1000 - now : NaN
    const minutes = Number.isFinite(resetMs) && resetMs > 0 ? Math.ceil(resetMs / 60_000) : null

    const when =
      minutes === null ? 'within an hour' : minutes === 1 ? 'in about a minute' : `in about ${minutes} minutes`

    return error.authenticated
      ? `GitHub API rate limit reached for your GITHUB_TOKEN (HTTP ${status}) — it resets ${when}.`
      : `GitHub API rate limit reached (HTTP ${status}): anonymous requests are limited to 60 per hour per network ` +
          `address, shared with everyone behind the same connection. It resets ${when}; ` +
          `setting GITHUB_TOKEN in the environment lifts the limit.`
  }

  if (typeof status === 'number' && status >= 500) {
    return `GitHub is having trouble (HTTP ${status} from api.github.com) — check githubstatus.com and try again later.`
  }

  if (typeof status === 'number') {
    return `api.github.com answered HTTP ${status}.`
  }

  if (code === 'ENOTFOUND' || code === 'EAI_AGAIN') {
    return 'DNS lookup for api.github.com failed — check your connection or proxy.'
  }

  if (code === 'ETIMEDOUT' || error?.message === 'timeout') {
    return 'api.github.com did not answer within 10 seconds.'
  }

  if (code === 'ECONNREFUSED' || code === 'ECONNRESET' || code === 'EHOSTUNREACH' || code === 'ENETUNREACH') {
    return `Connection to api.github.com failed (${code}) — a firewall or proxy may be blocking it.`
  }

  if (typeof code === 'string' && /CERT|SSL|TLS/i.test(code)) {
    return `TLS handshake with api.github.com failed (${code}) — a proxy may be intercepting HTTPS.`
  }

  return `api.github.com: ${error?.message || String(error)}`
}

/** A git command runner — main.ts injects its runGit; tests inject a fake. */
export type GitRunner = (
  args: string[],
  options?: { cwd?: string }
) => Promise<{
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
