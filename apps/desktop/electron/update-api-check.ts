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

/**
 * The status both update-check paths (API and ls-remote) return once the remote
 * tip is known and differs from HEAD. Extracted so the parked-branch case -
 * where the tip is already merged into a local branch - is answered the same
 * way for every install form instead of only on the non-GitHub path (#114946).
 */
export interface TipStatus {
  behind: number | null
  updateAvailable: boolean
  targetSha: string
  commits: CompareCommit[]
}

export interface TipStatusInput {
  /**
   * The compare payload when the call succeeded; null when it failed. GitHub's
   * compare endpoint needs BOTH refs on the remote, so an install whose HEAD
   * carries local commits (parked branch / `update_in_place`) gets a 404.
   */
  compared: { behind: number; commits: CompareCommit[] } | null
  /**
   * `git merge-base --is-ancestor <tip> HEAD` — the local graph's own answer.
   * True means everything the remote has is already in HEAD, which is exactly
   * "up to date" and is the only source that knows it when `compared` is null.
   */
  tipIsAncestorOfHead: boolean
  targetSha: string
}

export function resolveTipStatus({ compared, tipIsAncestorOfHead, targetSha }: TipStatusInput): TipStatus {
  // A local commit sitting AHEAD of the remote is not an update: flagging it
  // nudges the user into wiping their work. `compared.behind === 0` says so
  // from the compare payload; the local probe says so when the payload never
  // arrived (404, rate limit, offline) — without it, such installs reported
  // "update available" forever, immediately after a good update.
  if (tipIsAncestorOfHead || compared?.behind === 0) {
    return { behind: 0, updateAvailable: false, targetSha, commits: [] }
  }

  // Still an honest "update available, count unknown" when the payload is
  // missing — never a fabricated number, and never a fabricated changelog.
  return {
    behind: compared ? compared.behind : null,
    updateAvailable: true,
    targetSha,
    commits: compared?.commits ?? []
  }
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
    const when = minutes === null ? 'within an hour' : minutes === 1 ? 'in about a minute' : `in about ${minutes} minutes`

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
