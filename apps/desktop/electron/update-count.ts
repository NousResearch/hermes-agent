// Whether `git rev-list HEAD..origin/<branch> --count` produces a meaningful
// number worth computing. Installer checkouts are shallow (`--depth 1`), so
// their visible graph is incomplete even when `merge-base` happens to find a
// common commit. A merge can expose ancestry that the local shallow boundary
// hides from HEAD, inflating the count with old commits. Exact counts are only
// trustworthy in full clones; shallow checkouts use presence-only status plus
// any positively proven local-ahead ancestry.
function shouldCountCommits({ isShallow }) {
  return !isShallow
}

// Resolve how many commits the local checkout is behind origin for the desktop
// update indicator. Shallow checkouts use SHA equality plus any positively
// proven local-ahead ancestry; exact counts remain exclusive to full clones.
function resolveBehindCount({ countStr, currentSha, targetSha, isShallow, targetIsAncestorOfHead = false }) {
  if (!shouldCountCommits({ isShallow })) {
    if (currentSha && targetSha && (currentSha === targetSha || targetIsAncestorOfHead)) {
      return 0
    }

    // An update IS available, but its size is unknowable without a merge-base.
    // Return null — never a numeric sentinel: the UI used to render the old
    // `1` as a literal "1 change included" even when the true distance was
    // far larger. null lets every surface say "update available" honestly.
    return null
  }

  return Number.parseInt(countStr, 10) || 0
}

// Shallow history can also contaminate the changelog range. Trust the fetched
// remote tip itself, but do not walk its ancestry. Full clones retain the
// detailed range used by the existing update overlay.
function resolveCommitLogSelection({ branch, isShallow }) {
  const remote = `origin/${branch}`

  return isShallow ? { limit: 1, revision: remote } : { limit: 40, revision: `HEAD..${remote}` }
}

// When the local graph can't count (behind === null), the GitHub compare API
// still can: `GET /repos/<owner>/<repo>/compare/<current>...<target>` returns
// `ahead_by` — how many commits the remote tip is ahead of the local HEAD,
// i.e. exactly the behind count the shallow clone lost. Unauthenticated, no
// clone depth required. Pure URL builder + response parser here; the network
// call lives with the caller.
function compareApiUrl({ currentSha, originUrl, targetSha }) {
  const sha = /^[0-9a-f]{40}$/i

  if (!sha.test(currentSha || '') || !sha.test(targetSha || '')) {
    return null
  }

  // Only GitHub remotes have a compare API. Reuse the canonical form the
  // official-remote check produces: `github.com/<owner>/<repo>`.
  const canonical = canonicalRemoteForCompare(originUrl)

  if (!canonical) {
    return null
  }

  return `https://api.github.com/repos/${canonical}/compare/${currentSha}...${targetSha}`
}

function canonicalRemoteForCompare(originUrl) {
  const value = String(originUrl || '').trim()

  const match =
    /^git@github\.com:([^/]+\/[^/]+?)(?:\.git)?\/?$/i.exec(value) ||
    /^(?:ssh:\/\/git@|https:\/\/|http:\/\/)github\.com\/([^/]+\/[^/]+?)(?:\.git)?\/?$/i.exec(value)

  return match ? match[1] : null
}

// `ahead_by` counts target commits not reachable from current — the behind
// count. `status` is "ahead" / "behind" / "diverged" / "identical" relative to
// current...target; any shape surprise returns null so the caller keeps the
// honest "update available" fallback instead of trusting a partial answer.
function parseCompareBehindCount(payload) {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const ahead = payload.ahead_by

  if (typeof ahead !== 'number' || !Number.isInteger(ahead) || ahead < 0) {
    return null
  }

  return ahead
}

// Extract the user-facing commit rows from the same compare payload.
// `payload.commits` mirrors the fields `readCommitLog` emits (sha, summary,
// author, at) plus a few extras, newest first, capped by GitHub at 250 rows —
// the 250 cap is exactly why the update dialog's "+ N more changes" line can
// exceed it. Any shape surprise returns [] so callers fall back to the local
// changelog path instead of rendering garbage.
function parseCompareCommits(payload) {
  if (!payload || typeof payload !== 'object' || !Array.isArray(payload.commits)) {
    return []
  }

  return payload.commits.flatMap(commit => {
    const sha =
      typeof commit?.sha === 'string' && commit.sha
        ? commit.sha
        : typeof commit?.node_id === 'string' && commit.node_id
          ? commit.node_id
          : ''

    const message = typeof commit?.commit?.message === 'string' ? commit.commit.message : ''
    const summary = message.split(/\r?\n/, 1)[0]
    const authorName = typeof commit?.commit?.author?.name === 'string' ? commit.commit.author.name : ''
    const date = Date.parse(commit?.commit?.author?.date ?? '')

    if (!sha || !summary || !Number.isFinite(date)) {
      return []
    }

    return [
      {
        sha,
        summary,
        author: authorName,
        at: date
      }
    ]
  })
}

export {
  compareApiUrl,
  parseCompareBehindCount,
  parseCompareCommits,
  resolveBehindCount,
  resolveCommitLogSelection,
  shouldCountCommits
}
