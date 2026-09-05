import { isValidInstallCommit } from './install-stamp-identity'

const FULL_GIT_SHA = /^[0-9a-f]{40}$/i

// A packaged client executes the assets recorded by its build stamp. The
// checkout used to stage the next update may legitimately be older, so using
// that checkout's HEAD makes a freshly installed app look behind. Dev runs do
// execute from the checkout and keep using its live HEAD.
function resolveRunningClientSha({ checkoutSha, installStamp, isPackaged }) {
  const stampedSha = installStamp?.commit
  const hasUsableStamp = isValidInstallCommit(stampedSha)

  return isPackaged && hasUsableStamp ? stampedSha : checkoutSha
}

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

function commitRangeRevision({ currentSha, branch }) {
  return `${currentSha || 'HEAD'}..origin/${branch}`
}

// Resolve how many commits the running client is behind origin for the desktop
// update indicator. Shallow checkouts use SHA equality plus any positively
// proven local-ahead ancestry; exact counts remain exclusive to full clones.
function resolveBehindCount({
  countStr,
  currentSha,
  targetSha,
  isShallow,
  countAvailable = true,
  targetIsAncestorOfCurrent = false
}) {
  if (!shouldCountCommits({ isShallow })) {
    if (currentSha && targetSha && (currentSha === targetSha || targetIsAncestorOfCurrent)) {
      return 0
    }

    // An update IS available, but its size is unknowable without a merge-base.
    // Return null — never a numeric sentinel: the UI used to render the old
    // `1` as a literal "1 change included" even when the true distance was
    // far larger. null lets every surface say "update available" honestly.
    return null
  }

  // A packaged stamp can name a commit that is not present in the update
  // checkout (for example a locally built client). Do not turn a failed
  // rev-list into zero; let the compare API recover the count, or surface the
  // honest unknown-count state when the commit is not published.
  if (!countAvailable) {
    return null
  }

  return Number.parseInt(countStr, 10) || 0
}

// Shallow history can also contaminate the changelog range. Trust the fetched
// remote tip itself, but do not walk its ancestry. Full clones retain the
// detailed range used by the existing update overlay.
function resolveCommitLogSelection({ branch, isShallow, currentSha = null }) {
  const remote = `origin/${branch}`

  return isShallow
    ? { limit: 1, revision: remote }
    : { limit: 40, revision: commitRangeRevision({ currentSha, branch }) }
}

// When the local graph can't count (behind === null), the GitHub compare API
// still can: `GET /repos/<owner>/<repo>/compare/<current>...<target>` returns
// `ahead_by` — how many commits the remote tip is ahead of the running client,
// i.e. exactly the behind count the shallow clone lost. Unauthenticated, no
// clone depth required. Pure URL builder + response parser here; the network
// call lives with the caller.
function compareApiUrl({ currentSha, originUrl, targetSha }) {
  if (!FULL_GIT_SHA.test(currentSha || '') || !FULL_GIT_SHA.test(targetSha || '')) {
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

export {
  commitRangeRevision,
  compareApiUrl,
  parseCompareBehindCount,
  resolveBehindCount,
  resolveCommitLogSelection,
  resolveRunningClientSha,
  shouldCountCommits
}
