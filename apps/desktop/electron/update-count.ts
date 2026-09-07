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

// Ask git whether `targetSha` is already reachable from HEAD.
//
// Three outcomes, deliberately not two: `true` (proven ancestor — a carried
// local commit puts HEAD ahead), `false` (proven not an ancestor), and `null`
// (git could not answer). A spawn failure rejects rather than exiting
// non-zero, and a missing git binary is not evidence about the checkout.
// Folding that into `false` would let a broken git read as "update available"
// and nudge the user into an update that discards the commit they carry.
async function resolveAncestry({
  currentSha,
  runGit,
  targetSha,
  tipsEqual
}: {
  currentSha?: string
  runGit: (args: string[]) => Promise<{ code: number }>
  targetSha?: string
  tipsEqual: boolean
}): Promise<boolean | null> {
  if (tipsEqual || !currentSha || !targetSha) {
    return false
  }

  try {
    return (await runGit(['merge-base', '--is-ancestor', targetSha, 'HEAD'])).code === 0
  } catch {
    return null
  }
}

// The SSH-official passive check only learns tip SHAs from `ls-remote`, so it
// cannot count. Differing tips still do not prove the checkout is behind: a
// carried local commit puts HEAD ahead of the remote tip. Git can prove that
// case from local objects even on a shallow clone, and it is the only party
// that knows about a commit which exists nowhere else — the compare API 404s
// on it and returns null, which would otherwise read as "update available"
// forever, since updating never removes the carried commit.
//
// `targetIsAncestorOfHead` carries three states, not two. `false` means git
// answered "not an ancestor"; `null` means git could not answer at all. Only
// the first is evidence. When ancestry is unknown and the compare API is also
// silent, the honest answer is null — "something may be available, count
// unknown" — never a confident behind count derived from a failed probe.
function resolveSshBehindCount({
  compareBehind,
  currentSha,
  targetIsAncestorOfHead = false,
  targetSha
}: {
  compareBehind: number | null
  currentSha?: string
  targetIsAncestorOfHead?: boolean | null
  targetSha?: string
}) {
  if (currentSha && targetSha && (currentSha === targetSha || targetIsAncestorOfHead === true)) {
    return 0
  }

  return compareBehind
}

// The passive SSH path may know neither direction nor count. `null` keeps that
// uncertainty explicit; it must not turn into an update prompt until Git or the
// compare API proves that the remote has commits this checkout lacks.
function resolveSshUpdateAvailable(behind: number | null): boolean {
  return typeof behind === 'number' && behind > 0
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

export {
  compareApiUrl,
  parseCompareBehindCount,
  resolveAncestry,
  resolveBehindCount,
  resolveCommitLogSelection,
  resolveSshBehindCount,
  resolveSshUpdateAvailable,
  shouldCountCommits
}
