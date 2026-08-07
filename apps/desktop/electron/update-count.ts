// Whether `git rev-list HEAD..origin/<branch> --count` produces a meaningful
// number worth computing. On a SHALLOW checkout (installer and most binary /
// desktop installs clone with --depth 1) the local history is truncated, so
// `rev-list --count` enumerates the entire remote ancestry and returns a bogus
// huge number (e.g. 12104, and in practice thousands — see #51922 and the
// real-world 4650 a shallow pin reported). The CLI's banner.py makes the same
// decision: ANY shallow checkout skips the count and falls back to a SHA
// compare, surfacing a generic "update available" instead of a scary,
// unactionable commit distance. Full clones (developers / Docker dev images)
// keep the exact count path unchanged.
function shouldCountCommits({ isShallow, hasMergeBase }: { isShallow: boolean; hasMergeBase?: boolean }) {
  // Previously this was `!(isShallow && !hasMergeBase)`, which still trusted
  // the count whenever a shallow clone happened to share a merge-base with the
  // tip — that path produces the bogus 4650. Shallow history is never reliable
  // to count across (the local boundary is artificial), so treat every shallow
  // checkout as binary and compare tip SHAs instead.
  return !isShallow
}

// Resolve how many commits the local checkout is behind origin for the desktop
// update indicator. On a shallow checkout we have no reliable history to count
// across, so we fall back to a binary up-to-date check by SHA — exactly like
// the official-SSH path in checkUpdates() and the CLI guard in
// hermes_cli/banner.py. Full clones keep the exact count path unchanged.
function resolveBehindCount({ countStr, currentSha, targetSha, isShallow, hasMergeBase }: {
  countStr: string
  currentSha: string
  targetSha: string
  isShallow: boolean
  hasMergeBase?: boolean
}) {
  if (isShallow) {
    if (currentSha && targetSha && currentSha === targetSha) {
      return 0
    }

    return 1 // behind by an unknown amount — show a generic "update available"
  }

  return Number.parseInt(countStr, 10) || 0
}

export { resolveBehindCount, shouldCountCommits }
