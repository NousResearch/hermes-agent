// Whether `git rev-list HEAD..origin/<branch> --count` produces a meaningful
// number worth computing. On a SHALLOW checkout (installer clones with
// --depth 1) the local history often shares no merge-base with the freshly
// fetched origin tip, so the count enumerates the entire remote ancestry and
// returns a bogus huge number (e.g. 12104) — see #51922. resolveBehindCount
// discards that bogus count in favour of a SHA compare, so the caller should
// SKIP the expensive rev-list entirely in that case rather than run it and
// throw the result away.
function shouldCountCommits({ isShallow, hasMergeBase }) {
  return !(isShallow && !hasMergeBase)
}

// Resolve how many commits the local checkout is behind origin for the desktop
// update indicator. When the count isn't meaningful (shallow + no merge-base)
// fall back to a binary up-to-date check by SHA, exactly like the official-SSH
// path in checkUpdates() and the CLI guard in hermes_cli/banner.py. Full clones
// (developers / Docker dev images) keep the exact count path unchanged.
function resolveBehindCount({ countStr, currentSha, targetSha, isShallow, hasMergeBase }) {
  if (!shouldCountCommits({ isShallow, hasMergeBase })) {
    if (currentSha && targetSha && currentSha === targetSha) {
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

export { resolveBehindCount, shouldCountCommits }
