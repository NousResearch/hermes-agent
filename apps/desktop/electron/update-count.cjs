'use strict'

// Resolve how many commits the local checkout is behind origin for the desktop
// update indicator. On a SHALLOW checkout (installer clones with --depth 1) the
// local history often shares no merge-base with the freshly fetched origin tip,
// so `git rev-list HEAD..origin/<branch> --count` enumerates the entire remote
// ancestry and returns a bogus huge number (e.g. 12104) — see #51922. In that
// case the count is meaningless; fall back to a binary up-to-date check by SHA,
// exactly like the official-SSH path in checkUpdates() and the CLI guard in
// hermes_cli/banner.py. Full clones (developers / Docker dev images) keep the
// exact count path unchanged.
function resolveBehindCount({ countStr, currentSha, targetSha, isShallow, hasMergeBase }) {
  if (isShallow && !hasMergeBase) {
    if (currentSha && targetSha && currentSha === targetSha) return 0
    return 1 // behind by an unknown amount — show a generic "update available"
  }
  return Number.parseInt(countStr, 10) || 0
}

module.exports = { resolveBehindCount }
