// Whether `git rev-list HEAD..origin/<branch> --count` produces a meaningful
// number worth computing. Any checkout without a merge-base has unrelated or
// incomplete history, so the count enumerates the remote's whole ancestry and
// returns a bogus huge number (for example 14829). This can happen in shallow
// installer clones and in full custom checkouts with an unrelated root.
function shouldCountCommits({ hasMergeBase }) {
  return hasMergeBase
}

// Resolve how many commits the local checkout is behind origin for the desktop
// update indicator. When the count isn't meaningful (shallow + no merge-base)
// fall back to a binary up-to-date check by SHA, exactly like the official-SSH
// path in checkUpdates() and the CLI guard in hermes_cli/banner.py. Full clones
// (developers / Docker dev images) keep the exact count path unchanged.
function resolveBehindCount({ countStr, currentSha, targetSha, hasMergeBase }) {
  if (!shouldCountCommits({ hasMergeBase })) {
    if (currentSha && targetSha && currentSha === targetSha) {
      return 0
    }

    return 1 // behind by an unknown amount — show a generic "update available"
  }

  return Number.parseInt(countStr, 10) || 0
}

const GIT_OBJECT_ID_PATTERN = /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/i

function isValidGitObjectId(value) {
  return typeof value === 'string' && GIT_OBJECT_ID_PATTERN.test(value) && !/^0+$/.test(value)
}

function resolveUpdateCurrentSha({ checkoutSha, installStampSha, isPackaged }) {
  if (isPackaged && isValidGitObjectId(installStampSha)) {
    return installStampSha
  }

  return checkoutSha
}

export { resolveBehindCount, resolveUpdateCurrentSha, shouldCountCommits }
