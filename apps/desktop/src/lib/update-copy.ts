/**
 * Pure copy-selection for the updates overlay's "available" state.
 *
 * Names the update target (client vs the connected backend in remote mode) and
 * degrades honestly when there's no commit changelog to show (e.g. a pip /
 * non-git backend where `git log` yields nothing) instead of generic filler.
 *
 * Extracted from updates-overlay.tsx so the wording logic is unit-testable.
 */

export type UpdateTarget = 'client' | 'backend'

export interface UpdateCopyStrings {
  availableTitle: string
  availableBody: string
  availableTitleBackend: string
  availableBodyBackend: string
  availableBodyNoChangelog: string
  /** Stable channel: names the release instead of describing commits. */
  availableBodyRelease: (tag: string) => string
}

export interface ResolveUpdateCopyInput {
  target: UpdateTarget
  /** Number of commit rows actually shown in the changelog. 0 → no notes. */
  shownItems: number
  /**
   * 'stable': the update is a release, so the body names its tag and the
   * commit-changelog wording never appears (a release feed carries no
   * commit rows anyway). 'main' or absent: commit vocabulary as before.
   */
  channel?: 'stable' | 'main'
  /** Stable channel: the release tag the update moves to, when known. */
  latestTag?: null | string
  copy: UpdateCopyStrings
}

export interface UpdateCopyResult {
  title: string
  body: string
}

export function resolveUpdateCopy({
  target,
  shownItems,
  channel = 'main',
  latestTag = null,
  copy
}: ResolveUpdateCopyInput): UpdateCopyResult {
  const title = target === 'backend' ? copy.availableTitleBackend : copy.availableTitle

  if (channel === 'stable') {
    // No-changelog copy would be wrong here: the absence of commit rows is
    // structural on a release feed, not a degraded install type.
    return { title, body: latestTag ? copy.availableBodyRelease(latestTag) : copy.availableBody }
  }

  const body =
    shownItems === 0
      ? copy.availableBodyNoChangelog
      : target === 'backend'
        ? copy.availableBodyBackend
        : copy.availableBody

  return { title, body }
}
