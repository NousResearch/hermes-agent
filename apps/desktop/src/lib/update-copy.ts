/**
 * Pure copy-selection for the updates overlay's "available" state.
 *
 * Names the update target (client vs the connected backend in remote mode) and
 * degrades honestly when there's no commit changelog to show (e.g. a pip /
 * non-git backend where `git log` yields nothing) instead of generic filler.
 * On a release-feed channel it names the release tag instead of commit
 * vocabulary, because a release feed carries no commit rows by construction.
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
  /** App-installer mechanism: the OS owns the apply, so the body says so. */
  availableBodyAppInstaller: string
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
  /**
   * 'app-installer': the OS App Installer owns the apply (out-of-store MSIX)
   * — the body names Windows as the finisher, never commit vocabulary.
   */
  mechanism?: 'app-installer' | 'external' | 'windows-handoff' | 'posix-handoff' | 'manual'
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
  mechanism,
  copy
}: ResolveUpdateCopyInput): UpdateCopyResult {
  const title = target === 'backend' ? copy.availableTitleBackend : copy.availableTitle

  // App-installer: the OS owns the apply, and there is no commit list by
  // construction (the OS checker answers yes/no). Name Windows as the
  // finisher; the tag is named when known.
  if (mechanism === 'app-installer') {
    return { title, body: latestTag ? copy.availableBodyRelease(latestTag) : copy.availableBodyAppInstaller }
  }

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
