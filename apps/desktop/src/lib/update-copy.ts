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
  /** Git-current checkout under a stale renderer bundle: the update is a rebuild. */
  rebuildTitle: string
  rebuildBody: string
}

export interface ResolveUpdateCopyInput {
  target: UpdateTarget
  /** Number of commit rows actually shown in the changelog. 0 → no notes. */
  shownItems: number
  /** True when nothing is behind and the only work is rebuilding the stale
   *  client bundle (DesktopUpdateStatus.bundleOutOfSync with behind 0). */
  bundleRebuildOnly?: boolean
  copy: UpdateCopyStrings
}

export interface UpdateCopyResult {
  title: string
  body: string
}

export function resolveUpdateCopy({
  target,
  shownItems,
  bundleRebuildOnly = false,
  copy
}: ResolveUpdateCopyInput): UpdateCopyResult {
  if (bundleRebuildOnly && target === 'client') {
    return { title: copy.rebuildTitle, body: copy.rebuildBody }
  }

  const title = target === 'backend' ? copy.availableTitleBackend : copy.availableTitle

  const body =
    shownItems === 0
      ? copy.availableBodyNoChangelog
      : target === 'backend'
        ? copy.availableBodyBackend
        : copy.availableBody

  return { title, body }
}
