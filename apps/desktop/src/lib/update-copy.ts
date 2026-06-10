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
}

export interface ResolveUpdateCopyInput {
  target: UpdateTarget
  /** Number of commit rows actually shown in the changelog. 0 → no notes. */
  shownItems: number
  copy: UpdateCopyStrings
}

export interface UpdateCopyResult {
  title: string
  body: string
}

export function resolveUpdateCopy({ target, shownItems, copy }: ResolveUpdateCopyInput): UpdateCopyResult {
  const title = target === 'backend' ? copy.availableTitleBackend : copy.availableTitle

  const body =
    shownItems === 0
      ? copy.availableBodyNoChangelog
      : target === 'backend'
        ? copy.availableBodyBackend
        : copy.availableBody

  return { title, body }
}
