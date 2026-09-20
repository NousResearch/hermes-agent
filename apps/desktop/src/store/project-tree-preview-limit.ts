/**
 * How many session rows the sidebar's project tree previews per project before
 * the "show all sessions" toggle takes over. A device-local display preference
 * — it decides how much this window asks the backend for, never what the
 * backend stores — so it is not profile-scoped and needs no RPC.
 *
 * The value is clamped into [MIN, MAX]: localStorage is user-editable and the
 * number is forwarded to the backend as `preview_limit`, so an unbounded value
 * there would mean an unbounded tree payload.
 */

import { atom } from 'nanostores'

import { persistNumber, storedNumber } from '@/lib/storage'

export const PROJECT_TREE_PREVIEW_DEFAULT = 3
export const PROJECT_TREE_PREVIEW_MIN = 1
/** Matches the tree's own "show all" fan-out ceiling. */
export const PROJECT_TREE_PREVIEW_MAX = 2000

const KEY = 'hermes.desktop.projectTreePreviewLimit.v1'

export function clampProjectTreePreviewLimit(value: number): number {
  if (!Number.isFinite(value)) {
    return PROJECT_TREE_PREVIEW_DEFAULT
  }

  return Math.min(PROJECT_TREE_PREVIEW_MAX, Math.max(PROJECT_TREE_PREVIEW_MIN, Math.trunc(value)))
}

export const $projectTreePreviewLimit = atom<number>(
  typeof window === 'undefined'
    ? PROJECT_TREE_PREVIEW_DEFAULT
    : clampProjectTreePreviewLimit(storedNumber(KEY, PROJECT_TREE_PREVIEW_DEFAULT))
)

export function setProjectTreePreviewLimit(value: number): number {
  const next = clampProjectTreePreviewLimit(value)
  $projectTreePreviewLimit.set(next)

  return next
}

if (typeof window !== 'undefined') {
  $projectTreePreviewLimit.subscribe(limit => {
    persistNumber(KEY, limit)
  })
}
