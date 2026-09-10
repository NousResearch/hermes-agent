import { atom, type WritableAtom } from 'nanostores'

import type { WorldCondition, WorldEvent } from '@/app/lunar-city/world-events'
import { Codecs, persistentAtom } from '@/lib/persisted'

// Device-scoped desktop preferences. These control whether this renderer
// surface exists and intentionally do not follow a gateway profile.
export const WORLD_ENABLED_STORAGE_KEY = 'hermes.desktop.world.enabled.v1'
export const WORLD_ONBOARDING_DISMISSED_STORAGE_KEY = 'hermes.desktop.world.onboardingDismissed.v1'
export const WORLD_CURSORS_STORAGE_KEY = 'hermes.desktop.world.cursors.v1'

export const $worldEnabled = persistentAtom(WORLD_ENABLED_STORAGE_KEY, true, Codecs.bool)
export const $worldOnboardingDismissed = persistentAtom(WORLD_ONBOARDING_DISMISSED_STORAGE_KEY, false, Codecs.bool)

export interface WorldCursorState {
  bySource: Record<string, string>
  dismissedRecapIds: string[]
  lastOpenedAt: number | null
}

export interface WorldProjection {
  conditions: WorldCondition[]
  recentEvents: WorldEvent[]
  sourceError: string | null
  stale: boolean
  transitions: WorldEvent[]
}

const EMPTY_PROJECTION: WorldProjection = {
  conditions: [],
  recentEvents: [],
  sourceError: null,
  stale: false,
  transitions: []
}

function sanitizeCursors(value: unknown): WorldCursorState {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return { bySource: {}, dismissedRecapIds: [], lastOpenedAt: null }
  }

  const raw = value as Record<string, unknown>

  const bySource =
    raw.bySource && typeof raw.bySource === 'object' && !Array.isArray(raw.bySource)
      ? Object.fromEntries(
          Object.entries(raw.bySource).filter(
            (entry): entry is [string, string] =>
              typeof entry[0] === 'string' &&
              typeof entry[1] === 'string' &&
              entry[0].length <= 200 &&
              entry[1].length <= 200
          )
        )
      : {}

  const dismissedRecapIds = Array.isArray(raw.dismissedRecapIds)
    ? raw.dismissedRecapIds
        .filter((id): id is string => typeof id === 'string' && id.length > 0 && id.length <= 200)
        .slice(-100)
    : []

  const lastOpenedAt =
    typeof raw.lastOpenedAt === 'number' && Number.isFinite(raw.lastOpenedAt) ? raw.lastOpenedAt : null

  return { bySource, dismissedRecapIds, lastOpenedAt }
}

export const $worldCursors = persistentAtom<WorldCursorState>(
  WORLD_CURSORS_STORAGE_KEY,
  { bySource: {}, dismissedRecapIds: [], lastOpenedAt: null },
  Codecs.json(sanitizeCursors)
)

export const $worldProjection: WritableAtom<WorldProjection> = atom(EMPTY_PROJECTION)

export function setWorldEnabled(enabled: boolean): void {
  $worldEnabled.set(enabled)
}

export function setWorldOnboardingDismissed(dismissed = true): void {
  $worldOnboardingDismissed.set(dismissed)
}

export function setWorldProjection(projection: WorldProjection): void {
  $worldProjection.set(projection)
}

export function setWorldOpenedAt(timestamp = Date.now()): void {
  $worldCursors.set({ ...$worldCursors.get(), lastOpenedAt: timestamp })
}

export function recordWorldCursor(scope: string, id: string): void {
  if (!scope || !id) {
    return
  }

  $worldCursors.set({ ...$worldCursors.get(), bySource: { ...$worldCursors.get().bySource, [scope]: id } })
}

export function dismissWorldRecap(id: string): void {
  if (!id || $worldCursors.get().dismissedRecapIds.includes(id)) {
    return
  }

  $worldCursors.set({
    ...$worldCursors.get(),
    dismissedRecapIds: [...$worldCursors.get().dismissedRecapIds, id].slice(-100)
  })
}

export function resetWorldProjection(): void {
  $worldProjection.set(EMPTY_PROJECTION)
}
