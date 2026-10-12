import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

import { notifyError } from './notifications'
import {
  $activeSessionId,
  $currentFastMode,
  $currentReasoningEffort,
  $currentServiceTier,
  setCurrentFastMode,
  setCurrentReasoningEffort,
  setCurrentServiceTier
} from './session'
import { $sessionStates, sessionTileDelegate } from './session-states'

const STORAGE_KEY = 'hermes.desktop.model-presets'
const pendingWrites = new Map<string, Promise<unknown>>()
const confirmedValues = new Map<string, string>()

/** Per-model reasoning/fast preset, remembered globally across sessions and
 *  re-applied to the session whenever that model is selected. Unset dimensions
 *  fall back to the Hermes default (medium effort, no fast). */
export interface ModelPreset {
  effort?: string
  fast?: boolean
  serviceTier?: string
}

export const modelPresetServiceTier = ({ fast, serviceTier }: ModelPreset): string | undefined =>
  serviceTier ?? (fast === undefined ? undefined : fast ? 'priority' : 'normal')

/** The bounded policies (`/fast auto`, `/fast cold`): the fast window opens on
 *  the backend's clock, so the composer/tile boolean `fast` stays off and only
 *  the exact tier rides (#132275). */
export const isBoundedSpeedPolicy = (tier: string | undefined): boolean => tier === 'auto' || tier === 'cold'

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

/** Stable `provider::model` key (matches the visibility-store format). */
export const modelPresetKey = (provider: string, model: string): string => `${provider}::${model}`

function load(): Record<string, ModelPreset> {
  const raw = storedString(STORAGE_KEY)

  if (!raw) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw)

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, ModelPreset>) : {}
  } catch {
    return {}
  }
}

export const $modelPresets = atom<Record<string, ModelPreset>>(load())

export function getModelPreset(provider: string, model: string): ModelPreset {
  return $modelPresets.get()[modelPresetKey(provider, model)] ?? {}
}

/** Merge a partial preset for one model and persist. */
export function setModelPreset(provider: string, model: string, patch: ModelPreset): void {
  const key = modelPresetKey(provider, model)
  const tier = modelPresetServiceTier(patch)

  const next = {
    ...$modelPresets.get(),
    [key]: {
      ...$modelPresets.get()[key],
      ...patch,
      ...(tier !== undefined ? { serviceTier: tier, fast: tier !== 'normal' } : {})
    }
  }

  $modelPresets.set(next)
  persistString(STORAGE_KEY, JSON.stringify(next))
}

/** The confirmed preset for a failed write's rollback: the last value the chain
 *  confirmed for the dimension. A bounded policy (auto/cold) rolls back with the
 *  boolean fast OFF — the window opens on the backend's clock (#132275). */
function confirmedPreset(dimension: 'effort' | 'speed', confirmedValue: string): ModelPreset {
  if (dimension === 'effort') {
    return { effort: confirmedValue }
  }

  return {
    serviceTier: confirmedValue || 'normal',
    fast: !!confirmedValue && confirmedValue !== 'normal' && !isBoundedSpeedPolicy(confirmedValue)
  }
}

/** Roll one failed dimension back to its last confirmed value. A late failure
 *  never undoes a newer edit: each store write only lands when the store still
 *  holds the value this write was setting. */
function rollbackFailedDimension(
  dimension: 'effort' | 'speed',
  err: unknown,
  state: {
    confirmedValue: string
    effort: string | undefined
    failMessage: string
    isCurrent: boolean
    onFailure?: (dimension: 'effort' | 'speed', confirmed: ModelPreset) => void
    ownsPrimary: boolean
    sessionId: null | string
    tier: string | undefined
  }
): void {
  const confirmed = confirmedPreset(dimension, state.confirmedValue)

  state.onFailure?.(dimension, confirmed)

  if (state.isCurrent) {
    if (state.ownsPrimary) {
      if (dimension === 'effort' && $currentReasoningEffort.get() === state.effort) {
        setCurrentReasoningEffort(confirmed.effort ?? '')
      } else if (dimension === 'speed' && $currentServiceTier.get() === state.tier) {
        setCurrentFastMode(confirmed.fast ?? false)
        setCurrentServiceTier(confirmed.serviceTier ?? '')
      }
    }

    if (state.sessionId) {
      sessionTileDelegate()?.updateSession(state.sessionId, tile => {
        if (dimension === 'effort' && tile.reasoningEffort === state.effort) {
          return { ...tile, reasoningEffort: confirmed.effort ?? '', reasoningEffortWire: '' }
        }

        if (dimension === 'speed' && tile.serviceTier === state.tier) {
          return { ...tile, fast: confirmed.fast ?? false, serviceTier: confirmed.serviceTier ?? '' }
        }

        return tile
      })
    }
  }

  notifyError(err, state.failMessage)
}

/** Apply a model's preset to the composer, then push it to a live session.
 *  `undefined` skips that dimension; values are capability-gated upstream.
 *  Without a session the local draft still needs the preset, but must not call
 *  `config.set`: that falls back to persistent profile config when no session
 *  matches and would rewrite the user's defaults.
 *
 *  `primary: false` scopes the optimistic write to the tile's session slice —
 *  a tile's picker must not clobber the primary composer's effort/fast. */
export async function applyModelPreset(
  preset: ModelPreset,
  ctx: {
    failMessage: string
    primary?: boolean
    request: RequestGateway
    sessionId: null | string
    isCurrent?: (dimension: 'effort' | 'speed') => boolean
    scope?: string
    onFailure?: (dimension: 'effort' | 'speed', confirmed: ModelPreset) => void
  }
): Promise<void> {
  const { effort } = preset
  const tier = modelPresetServiceTier(preset)
  const fast = tier === undefined ? undefined : tier !== 'normal' && !isBoundedSpeedPolicy(tier)
  const primary = ctx.primary ?? true
  const oldOwner = $activeSessionId.get()
  const slice = $sessionStates.get()[ctx.sessionId ?? '']

  const previous =
    primary && !slice
      ? { effort: $currentReasoningEffort.get(), fast: $currentFastMode.get(), serviceTier: $currentServiceTier.get() }
      : {
          effort: $sessionStates.get()[ctx.sessionId ?? '']?.reasoningEffort,
          fast: $sessionStates.get()[ctx.sessionId ?? '']?.fast,
          serviceTier: $sessionStates.get()[ctx.sessionId ?? '']?.serviceTier
        }

  const writeKey = (dimension: string) => `${ctx.scope ?? ''}::${ctx.sessionId}::${dimension}`

  if (primary) {
    if (effort !== undefined) {
      setCurrentReasoningEffort(effort)
    }

    if (fast !== undefined) {
      setCurrentFastMode(fast)
      setCurrentServiceTier(tier!)
    }
  }

  if (ctx.sessionId) {
    sessionTileDelegate()?.updateSession(ctx.sessionId, state => ({
      ...state,
      // Like setCurrentReasoningEffort: the wire level belongs to the previous
      // effort, so it stays unknown until the next session.info re-stamps it.
      ...(effort !== undefined ? { reasoningEffort: effort, reasoningEffortWire: '' } : {}),
      ...(fast !== undefined ? { fast, serviceTier: tier! } : {})
    }))
  }

  if (!ctx.sessionId) {
    return
  }

  // Each dimension has its own write/failure: an effort error must not skip
  // the speed request. A late failure cannot undo a newer edit or chat.
  await Promise.all(
    (
      [
        ['effort', effort],
        ['speed', tier]
      ] as const
    ).map(async ([dimension, value]) => {
      if (value === undefined || ctx.isCurrent?.(dimension) === false) {
        return
      }

      const key = writeKey(dimension)
      const preceding = pendingWrites.get(key)

      if (!preceding) {
        confirmedValues.set(
          key,
          dimension === 'effort'
            ? (previous.effort ?? '')
            : previous.serviceTier || (previous.fast ? 'priority' : 'normal')
        )
      }

      let write: Promise<void> | undefined

      try {
        write = Promise.resolve(preceding)
          .catch(() => {})
          .then(async () => {
            await ctx.request('config.set', {
              key: dimension === 'effort' ? 'reasoning' : 'fast',
              session_id: ctx.sessionId,
              value: dimension === 'speed' && value === 'priority' ? 'fast' : value
            })
            confirmedValues.set(key, value)
          })

        pendingWrites.set(key, write)

        await write
      } catch (err) {
        rollbackFailedDimension(dimension, err, {
          confirmedValue: confirmedValues.get(writeKey(dimension)) ?? '',
          effort,
          failMessage: ctx.failMessage,
          isCurrent: ctx.isCurrent?.(dimension) ?? true,
          onFailure: ctx.onFailure,
          ownsPrimary: primary && $activeSessionId.get() === oldOwner,
          sessionId: ctx.sessionId && (!primary || $activeSessionId.get() === oldOwner) ? ctx.sessionId : null,
          tier
        })
      } finally {
        // Rollback must read the last confirmed value before the final writer
        // releases the chain. Draft/skipped writes never enter these maps.
        if (pendingWrites.get(key) === write) {
          pendingWrites.delete(key)
          confirmedValues.delete(key)
        }
      }
    })
  )
}
