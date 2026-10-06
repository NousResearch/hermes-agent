import { DEFAULT_REASONING_EFFORT, type ModelOptionsResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { atom } from 'nanostores'
import { useRef } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { currentPickerSelection } from '@/lib/model-status-label'
import {
  $modelPresets,
  applyModelPreset,
  getModelPreset,
  type ModelPreset,
  modelPresetKey,
  modelPresetServiceTier,
  setModelPreset
} from '@/store/model-presets'
import { $defaultReasoningEffort, markComposerSelectionManual } from '@/store/session'

import type { ModelMenuController } from './model-catalog-menu'
import { useGatewayConnected } from './use-gateway-connected'

const UNKNOWN_SERVICE_TIER = atom('')
const optionEdits = new Map<string, symbol>()

const nextEdit = (key: string) => {
  const revision = Symbol(key)
  optionEdits.set(key, revision)

  return revision
}

export interface ModelSelection {
  model: string
  provider: string
  /** Runtime id of the surface that opened the menu. When set, the switch
   *  targets that session (a tile) instead of the primary `$activeSessionId`. */
  sessionId?: null | string
}

export interface ModelMenuHostProps {
  gateway?: HermesGateway
  ownerConnectionId?: string
  onSelectModel: (selection: ModelSelection) => Promise<boolean> | void
  profile?: string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

/**
 * The controller that gives a model-menu edit its meaning on a chat surface —
 * write through to THIS surface's session, remember the pick as a global
 * preset, keep the optimistic stores honest, and roll back on a failed gateway
 * write. Shared by the composer's model menu and its reasoning menu so both
 * pills edit the same session through one code path.
 */
export function useModelMenuController({
  gateway,
  onSelectModel,
  ownerConnectionId,
  profile = 'default',
  requestGateway
}: ModelMenuHostProps) {
  const { t } = useI18n()
  // Bind to THIS surface's SessionView (primary or tile) so each pane's menu
  // shows/switches its own model — not the primary-only globals.
  const view = useSessionView()
  const activeSessionId = useStore(view.$runtimeId)
  const storedSessionId = useStore(view.$storedId)
  const currentFastMode = useStore(view.$fast)
  const currentServiceTier = useStore(view.$serviceTier ?? UNKNOWN_SERVICE_TIER)
  const currentModel = useStore(view.$model)
  const currentProvider = useStore(view.$provider)
  const currentReasoningEffort = useStore(view.$reasoningEffort)
  const currentReasoningEffortWire = useStore(view.$reasoningEffortWire)
  const currentReasoningEffortPending = useStore(view.$reasoningEffortPending)
  const modelPresets = useStore($modelPresets)
  const defaultEffort = useStore($defaultReasoningEffort) || DEFAULT_REASONING_EFFORT
  const gatewayConnected = useGatewayConnected(gateway)
  const touchesPrimary = view.kind === 'primary'

  // Subscribe to the SAME query the menu runs (identical key ⇒ React Query
  // dedupes, no second fetch). It must be a live subscription, not a cache
  // peek: with no model in the session store yet, currentPickerSelection falls
  // back to the catalog's reported current, and a non-reactive read would
  // never repaint that fallback once the catalog resolved.
  const modelOptions = useQuery({
    enabled: gatewayConnected,
    queryKey: modelOptionsQueryKey(profile, activeSessionId, ownerConnectionId),
    queryFn: (): Promise<ModelOptionsResult> =>
      requestModelOptions({ gateway, profile, request: requestGateway, sessionId: activeSessionId })
  })

  const { model: optionsModel, provider: optionsProvider } = currentPickerSelection(
    { model: currentModel, provider: currentProvider },
    modelOptions.data
  )

  const hostScope = `${ownerConnectionId ?? ''}::${profile}`
  const latestHostScope = useRef(hostScope)
  latestHostScope.current = hostScope

  const patchPreset = (patch: ModelPreset, row: { model: string; provider: string }, failMessage: string) => {
    const dimensions = [
      ...(patch.effort !== undefined ? ['effort' as const] : []),
      ...(modelPresetServiceTier(patch) !== undefined ? ['speed' as const] : [])
    ]

    const stamps = new Map(
      dimensions.map(dimension => {
        const ownerKey = `${hostScope}::${activeSessionId ?? 'draft'}::${dimension}`
        const presetKey = `${modelPresetKey(row.provider, row.model)}::${dimension}`

        return [dimension, { ownerKey, presetKey, owner: nextEdit(ownerKey), preset: nextEdit(presetKey) }]
      })
    )

    setModelPreset(row.provider, row.model, patch)

    if (touchesPrimary) {
      markComposerSelectionManual()
    }

    void applyModelPreset(patch, {
      failMessage,
      scope: hostScope,
      primary: touchesPrimary,
      request: requestGateway,
      sessionId: activeSessionId,
      isCurrent: dimension => {
        const stamp = stamps.get(dimension)!

        return (
          optionEdits.get(stamp.ownerKey) === stamp.owner &&
          latestHostScope.current === hostScope &&
          view.$runtimeId.get() === activeSessionId &&
          (!view.$model.get() || view.$model.get().replace(/-fast$/, '') === row.model) &&
          (!view.$provider.get() || view.$provider.get() === row.provider)
        )
      },
      onFailure: (dimension, confirmed) => {
        const stamp = stamps.get(dimension)!

        if (optionEdits.get(stamp.presetKey) !== stamp.preset) {
          return
        }

        const current = getModelPreset(row.provider, row.model)

        if (dimension === 'effort' && current.effort === patch.effort) {
          setModelPreset(row.provider, row.model, { effort: confirmed.effort })
        } else if (dimension === 'speed' && modelPresetServiceTier(current) === modelPresetServiceTier(patch)) {
          setModelPreset(row.provider, row.model, {
            fast: confirmed.fast ?? false,
            serviceTier: modelPresetServiceTier(confirmed) ?? 'normal'
          })
        }
      }
    }).finally(() => {
      // A settled request only releases its own stamps; a newer writer keeps
      // its identity, even when another session edits the same model preset.
      for (const stamp of stamps.values()) {
        if (optionEdits.get(stamp.ownerKey) === stamp.owner) {
          optionEdits.delete(stamp.ownerKey)
        }

        if (optionEdits.get(stamp.presetKey) === stamp.preset) {
          optionEdits.delete(stamp.presetKey)
        }
      }
    })
  }

  const controller: ModelMenuController = {
    // Selecting a model row restores that model's remembered preset onto the
    // session (effort/fast). applyModelPreset owns the batched gateway write.
    applyPreset: (preset, row) => patchPreset(preset, row, t.shell.modelOptions.updateFailed),

    current: {
      effort: currentReasoningEffort,
      effortPending: currentReasoningEffortPending,
      effortWire: currentReasoningEffortWire,
      fast: currentFastMode,
      serviceTier: currentServiceTier,
      model: optionsModel,
      provider: optionsProvider
    },

    presetFor: (provider, model) => modelPresets[modelPresetKey(provider, model)] ?? {},

    // The composer picker never persists the profile default. With a session it
    // scopes the switch to that session; with none it's UI state shipped on the
    // next session.create. Always stamp sessionId from this surface so a tile
    // switch never hits the primary (busy) session by accident.
    select: (model, provider) => {
      return onSelectModel({ model, provider, sessionId: activeSessionId || null })
    },

    setOptions: (patch, row) => {
      // Editing always records the model's global preset (keyed by
      // provider::model, not per-surface — a tile edit re-applies to that model
      // everywhere); the active model also gets it pushed onto its OWN session.
      // Non-active edits stay preset-only — no model switch, no session write.
      if (!row.isActive) {
        setModelPreset(row.provider, row.model, patch)

        // Invalidate a pending rollback for the same globally remembered dimension.
        if (patch.effort !== undefined) {
          nextEdit(`${modelPresetKey(row.provider, row.model)}::effort`)
        }

        if (modelPresetServiceTier(patch) !== undefined) {
          nextEdit(`${modelPresetKey(row.provider, row.model)}::speed`)
        }

        return
      }

      patchPreset(patch, row, t.shell.modelOptions.updateFailed)
    }
  }

  return { activeSessionId, controller, defaultEffort, modelOptions }
}
