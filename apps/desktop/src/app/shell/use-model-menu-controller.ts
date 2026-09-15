import { DEFAULT_REASONING_EFFORT, type ModelOptionsResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

import { useSessionView } from '@/app/chat/session-view'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { currentPickerSelection } from '@/lib/model-status-label'
import { $modelPresets, applyModelPreset, modelPresetKey, setModelPreset } from '@/store/model-presets'
import { notifyError } from '@/store/notifications'
import {
  $defaultReasoningEffort,
  markComposerSelectionManual,
  setCurrentFastMode,
  setCurrentReasoningEffort
} from '@/store/session'
import { sessionTileDelegate } from '@/store/session-states'

import type { ModelMenuController } from './model-catalog-menu'

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
  onUseProfileDefault?: () => Promise<void> | void
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
  const currentFastMode = useStore(view.$fast)
  const currentModel = useStore(view.$model)
  const currentProvider = useStore(view.$provider)
  const currentReasoningEffort = useStore(view.$reasoningEffort)
  const modelPresets = useStore($modelPresets)
  const defaultEffort = useStore($defaultReasoningEffort) || DEFAULT_REASONING_EFFORT
  const touchesPrimary = view.kind === 'primary'

  // Subscribe to the SAME query the menu runs (identical key ⇒ React Query
  // dedupes, no second fetch). It must be a live subscription, not a cache
  // peek: with no model in the session store yet, currentPickerSelection falls
  // back to the catalog's reported current, and a non-reactive read would
  // never repaint that fallback once the catalog resolved.
  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, activeSessionId, ownerConnectionId),
    queryFn: (): Promise<ModelOptionsResult> =>
      requestModelOptions({ gateway, profile, request: requestGateway, sessionId: activeSessionId })
  })

  const { model: optionsModel, provider: optionsProvider } = currentPickerSelection(
    { model: currentModel, provider: currentProvider },
    modelOptions.data
  )

  // Push a reasoning change onto the session that owns it, with rollback.
  const patchReasoning = async (next: string, previous: string, provider: string, model: string) => {
    if (touchesPrimary) {
      markComposerSelectionManual()
      setCurrentReasoningEffort(next)
    } else if (activeSessionId) {
      sessionTileDelegate()?.updateSession(activeSessionId, state => ({ ...state, reasoningEffort: next }))
    }

    // Preset-only without a session: the gateway's `config.set` falls back to
    // global config when none matches — so don't reach it (preset + optimistic
    // store are the whole effect).
    if (!activeSessionId) {
      return
    }

    try {
      await requestGateway('config.set', { key: 'reasoning', session_id: activeSessionId, value: next })
    } catch (err) {
      if (touchesPrimary) {
        setCurrentReasoningEffort(previous)
      } else {
        sessionTileDelegate()?.updateSession(activeSessionId, state => ({ ...state, reasoningEffort: previous }))
      }

      setModelPreset(provider, model, { effort: previous })
      notifyError(err, t.shell.modelOptions.updateFailed)
    }
  }

  const patchFast = async (enabled: boolean, provider: string, model: string) => {
    if (touchesPrimary) {
      markComposerSelectionManual()
      setCurrentFastMode(enabled)
    } else if (activeSessionId) {
      sessionTileDelegate()?.updateSession(activeSessionId, state => ({ ...state, fast: enabled }))
    }

    if (!activeSessionId) {
      return
    }

    try {
      await requestGateway('config.set', {
        key: 'fast',
        session_id: activeSessionId,
        value: enabled ? 'fast' : 'normal'
      })
    } catch (err) {
      if (touchesPrimary) {
        setCurrentFastMode(!enabled)
      } else {
        sessionTileDelegate()?.updateSession(activeSessionId, state => ({ ...state, fast: !enabled }))
      }

      setModelPreset(provider, model, { fast: !enabled })
      notifyError(err, t.shell.modelOptions.fastFailed)
    }
  }

  const controller: ModelMenuController = {
    // Selecting a model row restores that model's remembered preset onto the
    // session (effort/fast). applyModelPreset owns the batched gateway write.
    applyPreset: (preset, row) => {
      setModelPreset(row.provider, row.model, preset)

      void applyModelPreset(preset, {
        failMessage: t.shell.modelOptions.updateFailed,
        primary: touchesPrimary,
        request: requestGateway,
        sessionId: activeSessionId
      })
    },

    current: {
      effort: currentReasoningEffort,
      fast: currentFastMode,
      model: optionsModel,
      provider: optionsProvider
    },

    presetFor: (provider, model) => modelPresets[modelPresetKey(provider, model)] ?? {},

    // The composer picker never persists the profile default. With a session it
    // scopes the switch to that session; with none it's UI state shipped on the
    // next session.create. Always stamp sessionId from this surface so a tile
    // switch never hits the primary (busy) session by accident.
    select: (model, provider) => onSelectModel({ model, provider, sessionId: activeSessionId || null }),

    setOptions: (patch, row) => {
      // Editing always records the model's global preset (keyed by
      // provider::model, not per-surface — a tile edit re-applies to that model
      // everywhere); the active model also gets it pushed onto its OWN session.
      // Non-active edits stay preset-only — no model switch, no session write.
      if (patch.effort !== undefined || patch.fast !== undefined) {
        setModelPreset(row.provider, row.model, patch)
      }

      if (!row.isActive) {
        return
      }

      if (patch.effort !== undefined) {
        void patchReasoning(patch.effort, currentReasoningEffort, row.provider, row.model)
      }

      if (patch.fast !== undefined) {
        void patchFast(patch.fast, row.provider, row.model)
      }
    }
  }

  return { activeSessionId, controller, defaultEffort, modelOptions, touchesPrimary }
}
