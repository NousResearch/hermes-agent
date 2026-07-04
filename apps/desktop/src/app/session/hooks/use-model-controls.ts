import { type QueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'

import { getGlobalModelInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { activeProfileQueryKey } from '@/store/profile'
import type { ModelOptionsResponse } from '@/types/hermes'

import { MAIN_PANE_VIEW, type PaneSessionView } from '../../chat/pane-view'

interface ModelSelection {
  model: string
  provider: string
}

interface ModelControlsOptions {
  activeSessionId: string | null
  queryClient: QueryClient
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  /** The pane view whose model chip a pick writes. Defaults to the main
   *  bundle — the exact global atoms/persisting setters this hook always
   *  used — so the controller's instance is unchanged; SplitChatPane runs a
   *  second instance bound to its own bundle (plain, non-persisting setters)
   *  and its profile-pinned request, so a split pick can never clobber the
   *  main chip or the sticky composer-model localStorage keys. */
  view?: PaneSessionView
}

export function useModelControls({
  activeSessionId,
  queryClient,
  requestGateway,
  view = MAIN_PANE_VIEW
}: ModelControlsOptions) {
  const { t } = useI18n()
  const copy = t.desktop

  const updateModelOptionsCache = useCallback(
    (provider: string, model: string, includeGlobal: boolean) => {
      const patch = (prev: ModelOptionsResponse | undefined) => ({ ...(prev ?? {}), provider, model })

      // Keys carry the live gateway profile segment (see the model-options
      // useQuery call sites) — writes land on the active profile's slot only.
      queryClient.setQueryData<ModelOptionsResponse>(
        ['model-options', activeProfileQueryKey(), activeSessionId || 'global'],
        patch
      )

      if (includeGlobal) {
        queryClient.setQueryData<ModelOptionsResponse>(['model-options', activeProfileQueryKey(), 'global'], patch)
      }
    },
    [activeSessionId, queryClient]
  )

  // Seed the composer's model state from the profile default. `force` reseeds
  // for a profile swap (the new profile has its own default); otherwise this
  // only fills an EMPTY selection so a user's pick (plain UI state in
  // $currentModel) survives the lifecycle refreshes that fire on boot / fresh
  // draft / session events. A live session owns the footer, so skip entirely.
  const refreshCurrentModel = useCallback(
    async (force = false) => {
      try {
        if (view.$activeSessionId.get()) {
          return
        }

        if (!force && view.$currentModel.get()) {
          return
        }

        const result = await getGlobalModelInfo()

        if (view.$activeSessionId.get() || (!force && view.$currentModel.get())) {
          return
        }

        if (typeof result.model === 'string') {
          view.setCurrentModel(result.model)
        }

        if (typeof result.provider === 'string') {
          view.setCurrentProvider(result.provider)
        }
      } catch {
        // The delayed session.info event still updates this once the agent is ready.
      }
    },
    [view]
  )

  // Returns whether the switch succeeded so callers can await it before applying
  // follow-up changes. The composer model is plain UI state: with no live
  // session it's just stored (and shipped on the next session.create); with one
  // it's scoped to that session via config.set. It NEVER writes the profile
  // default — that lives in Settings → Model — so picking a model here can't
  // silently mutate global config.
  const selectModel = useCallback(
    async (selection: ModelSelection): Promise<boolean> => {
      // Snapshot for rollback: the switch is applied optimistically, so a
      // failure must restore the prior model/provider (store + query cache)
      // rather than leave the UI showing a model the backend never selected.
      const prevModel = view.$currentModel.get()
      const prevProvider = view.$currentProvider.get()

      view.setCurrentModel(selection.model)
      view.setCurrentProvider(selection.provider)
      updateModelOptionsCache(selection.provider, selection.model, !activeSessionId)

      // No live session yet: the pick is pure UI state. session.create reads
      // $currentModel/$currentProvider and applies it as that session's override.
      if (!activeSessionId) {
        return true
      }

      try {
        await requestGateway('config.set', {
          session_id: activeSessionId,
          key: 'model',
          value: `${selection.model} --provider ${selection.provider}`
        })

        void queryClient.invalidateQueries({ queryKey: ['model-options', activeProfileQueryKey(), activeSessionId] })

        return true
      } catch (err) {
        view.setCurrentModel(prevModel)
        view.setCurrentProvider(prevProvider)
        updateModelOptionsCache(prevProvider, prevModel, !activeSessionId)
        notifyError(err, copy.modelSwitchFailed)

        return false
      }
    },
    [activeSessionId, copy.modelSwitchFailed, queryClient, requestGateway, updateModelOptionsCache, view]
  )

  return { refreshCurrentModel, selectModel, updateModelOptionsCache }
}
