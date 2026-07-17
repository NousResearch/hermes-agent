import { type QueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'

import { getGlobalModelInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  getCurrentModelSource,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'
import type { ModelOptionsResponse } from '@/types/hermes'

interface ModelSelection {
  model: string
  provider: string
}

interface ModelControlsOptions {
  queryClient: QueryClient
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function useModelControls({ queryClient, requestGateway }: ModelControlsOptions) {
  const { t } = useI18n()
  const copy = t.desktop

  // All callbacks here read reactive session state from the store (.get())
  // rather than capturing it as a prop. The actions bag in wiring.tsx mutates
  // in place to keep a stable identity, so memoized surfaces capture these
  // callbacks once and never re-evaluate — a captured prop would be stale
  // forever. The store read is always current.
  const updateModelOptionsCache = useCallback(
    (provider: string, model: string, includeGlobal: boolean) => {
      const patch = (prev: ModelOptionsResponse | undefined) => ({ ...(prev ?? {}), provider, model })

      queryClient.setQueryData<ModelOptionsResponse>(['model-options', $activeSessionId.get() || 'global'], patch)

      if (includeGlobal) {
        queryClient.setQueryData<ModelOptionsResponse>(['model-options', 'global'], patch)
      }
    },
    [queryClient]
  )

  // Seed the composer's model state from the profile default. `force` reseeds
  // for a profile swap (the new profile has its own default); otherwise this
  // only fills an EMPTY selection so a user's pick (plain UI state in
  // $currentModel) survives the lifecycle refreshes that fire on boot / fresh
  // draft / session events. A live session owns the footer, so skip entirely.
  const refreshCurrentModel = useCallback(async (force = false) => {
    try {
      if ($activeSessionId.get()) {
        return
      }

      if (!force && $currentModel.get() && getCurrentModelSource() === 'manual') {
        return
      }

      const result = await getGlobalModelInfo()

      if ($activeSessionId.get() || (!force && $currentModel.get() && getCurrentModelSource() === 'manual')) {
        return
      }

      if (typeof result.model === 'string') {
        setCurrentModel(result.model)
      }

      if (typeof result.provider === 'string') {
        setCurrentProvider(result.provider)
      }

      if (typeof result.model === 'string' || typeof result.provider === 'string') {
        setCurrentModelSource('default')
      }
    } catch {
      // The delayed session.info event still updates this once the agent is ready.
    }
  }, [])

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
      const prevModel = $currentModel.get()
      const prevProvider = $currentProvider.get()
      const prevSource = getCurrentModelSource()

      const liveSessionId = $activeSessionId.get()

      setCurrentModel(selection.model)
      setCurrentProvider(selection.provider)
      setCurrentModelSource('manual')
      updateModelOptionsCache(selection.provider, selection.model, !liveSessionId)

      // No live session yet: the pick is pure UI state. session.create reads
      // $currentModel/$currentProvider and applies it as that session's override.
      if (!liveSessionId) {
        return true
      }

      try {
        await requestGateway('config.set', {
          session_id: liveSessionId,
          key: 'model',
          value: `${selection.model} --provider ${selection.provider} --session`
        })

        void queryClient.invalidateQueries({ queryKey: ['model-options', liveSessionId] })

        return true
      } catch (err) {
        setCurrentModel(prevModel)
        setCurrentProvider(prevProvider)
        setCurrentModelSource(prevSource)
        updateModelOptionsCache(prevProvider, prevModel, !liveSessionId)
        notifyError(err, copy.modelSwitchFailed)

        return false
      }
    },
    [copy.modelSwitchFailed, queryClient, requestGateway, updateModelOptionsCache]
  )

  return { refreshCurrentModel, selectModel, updateModelOptionsCache }
}
