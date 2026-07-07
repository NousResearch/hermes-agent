import { type QueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'

import { getGlobalModelInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $activeSessionId, $currentModel, $currentProvider, setCurrentModel, setCurrentProvider } from '@/store/session'
import type { ModelOptionsResponse } from '@/types/hermes'

interface ModelSelection {
  model: string
  provider: string
}

interface ModelControlsOptions {
  activeSessionId: string | null
  queryClient: QueryClient
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function useModelControls({ activeSessionId, queryClient, requestGateway }: ModelControlsOptions) {
  const { t } = useI18n()
  const copy = t.desktop

  const updateModelOptionsCache = useCallback(
    (provider: string, model: string, includeGlobal: boolean) => {
      const patch = (prev: ModelOptionsResponse | undefined) => ({ ...(prev ?? {}), provider, model })

      queryClient.setQueryData<ModelOptionsResponse>(['model-options', activeSessionId || 'global'], patch)

      if (includeGlobal) {
        queryClient.setQueryData<ModelOptionsResponse>(['model-options', 'global'], patch)
      }
    },
    [activeSessionId, queryClient]
  )

  // Seed the composer's model state from the profile default. `force` reseeds
  // for a profile swap (the new profile has its own default); otherwise this
  // fills an EMPTY selection OR corrects a stale one so a config.yaml change
  // (model.default / model.provider) is reflected on next boot without the
  // user having to manually switch in the model picker.
  //
  // The early-return below previously bailed out whenever $currentModel was
  // non-empty, which meant a model written by onboarding would stick forever
  // even after the user updated config.yaml — the Desktop GUI would continue
  // routing to the onboarding-chosen provider/model, ignoring the new config.
  // (Issue #59979: Desktop GUI ignores model.default and model.provider.)
  //
  // Fix: always fetch the backend's current config-driven model and provider.
  // Update the atoms only when the stored value differs, so a user's deliberate
  // mid-session pick (selectModel) is still respected — it lands in localStorage
  // too, so the stored value will match what was deliberately chosen. A live
  // session is left untouched regardless.
  const refreshCurrentModel = useCallback(async (force = false) => {
    try {
      if ($activeSessionId.get()) {
        return
      }

      const result = await getGlobalModelInfo()

      // Re-check after the async fetch; bail if a session started in the gap.
      if ($activeSessionId.get()) {
        return
      }

      const configModel = typeof result.model === 'string' ? result.model : ''
      const configProvider = typeof result.provider === 'string' ? result.provider : ''

      // On a forced refresh (profile swap) always apply; on a normal refresh
      // apply when either:
      //   (a) the composer has no model selected yet (first boot / cleared), or
      //   (b) the config-driven value differs from what's stored — meaning the
      //       user changed config.yaml since the last time the app ran.
      const storedModel = $currentModel.get()
      const storedProvider = $currentProvider.get()
      const shouldUpdate =
        force ||
        !storedModel ||
        configModel !== storedModel ||
        configProvider !== storedProvider

      if (!shouldUpdate) {
        return
      }

      if (configModel) {
        setCurrentModel(configModel)
      }

      if (configProvider) {
        setCurrentProvider(configProvider)
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

      setCurrentModel(selection.model)
      setCurrentProvider(selection.provider)
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

        void queryClient.invalidateQueries({ queryKey: ['model-options', activeSessionId] })

        return true
      } catch (err) {
        setCurrentModel(prevModel)
        setCurrentProvider(prevProvider)
        updateModelOptionsCache(prevProvider, prevModel, !activeSessionId)
        notifyError(err, copy.modelSwitchFailed)

        return false
      }
    },
    [activeSessionId, copy.modelSwitchFailed, queryClient, requestGateway, updateModelOptionsCache]
  )

  return { refreshCurrentModel, selectModel, updateModelOptionsCache }
}
