import { type QueryClient } from '@tanstack/react-query'
import { useCallback, useRef } from 'react'

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

const FORCE_REFRESH_RETRY_DELAYS_MS = [250, 750, 2_000] as const

function delay(ms: number): Promise<void> {
  return new Promise(resolve => {
    setTimeout(resolve, ms)
  })
}

export function useModelControls({ activeSessionId, queryClient, requestGateway }: ModelControlsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const refreshRequestIdRef = useRef(0)

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
  // only fills an EMPTY selection so a user's pick (plain UI state in
  // $currentModel) survives the lifecycle refreshes that fire on boot / fresh
  // draft / session events. A live session owns the footer, so skip entirely.
  const refreshCurrentModel = useCallback(async (force = false) => {
    if ($activeSessionId.get()) {
      return
    }

    if (!force && $currentModel.get()) {
      return
    }

    const requestId = ++refreshRequestIdRef.current
    const retryDelays = force ? FORCE_REFRESH_RETRY_DELAYS_MS : []

    for (let attempt = 0; attempt <= retryDelays.length; attempt += 1) {
      try {
        const result = await getGlobalModelInfo()

        if (refreshRequestIdRef.current !== requestId) {
          return
        }

        if ($activeSessionId.get() || (!force && $currentModel.get())) {
          return
        }

        if (typeof result.model === 'string') {
          setCurrentModel(result.model)
        }

        if (typeof result.provider === 'string') {
          setCurrentProvider(result.provider)
        }

        return
      } catch {
        if (refreshRequestIdRef.current !== requestId) {
          return
        }

        const retryDelay = retryDelays[attempt]

        if (typeof retryDelay !== 'number') {
          return
        }

        await delay(retryDelay)

        if (refreshRequestIdRef.current !== requestId) {
          return
        }
      }
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
          value: `${selection.model} --provider ${selection.provider} --session`
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
