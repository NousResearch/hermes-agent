import { type QueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'

import { getGlobalModelInfo, setGlobalModel } from '@/hermes'
import { notifyError } from '@/store/notifications'
import { setCurrentModel, setCurrentProvider } from '@/store/session'
import type { ModelOptionsResponse } from '@/types/hermes'

interface ModelSelection {
  model: string
  persistGlobal: boolean
  provider: string
}

interface ModelControlsOptions {
  activeSessionId: string | null
  queryClient: QueryClient
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function useModelControls({ activeSessionId, queryClient, requestGateway }: ModelControlsOptions) {
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

  const refreshCurrentModel = useCallback(async () => {
    try {
      const result = await getGlobalModelInfo()

      if (typeof result.model === 'string') {
        setCurrentModel(result.model)
      }

      if (typeof result.provider === 'string') {
        setCurrentProvider(result.provider)
      }
    } catch {
      // The delayed session.info event still updates this once the agent is ready.
    }
  }, [])

  const selectModel = useCallback(
    (selection: ModelSelection) => {
      setCurrentModel(selection.model)
      setCurrentProvider(selection.provider)
      updateModelOptionsCache(selection.provider, selection.model, selection.persistGlobal || !activeSessionId)

      void (async () => {
        try {
          if (activeSessionId) {
            await requestGateway('slash.exec', {
              session_id: activeSessionId,
              command: `/model ${selection.model} --provider ${selection.provider}${selection.persistGlobal ? ' --global' : ''}`
            })

            if (selection.persistGlobal) {
              void refreshCurrentModel()
            }

            void queryClient.invalidateQueries({
              queryKey: selection.persistGlobal ? ['model-options'] : ['model-options', activeSessionId]
            })

            return
          }

          await setGlobalModel(selection.provider, selection.model)
          void refreshCurrentModel()
          void queryClient.invalidateQueries({ queryKey: ['model-options'] })
        } catch (err) {
          notifyError(err, 'Model switch failed')
        }
      })()
    },
    [activeSessionId, queryClient, refreshCurrentModel, requestGateway, updateModelOptionsCache]
  )

  return { refreshCurrentModel, selectModel, updateModelOptionsCache }
}
