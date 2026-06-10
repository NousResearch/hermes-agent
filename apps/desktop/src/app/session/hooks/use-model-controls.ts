import { type QueryClient } from '@tanstack/react-query'
import { useCallback, useRef } from 'react'

import { getGlobalModelInfo, setGlobalModel } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $currentModel, $currentProvider, setCurrentModel, setCurrentProvider } from '@/store/session'
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

  // Monotonic id per selectModel call. Each call captures its own token; once a
  // newer switch starts, every older call sees token !== switchSeq.current and
  // must leave the store/query cache alone — the newest switch owns that state.
  // Purely client-side: no protocol support needed for correct sequencing.
  const switchSeq = useRef(0)

  const refreshCurrentModel = useCallback(async (isCurrent: () => boolean = () => true) => {
    try {
      const result = await getGlobalModelInfo()

      // A newer model switch started while this refresh was in flight: the
      // snapshot may predate that switch, so applying it would clobber the
      // newer switch's optimistic state. Drop it — the newer call refreshes.
      if (!isCurrent()) {
        return
      }

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

  // Returns whether the switch succeeded so callers can await it before
  // applying follow-up changes (e.g. editing a model's reasoning/fast must land
  // on the right active model — bail rather than write to the previous one).
  const selectModel = useCallback(
    async (selection: ModelSelection): Promise<boolean> => {
      // In-flight correlation: a slow switch A must not commit or roll back
      // after a faster switch B already resolved — B owns the store/cache.
      // After every await, token !== switchSeq.current marks this call stale.
      const token = ++switchSeq.current
      const isCurrent = () => token === switchSeq.current
      const includeGlobal = selection.persistGlobal || !activeSessionId
      // Snapshot for rollback: the switch is applied optimistically, so a
      // failure must restore the prior model/provider (store + query cache)
      // rather than leave the UI showing a model the backend never selected.
      const prevModel = $currentModel.get()
      const prevProvider = $currentProvider.get()

      setCurrentModel(selection.model)
      setCurrentProvider(selection.provider)
      updateModelOptionsCache(selection.provider, selection.model, includeGlobal)

      try {
        if (activeSessionId) {
          const result = await requestGateway<{ error?: string }>('slash.exec', {
            session_id: activeSessionId,
            // Additive correlation id; the gateway will echo it back in a
            // parallel change. Nothing here depends on the echo — staleness
            // is decided purely by the client-side token above.
            task_id: crypto.randomUUID(),
            command: `/model ${selection.model} --provider ${selection.provider}${selection.persistGlobal ? ' --global' : ''}`
          })

          // slash.exec resolves _ok even when the backend rejected the live
          // switch — the failure rides in `result.error` (see slash.exec in
          // tui_gateway/server.py). Funnel it into the catch below so the
          // optimistic model is rolled back and the reason surfaced, rather
          // than leaving the UI on a model the backend never selected.
          if (result?.error) {
            throw new Error(result.error)
          }

          // Stale success: a newer switch took over while this one was in
          // flight. Don't touch the store/cache (neither commit nor refresh)
          // and report failure so callers don't chain follow-up edits onto a
          // selection the UI no longer shows.
          if (!isCurrent()) {
            return false
          }

          if (selection.persistGlobal) {
            void refreshCurrentModel(isCurrent)
          }

          void queryClient.invalidateQueries({
            queryKey: selection.persistGlobal ? ['model-options'] : ['model-options', activeSessionId]
          })

          return true
        }

        await setGlobalModel(selection.provider, selection.model)

        // Same stale-success policy as the session path above.
        if (!isCurrent()) {
          return false
        }

        void refreshCurrentModel(isCurrent)
        void queryClient.invalidateQueries({ queryKey: ['model-options'] })

        return true
      } catch (err) {
        // Stale failure: skip the rollback — the snapshot predates the newer
        // switch, so restoring it would clobber that switch's state (the
        // original race: A's late failure reverting to the pre-A model after
        // B succeeded). The error toast still fires either way: the user
        // asked for this switch and deserves to know it failed, and a toast
        // is fire-and-forget — unlike the store write, it can't corrupt the
        // newer switch's state.
        if (isCurrent()) {
          setCurrentModel(prevModel)
          setCurrentProvider(prevProvider)
          updateModelOptionsCache(prevProvider, prevModel, includeGlobal)
        }

        notifyError(err, copy.modelSwitchFailed)

        return false
      }
    },
    [activeSessionId, copy.modelSwitchFailed, queryClient, refreshCurrentModel, requestGateway, updateModelOptionsCache]
  )

  return { refreshCurrentModel, selectModel, updateModelOptionsCache }
}
