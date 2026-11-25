import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { getLocalModelsStatus, preloadLocalModel } from '@/api/local-models'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $preloadingModel } from '@/store/local-model-preload'
import { $currentModel, $currentProvider } from '@/store/session'

// Provider slug of the managed local server (inventory.py's _local_runtime_row).
const LOCAL_PROVIDER_SLUG = 'llamacpp'

const POLL_MS = 1_000
// Consecutive polls that miss the model: the load has not shown up yet, or has ended.
const MISSES_TO_STOP = 3
const MAX_POLLS = 600

/**
 * Ask the backend to start loading the managed local model once the user begins typing in a new
 * chat, so the first reply does not wait on the load, and publish it in `$preloadingModel` while the
 * server reports it loading. `typing` is the caller's "this is a visible, enabled new-chat composer
 * with text" edge.
 *
 * The backend owns whether it happens: `local_runtime.preload` is off by default, and a load that
 * would evict a resident model is skipped there. A failed request changes nothing for the user (the
 * model loads on the first message as usual), so it is dropped. The effect is keyed on the model it
 * would load, so it fires once per draft and again if the model is switched mid-draft; its cleanup
 * (the draft emptied or sent) cancels the poll and clears the hint.
 */
export function useLocalModelPreload(typing: boolean) {
  const model = useStore($currentModel)
  const provider = useStore($currentProvider)
  const target = typing && $localModelsEnabled.get() && provider === LOCAL_PROVIDER_SLUG && model ? model : null

  useEffect(() => {
    if (!target) {
      return undefined
    }

    let cancelled = false
    let timer: number | undefined

    const watchLoad = async () => {
      let misses = 0

      for (let polls = 0; !cancelled && misses < MISSES_TO_STOP && polls < MAX_POLLS; polls++) {
        const loading = await getLocalModelsStatus()
          // Either signal: `loading` is SSE-fed and starts listening on the first poll, so it can miss a
          // load that began just before; the router's own status in `loaded_models` cannot.
          .then(status => Boolean(status.loading?.[target]) || status.loaded_models?.[target] === 'loading')
          // A busy backend times out the status call. That says nothing about the load, so it neither
          // ends the watch nor changes the hint; only an answer that says "not loading" does.
          .catch(() => undefined)

        if (cancelled) {
          return
        }

        if (loading !== undefined) {
          misses = loading ? 0 : misses + 1
          $preloadingModel.set(loading ? target : null)
        }

        await new Promise<void>(resolve => {
          timer = window.setTimeout(resolve, POLL_MS)
        })
      }

      $preloadingModel.set(null)
    }

    void preloadLocalModel(target)
      .then(({ status }) => (status === 'started' || status === 'held' ? watchLoad() : undefined))
      .catch(() => undefined)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
      $preloadingModel.set(null)
    }
  }, [target])
}
