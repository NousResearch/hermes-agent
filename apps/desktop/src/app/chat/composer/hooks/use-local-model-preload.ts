import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { preloadLocalModel } from '@/api/local-models'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $currentModel, $currentProvider } from '@/store/session'

// Provider slug of the managed local server (inventory.py's _local_runtime_row).
const LOCAL_PROVIDER_SLUG = 'llamacpp'

/**
 * Ask the backend to start loading the managed local model once the user begins typing in a new
 * chat, so the first reply does not wait on the load. `typing` is the caller's "this is a visible,
 * enabled new-chat composer with text" edge.
 *
 * The backend owns whether it happens: `local_runtime.preload` is off by default, and a load that
 * would evict a resident model is skipped there. A failed request changes nothing for the user
 * (the model loads on the first message as usual), so it is dropped. The effect is keyed on the
 * model it would load, so it fires once per draft and again if the model is switched mid-draft.
 */
export function useLocalModelPreload(typing: boolean) {
  const model = useStore($currentModel)
  const provider = useStore($currentProvider)
  const target = typing && $localModelsEnabled.get() && provider === LOCAL_PROVIDER_SLUG && model ? model : null

  useEffect(() => {
    if (target) {
      void preloadLocalModel(target).catch(() => undefined)
    }
  }, [target])
}
