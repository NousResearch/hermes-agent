import { atom } from 'nanostores'

import { persistStringArray, storedStringArray } from '@/lib/storage'

import { modelVisibilityKey } from './model-visibility'

const STORAGE_KEY = 'hermes.desktop.pinned-models'

/** Stable `provider::model` key for a pin. Reuses the visibility-store format
 *  so every model identity in the app is written the same way. */
export const pinnedModelKey = modelVisibilityKey

/**
 * Models the user pinned to the top of the model catalog menu, ordered by pin
 * time (first pinned = first shown). A renderer-local presentation preference
 * in the same family as the Edit Models shortlist and model presets: pins only
 * reshape THIS dropdown's ordering, never the backend catalog or another
 * surface's picker. A key whose model the current catalog no longer carries is
 * kept, not dropped, so it returns when its provider does.
 */
export const $pinnedModels = atom<string[]>(storedStringArray(STORAGE_KEY))

/** Whether a provider/model pair is currently pinned. */
export function isModelPinned(provider: string, model: string): boolean {
  return $pinnedModels.get().includes(pinnedModelKey(provider, model))
}

/** Replace the whole pin list. Deduped, and an empty list clears the key. */
export function setPinnedModels(keys: readonly string[]): void {
  const next = [...new Set(keys)]

  $pinnedModels.set(next)
  persistStringArray(STORAGE_KEY, next)
}

/** Pin a provider/model pair (appended to the order) or unpin it. */
export function togglePinnedModel(provider: string, model: string): void {
  const key = pinnedModelKey(provider, model)
  const current = $pinnedModels.get()

  setPinnedModels(current.includes(key) ? current.filter(entry => entry !== key) : [...current, key])
}
