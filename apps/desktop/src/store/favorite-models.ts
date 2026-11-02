import { atom } from 'nanostores'

import { persistStringArray, storedStringArray } from '@/lib/storage'

import { modelVisibilityKey } from './model-visibility'

const STORAGE_KEY = 'hermes.desktop.favorite-models'

/** Stable `provider::model` key for a favorite. Reuses the visibility-store
 *  format so every model identity in the app is written the same way. */
export const favoriteModelKey = modelVisibilityKey

/**
 * Models the user starred into the Favorites section at the top of the model
 * catalog menu, ordered by when they were starred (first starred = first
 * shown). A renderer-local presentation preference in the same family as the
 * Edit Models shortlist and model presets: favorites only reshape THIS
 * dropdown's ordering, never the backend catalog. A key whose model the
 * current catalog no longer carries is kept, not dropped, so it returns when
 * its provider does.
 */
export const $favoriteModels = atom<string[]>(storedStringArray(STORAGE_KEY))

/** Whether a provider/model pair is currently a favorite. */
export function isModelFavorite(provider: string, model: string): boolean {
  return $favoriteModels.get().includes(favoriteModelKey(provider, model))
}

/** Replace the whole favorites list. Deduped, and an empty list clears the key. */
export function setFavoriteModels(keys: readonly string[]): void {
  const next = [...new Set(keys)]

  $favoriteModels.set(next)
  persistStringArray(STORAGE_KEY, next)
}

/** Favorite a provider/model pair (appended to the order) or unfavorite it. */
export function toggleFavoriteModel(provider: string, model: string): void {
  const key = favoriteModelKey(provider, model)
  const current = $favoriteModels.get()

  setFavoriteModels(current.includes(key) ? current.filter(entry => entry !== key) : [...current, key])
}
