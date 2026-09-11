import { atom } from 'nanostores'

import { persistStringArray, storedStringArray } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.favorite-models'

/** Models the user starred, as ordered `provider::model` keys. The key is the
 *  same identity the visibility shortlist uses (`modelVisibilityKey`), so both
 *  curation surfaces always name the same row. Order is meaningful: it is the
 *  order the user starred them in, and the order the Favorites section paints.
 *  A key whose model is not in the current catalog simply has no row to show —
 *  it is kept, not dropped, so it returns when its provider does. */
export const $favoriteModels = atom<string[]>(storedStringArray(STORAGE_KEY))

/** Replace the whole star list. Deduped, and an empty list clears the key. */
export function setFavoriteModels(keys: readonly string[]): void {
  const next = [...new Set(keys)]

  $favoriteModels.set(next)
  persistStringArray(STORAGE_KEY, next)
}

/** Add or remove one key, APPENDING new stars so the user's order survives.
 *  Pure so the ordering contract can be tested without a store. */
export function toggleFavoriteKey(keys: readonly string[], key: string): string[] {
  return keys.includes(key) ? keys.filter(existing => existing !== key) : [...keys, key]
}

/** Star or unstar one model and persist the result. */
export function toggleFavoriteModel(key: string): void {
  setFavoriteModels(toggleFavoriteKey($favoriteModels.get(), key))
}
