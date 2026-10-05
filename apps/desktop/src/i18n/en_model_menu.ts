import type { Translations } from './types'

export const enModelMenu: Translations['shell']['modelMenu'] = {
  search: 'Search models',
  noModels: 'No models found',
  editModels: 'Edit models…',
  followDefault: 'Use Settings default',
  refreshModels: 'Refresh models',
  favorites: 'Favorites',
  addFavorite: 'Add to favorites',
  removeFavorite: 'Remove from favorites',
  favoriteShortcut: '⇧ Click',
  fast: 'Fast',
  free: 'free',
  cacheRead: 'cached read',
  priceTitle: (input: string, output: string, cache: string) =>
    `Input ${input}/Mtok · Output ${output}/Mtok` + (cache ? ` · Cached read ${cache}/Mtok` : ''),
  limited: 'Limited',
  limitedUntil: (time: string) => `Limited until ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} hit its usage limit. It resets at ${time}; you can still pick a model for after.`
      : `${provider} hit its usage limit. You can still pick a model for after it resets.`,
  modelResets: (time: string) => `resets ${time}`,
  modelLimitedTip: (time: string) => `This model hit its own limit and resets at ${time}. Other models here still work.`
}
