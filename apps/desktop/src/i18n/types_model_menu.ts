export interface ModelMenuTranslations {
  search: string
  noModels: string
  editModels: string
  followDefault: string
  refreshModels: string
  favorites: string
  addFavorite: string
  removeFavorite: string
  favoriteShortcut: string
  fast: string
  free: string
  cacheRead: string
  priceTitle: (input: string, output: string, cache: string) => string
  limited: string
  limitedUntil: (time: string) => string
  limitedTip: (provider: string, time: null | string) => string
  modelResets: (time: string) => string
  modelLimitedTip: (time: string) => string
  usageLeft: (percent: number, time: null | string) => string
  poolAccounts: (count: number) => string
  poolLimited: (limited: number, total: number) => string
  poolAccount: (number: number) => string
  poolUnknown: string
  poolUnavailable: string
  usageTip: (provider: string) => string
  usageWindow: (label: string, percent: number, time: null | string) => string
}
