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
  /** Tooltip line when a price is the models.dev list price, not the provider's own. */
  catalogPrice: string
  contextTitle: (context: string) => string
  vision: string
  maxOutputLabel: (tokens: string) => string
  maxOutputTitle: (tokens: string) => string
  perThousandTitle: (input: string, output: string) => string
  tools: string
  /** Appended to a picker price shown per 1K tokens, so the unit is never ambiguous. */
  perThousandSuffix: string
  /** Settings → Appearance row choosing that unit. */
  priceUnitTitle: string
  priceUnitDesc: string
  priceUnitPerMillion: string
  priceUnitPerThousand: string
  localSetup: { title: string; text: (model: string, size: string) => string; action: string }
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
