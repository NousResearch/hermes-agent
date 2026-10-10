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
  catalogPrice: 'List price from the models.dev catalog; this provider did not report its own',
  contextTitle: (context: string) => `${context}-token context window`,
  vision: 'Accepts images',
  maxOutputLabel: (tokens: string) => `${tokens} out`,
  maxOutputTitle: (tokens: string) => `Replies up to ${tokens} tokens`,
  perThousandTitle: (input: string, output: string) => `Per 1K tokens: input ${input} · output ${output}`,
  tools: 'Can call tools',
  perThousandSuffix: '/1K',
  priceUnitTitle: 'Price Unit',
  priceUnitDesc: 'Show picker prices per million tokens or per thousand tokens.',
  priceUnitPerMillion: 'Per 1M',
  priceUnitPerThousand: 'Per 1K',
  localSetup: {
    title: 'Run locally · free, private',
    text: (model: string, size: string) => `${model} fits this machine · ${size} download`,
    action: 'Set up'
  },
  limited: 'Limited',
  limitedUntil: (time: string) => `Limited until ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} hit its usage limit. It resets at ${time}; you can still pick a model for after.`
      : `${provider} hit its usage limit. You can still pick a model for after it resets.`,
  modelResets: (time: string) => `resets ${time}`,
  modelLimitedTip: (time: string) =>
    `This model hit its own limit and resets at ${time}. Other models here still work.`,
  usageLeft: (percent: number, time: null | string) =>
    time ? `${percent}% left · resets ${time}` : `${percent}% left`,
  poolAccounts: (count: number) => `${count} ${count === 1 ? 'account' : 'accounts'}`,
  poolLimited: (limited: number, total: number) => `${limited}/${total} accounts limited`,
  poolAccount: (number: number) => `Account ${number}`,
  poolUnknown: 'Usage unavailable',
  poolUnavailable: 'Sign in again',
  usageTip: (provider: string) => `${provider} is close to its usage limit.`,
  usageWindow: (label: string, percent: number, time: null | string) =>
    time ? `${label}: ${percent}% left, resets ${time}` : `${label}: ${percent}% left`
}
