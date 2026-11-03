import type { ModelOptionProvider } from '@hermes/shared'
import { planBlocks } from '@hermes/shared'

import { accountResetMs, modelResetMs, poolUsage } from './provider-limit'

/** Capacity, not a subscription balance. Missing catalog values stay unknown. */
export function contextLabel(tokens: number | null | undefined): string {
  if (!tokens || !Number.isFinite(tokens) || tokens < 0) {
    return 'Context unknown'
  }

  const value =
    tokens >= 1_000_000
      ? `${Number((tokens / 1_000_000).toFixed(2))}M`
      : tokens >= 1000
        ? `${Number((tokens / 1000).toFixed(1))}K`
        : `${tokens}`

  return `${value} context`
}

/** Unknown quota is not zero. Do not infer account exhaustion from model-scoped windows. */
export function availabilityRank(provider: ModelOptionProvider, model: string): number {
  if (provider.authenticated === false) {
    return 3
  }

  if (provider.unavailable_models?.includes(model)) {
    return 2
  }

  const pool = poolUsage(provider)

  if (pool && pool.accounts.every(account => account.state === 'limited' || account.state === 'unavailable')) {
    return 2
  }

  if (accountResetMs(provider) !== null || modelResetMs(provider, model) !== null || planBlocks(provider, model)) {
    return 2
  }

  const exhausted =
    !provider.usage?.accounts?.length &&
    provider.usage?.windows?.some(
      window =>
        window.scope !== 'model' &&
        window.used_percent >= 100 &&
        (!window.resets_at || Date.parse(window.resets_at) > Date.now())
    )

  return exhausted ? 2 : 0
}
