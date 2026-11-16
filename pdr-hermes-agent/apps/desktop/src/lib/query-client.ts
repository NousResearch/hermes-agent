import type { ModelOptionProvider, ModelOptionsResult } from '@hermes/shared'
import { QueryClient, type QueryKey, replaceEqualDeep } from '@tanstack/react-query'

// Shared React Query client. Lives in its own module (not main.tsx) so non-React
// code — e.g. the profile store on a gateway swap — can invalidate cached,
// profile-scoped settings without importing the app entry point.
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 60_000
    }
  }
})

// A cache-only `model.options` read can answer for Nous before the backend has re-warmed its catalog /
// entitlement caches (`pricing_pending` / `free_tier_pending`): no Large / Free tags and, while the
// entitlement is unknown, every model locked. The model menu re-reads every 2s until a warm row lands,
// so a warm reply followed by a pending one made the tags blink off and on. Keep what the last warm row
// said until a warm one replaces it. Window plans only: only their rows carry `usage_levels`.
const WARM_NOUS_FIELDS = ['usage_levels', 'pricing', 'unavailable_models', 'free_tier', 'warning'] as const

const isPendingNous = (row: ModelOptionProvider) =>
  row.slug === 'nous' && Boolean(row.pricing_pending || row.free_tier_pending)

// A merged result keeps its pending flags, so its own row no longer reads as warm: remember which warm row
// it was built from, or a second pending reply in a row would drop the tags again.
const warmSource = new WeakMap<object, ModelOptionProvider>()

export function keepWarmNousRow(oldData: unknown, newData: unknown): unknown {
  const prev = oldData as ModelOptionsResult | undefined
  const next = newData as ModelOptionsResult | undefined

  const warm =
    (prev && warmSource.get(prev)) ??
    prev?.providers?.find(row => row.slug === 'nous' && row.usage_levels && !isPendingNous(row))

  if (!warm || !next?.providers?.some(isPendingNous)) {
    return replaceEqualDeep(oldData, newData)
  }

  const providers = next.providers.map(row => {
    if (!isPendingNous(row)) {
      return row
    }

    // The pending flags stay, so the menu's bounded re-read still picks up the warm row.
    const kept: Record<string, unknown> = {}

    for (const field of WARM_NOUS_FIELDS) {
      kept[field] = warm[field]
    }

    // The kept `usage_levels` still say `free` for subscription-billed rows; the ids
    // that keep the "Free" tag off them ride on `plan_usage` (whose gauges stay fresh), so carry them too.
    const subscribed = warm.plan_usage?.subscription_model_ids

    if (row.plan_usage && subscribed?.length) {
      kept.plan_usage = { ...row.plan_usage, subscription_model_ids: subscribed }
    }

    return { ...row, ...kept } as ModelOptionProvider
  })

  const merged = replaceEqualDeep(oldData, { ...next, providers })

  warmSource.set(merged, warm)

  return merged
}

queryClient.setQueryDefaults(['model-options'], { structuralSharing: keepWarmNousRow })

// Curried, setState-shaped cache writer for optimistic write-through: keeps
// mutation sites terse (`setX(next)` or `setX(prev => …)`) over one query key.
export const writeCache =
  <T>(key: QueryKey) =>
  (next: T | undefined | ((prev: T | undefined) => T | undefined)): void =>
    void queryClient.setQueryData<T>(key, next)

// Query-key roots that are NOT profile-scoped: account/billing, the theme
// marketplace, onboarding, and contrib log tails all read global or
// account-level state, so a profile/gateway swap must not refetch them. Any
// other key is treated as profile-scoped and invalidated -- a denylist is
// correctness-safe here: a root we forget to list just gets refetched (a small
// cost), whereas an allowlist that misses a profile-scoped key would paint the
// previous profile's data (a bug).
const PROFILE_INDEPENDENT_QUERY_ROOTS = new Set<string>([
  'billing',
  'marketplace-themes',
  'marketplace-themes-settings',
  'onboarding-model-options',
  'contrib-logs-tail'
])

// Invalidate profile-scoped query caches on a profile / gateway switch, leaving
// account/global caches intact. Replaces a keyless invalidateQueries() that
// refetched everything (billing, marketplace, onboarding) on every switch.
export function invalidateProfileScopedQueries(): void {
  void queryClient.invalidateQueries({
    predicate: query => {
      const root = query.queryKey[0]

      return typeof root !== 'string' || !PROFILE_INDEPENDENT_QUERY_ROOTS.has(root)
    }
  })
}
