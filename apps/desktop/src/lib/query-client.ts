import { QueryClient, type QueryKey } from '@tanstack/react-query'

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

// Curried, setState-shaped cache writer for optimistic write-through: keeps
// mutation sites terse (`setX(next)` or `setX(prev => …)`) over one query key.
// Writing `undefined` does NOT clear the entry — React Query's setQueryData
// treats undefined as a bail-out. To drop cached data use
// queryClient.resetQueries / removeQueries.
export const writeCache =
  <T>(key: QueryKey) =>
  (next: T | undefined | ((prev: T | undefined) => T | undefined)): void =>
    void queryClient.setQueryData<T>(key, next)

// Key of the shared profile config record (`GET /api/config`). Owned here —
// not in app/hooks/use-config-record.ts, which re-exports it — because the
// profile-switch boundary below needs it and lib must not import from app.
export const HERMES_CONFIG_QUERY_KEY = ['hermes-config-record'] as const

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
//
// The config record gets a hard RESET instead of an invalidation. Settings
// surfaces seed editable drafts (and autosave whole records) from that cache,
// and invalidate() keeps the previous profile's record visible — and seedable —
// while the new profile's fetch is in flight, so a mounted panel could still
// read profile A's record after a switch to B. resetQueries drops the data to
// undefined immediately and refetches for active observers, closing that window
// for every consumer at once, regardless of which panels happen to be mounted.
// (`setQueryData(key, undefined)` cannot do this: React Query treats an
// undefined value as a bail-out, not a delete.)
export function invalidateProfileScopedQueries(): void {
  void queryClient.resetQueries({ queryKey: HERMES_CONFIG_QUERY_KEY })
  void queryClient.invalidateQueries({
    predicate: query => {
      const root = query.queryKey[0]

      if (typeof root !== 'string') {
        return true
      }

      // The config record was hard-reset above; re-invalidating it here would
      // cancel and restart its in-flight refetch for nothing.
      return !PROFILE_INDEPENDENT_QUERY_ROOTS.has(root) && root !== HERMES_CONFIG_QUERY_KEY[0]
    }
  })
}
