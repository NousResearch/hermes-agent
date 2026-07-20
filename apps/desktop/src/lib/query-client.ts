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
export const writeCache =
  <T>(key: QueryKey) =>
  (next: T | undefined | ((prev: T | undefined) => T | undefined)): void =>
    void queryClient.setQueryData<T>(key, next)
