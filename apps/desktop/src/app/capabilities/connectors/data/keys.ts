// Query keys for the Connectors page, in their own module so the operation
// store can invalidate without importing the hooks (and the hooks' React and
// TanStack imports) into the gateway event path.
//
// Shape: `['connectors', <scope key>, <read>, …]`. The scope key sits in front
// of the read so `['connectors']` invalidates every scope, `['connectors', key]`
// invalidates one profile, and one connector's tool list stays addressable on
// its own. `connectors` is NOT in `PROFILE_INDEPENDENT_QUERY_ROOTS`, so a
// profile switch already drops the lot (AGENTS.md: scope in the key).

import { type ProfileScope, profileScopeKey } from '@/hermes'

export const CONNECTORS_QUERY_ROOT = ['connectors'] as const

const scoped = (scope: ProfileScope, ...rest: string[]) =>
  [...CONNECTORS_QUERY_ROOT, profileScopeKey(scope), ...rest] as const

/** Everything this page reads for one scope. The settle of a connect and the
 *  page's Refresh both invalidate here. */
export const connectorsScopeQueryKey = (scope: ProfileScope) => scoped(scope)

export const connectorsListQueryKey = (scope: ProfileScope) => scoped(scope, 'list')

export const connectorsCatalogQueryKey = (scope: ProfileScope) => scoped(scope, 'catalog')

export const connectorsAccountsQueryKey = (scope: ProfileScope) => scoped(scope, 'accounts')

export const connectorsPolicyQueryKey = (scope: ProfileScope) => scoped(scope, 'policy')

export const connectorToolsQueryKey = (scope: ProfileScope, slug: string) => scoped(scope, 'tools', slug)
