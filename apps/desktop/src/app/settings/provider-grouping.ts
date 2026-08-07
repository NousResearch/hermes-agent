// Pure, framework-free helpers for grouping + ordering the Provider Manager's
// left-pane provider list. Unit-tested in provider-grouping.test.ts without
// React. Keeps all grouping/sorting logic in one place so the nav component
// stays a thin renderer.

import type { ModelOptionProvider } from '@/types/hermes'

export type ProviderGroupId = 'local' | 'configured' | 'unconfigured'

export interface ProviderGroup {
  id: ProviderGroupId
  providers: ModelOptionProvider[]
}

const GROUP_ORDER: ProviderGroupId[] = ['local', 'configured', 'unconfigured']

/**
 * Classify a single provider into a group.
 *
 * The desktop catalog's `ModelOptionProvider` does NOT expose an `is_local`
 * field (unlike the web-UI plan) and does not surface `base_url`/`api_key`
 * (those live in the user's config, not the catalog row). So classification
 * uses the fields the catalog DOES expose:
 *   - `local`: `slug === 'local'` (the canonical self-hosted/local provider).
 *   - `configured`: has usable credentials (`authenticated === true`), OR is a
 *     user-defined custom provider (always user-set-up), OR is an enabled
 *     provider that already exposes models.
 *   - `unconfigured`: everything else (unauthenticated, no models yet).
 */
export function classifyProvider(provider: ModelOptionProvider): ProviderGroupId {
  if (provider.slug === 'local') {
    return 'local'
  }

  const configured =
    provider.authenticated === true ||
    provider.is_user_defined === true ||
    (provider.enabled !== false && (provider.models?.length ?? 0) > 0)

  return configured ? 'configured' : 'unconfigured'
}

/**
 * Group providers into Local → Configured → Unconfigured, each group sorted
 * with active providers first, then alphabetically by name. Returns only
 * non-empty groups (preserves group order).
 */
export function groupProviders(providers: readonly ModelOptionProvider[]): ProviderGroup[] {
  const buckets: Record<ProviderGroupId, ModelOptionProvider[]> = {
    local: [],
    configured: [],
    unconfigured: []
  }

  for (const provider of providers) {
    buckets[classifyProvider(provider)].push(provider)
  }

  const sortGroup = (list: ModelOptionProvider[]) =>
    [...list].sort((a, b) => {
      const aActive = a.enabled !== false ? 1 : 0
      const bActive = b.enabled !== false ? 1 : 0

      if (aActive !== bActive) {
        return bActive - aActive
      }

      return a.name.localeCompare(b.name)
    })

  return GROUP_ORDER.map(id => ({ id, providers: sortGroup(buckets[id]) })).filter(group => group.providers.length > 0)
}

/**
 * Flatten grouped providers back into a single ordered list (used for
 * keyboard navigation indices so arrow keys still traverse every provider
 * regardless of visual group headers).
 */
export function flattenGroups(groups: readonly ProviderGroup[]): ModelOptionProvider[] {
  return groups.flatMap(group => group.providers)
}

/**
 * True when a provider can be configured inline from the Provider Manager —
 * i.e. it is a built-in provider using the `api_key` auth flow that the user
 * hasn't set up yet. Such providers expose a `key_env` (the env var to write
 * the API key into) and are surfaced by the catalog with `includeUnconfigured`.
 *
 * OAuth / external providers are NOT inline-configurable (they need the CLI /
 * onboarding OAuth flow), so they are excluded here and stay hidden from the
 * Unconfigured group.
 */
export function isConfigurableProvider(provider: ModelOptionProvider): boolean {
  return (
    !provider.is_user_defined &&
    provider.auth_type === 'api_key' &&
    Boolean(provider.key_env) &&
    provider.authenticated !== true
  )
}
