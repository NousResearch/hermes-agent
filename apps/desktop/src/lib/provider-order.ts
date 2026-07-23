import type { ModelOptionProvider } from '@/types/hermes'

type ProviderOrderFields = Pick<ModelOptionProvider, 'name' | 'slug'> &
  Pick<Partial<ModelOptionProvider>, 'authenticated'>

/**
 * Whether Hermes has usable credentials for this catalog row.
 *
 * Authenticated rows from `list_authenticated_providers` often omit the flag
 * (treat as connected). Explicit `authenticated: false` is used for
 * configured-but-deauthed skeletons that still need a setup affordance.
 */
export function isProviderConnected(provider: Pick<Partial<ModelOptionProvider>, 'authenticated'>): boolean {
  return provider.authenticated !== false
}

/**
 * Model-picker group order:
 * 1. Current provider (session / global selection)
 * 2. Connected providers
 * 3. Needs-setup / unauthenticated rows
 * 4. Alphabetical by display name within each tier
 *
 * Intentionally stable: switching models only moves the "current" tier, not
 * the relative order of peers (avoids the list thrashing the old backend
 * is_current-first ordering caused).
 */
export function compareModelProviders(
  a: ProviderOrderFields,
  b: ProviderOrderFields,
  currentSlug?: null | string
): number {
  const current = (currentSlug || '').trim().toLowerCase()
  const aSlug = (a.slug || '').trim().toLowerCase()
  const bSlug = (b.slug || '').trim().toLowerCase()

  if (current) {
    if (aSlug === current && bSlug !== current) {
      return -1
    }

    if (bSlug === current && aSlug !== current) {
      return 1
    }
  }

  const aConnected = isProviderConnected(a)
  const bConnected = isProviderConnected(b)

  if (aConnected !== bConnected) {
    return aConnected ? -1 : 1
  }

  return a.name.localeCompare(b.name)
}

/** Sort a provider list with {@link compareModelProviders}. Returns a new array. */
export function sortModelProviders<T extends ProviderOrderFields>(providers: T[], currentSlug?: null | string): T[] {
  return [...providers].sort((a, b) => compareModelProviders(a, b, currentSlug))
}
