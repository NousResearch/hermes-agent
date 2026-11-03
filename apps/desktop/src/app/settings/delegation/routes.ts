import type { DelegationModelMap, DelegationModelRoute, DelegationProfileRoutes } from '@/hermes'

/** The fields the route dialog edits; every other key of a stored route round-trips untouched. */
export interface ModelRouteDraft {
  name: string
  description: string
  provider: string
  model: string
  reasoning_effort: string
  service_tier: string
}

export type RouteField = keyof ModelRouteDraft
export type RouteError = 'duplicate' | 'required' | 'spaces'

/** Saved as written: an explicit `normal` turns a provider's own Fast setting off; no tier leaves it alone. */
export const SERVICE_TIERS = ['normal', 'fast', 'ultrafast'] as const

export const emptyRouteDraft = (): ModelRouteDraft => ({
  name: '',
  description: '',
  provider: '',
  model: '',
  reasoning_effort: '',
  service_tier: ''
})

export const draftFromRoute = (route: DelegationModelRoute): ModelRouteDraft => ({
  name: route.name ?? '',
  description: route.description ?? '',
  provider: route.provider ?? '',
  model: route.model ?? '',
  reasoning_effort: route.reasoning_effort ?? '',
  service_tier: route.service_tier ?? ''
})

/** Field → problem, empty when the draft can be saved. `taken` = names of the OTHER routes.
 *  A stored route with only a base_url (hand-written YAML) keeps it and needs no provider. */
export function validateRouteDraft(
  draft: ModelRouteDraft,
  taken: readonly string[],
  hasBaseUrl = false
): Partial<Record<RouteField, RouteError>> {
  const errors: Partial<Record<RouteField, RouteError>> = {}
  const name = draft.name.trim()

  if (!name) {
    errors.name = 'required'
  } else if (/\s/.test(name)) {
    errors.name = 'spaces'
  } else if (taken.includes(name)) {
    errors.name = 'duplicate'
  }

  for (const field of ['description', 'model'] as const) {
    if (!draft[field].trim()) {
      errors[field] = 'required'
    }
  }

  if (!draft.provider.trim() && !hasBaseUrl) {
    errors.provider = 'required'
  }

  return errors
}

/** The stored route for `draft`: edited fields replace, blank optional fields drop, unknown keys stay. An edited
 *  route names the route it came from (`original_name`), so the server keeps its stored key through a rename. */
export function routeFromDraft(draft: ModelRouteDraft, previous?: DelegationModelRoute): DelegationModelRoute {
  const next: Record<string, unknown> = { ...previous, ...(previous ? { original_name: previous.name } : {}) }

  for (const field of Object.keys(draft) as RouteField[]) {
    const value = draft[field].trim()

    if (value) {
      next[field] = value
    } else {
      delete next[field]
    }
  }

  return next as DelegationModelRoute
}

/** A default route always exists while routes do, so the subagent model below is plainly unused. */
function withDefault(options: DelegationModelRoute[], preferred: null | string): DelegationModelMap {
  const names = options.map(option => option.name)

  return { options, default: preferred && names.includes(preferred) ? preferred : (names[0] ?? null) }
}

/** Insert (`original` null) or replace the route called `original`; a rename carries the default along. */
export function upsertRoute(map: DelegationModelMap, original: null | string, route: DelegationModelRoute) {
  const index = original === null ? -1 : map.options.findIndex(option => option.name === original)
  const options = index < 0 ? [...map.options, route] : map.options.map((o, i) => (i === index ? route : o))
  const preferred = original !== null && map.default === original ? route.name : map.default

  return withDefault(options, preferred)
}

export const removeRoute = (map: DelegationModelMap, name: string) =>
  withDefault(
    map.options.filter(option => option.name !== name),
    map.default === name ? null : map.default
  )

export const setDefaultRoute = (map: DelegationModelMap, name: string) => withDefault(map.options, name)

/** Check or uncheck a profile; the first checked profile becomes the default when none is. */
export function toggleProfileRoute(
  routes: DelegationProfileRoutes,
  name: string,
  on: boolean
): DelegationProfileRoutes {
  const profiles = on
    ? routes.profiles.some(p => p.name === name)
      ? routes.profiles
      : [...routes.profiles, { name }]
    : routes.profiles.filter(p => p.name !== name)

  const names = profiles.map(p => p.name)
  const preferred = routes.default && names.includes(routes.default) ? routes.default : null

  return { profiles, default: preferred ?? names[0] ?? null }
}

export function describeProfileRoute(
  routes: DelegationProfileRoutes,
  name: string,
  description: string
): DelegationProfileRoutes {
  return {
    ...routes,
    profiles: routes.profiles.map(p => {
      if (p.name !== name) {
        return p
      }

      const text = description.trim()

      return text ? { name, description: text } : { name }
    })
  }
}

export const setDefaultProfileRoute = (routes: DelegationProfileRoutes, name: string): DelegationProfileRoutes => ({
  ...routes,
  default: name
})
