import { capabilityScoped, type ProfileScope, profileScopeKey } from '@/api/client'
import type { SkillHubResult } from '@/types/hermes'
import type { CatalogAction, CatalogOperation, CatalogPreview, CatalogProvider, CatalogReview } from '@/types/skill-catalog'

export interface ProviderSearchPage {
  results: (SkillHubResult & { provider?: CatalogProvider })[]
  next_cursor?: string | null
  timed_out?: string[]
  provider_errors?: { source: string; code: string; message: string }[]
}

/** Capture both routing pins before any asynchronous work. */
export function catalogRequest<T>(profile: ProfileScope, path: string, body?: unknown): Promise<T> {
  return window.hermesDesktop.api<T>({
    ...capabilityScoped(profile),
    path: `/api/skills/hub/${path}`,
    timeoutMs: 65_000,
    ...(body === undefined ? {} : { method: 'POST', body })
  })
}

// Core admits one catalog search per profile, including searches from different query keys.
const searchQueues = new Map<string, Promise<unknown>>()

export function searchProvider(profile: ProfileScope, source: string, query: string, cursor: string | null) {
  const pin = capabilityScoped(profile)
  const scope = profileScopeKey(pin)
  const params = new URLSearchParams({ source, q: query, limit: '20' })

  if (cursor) { params.set('cursor', cursor) }

  const request = (searchQueues.get(scope) ?? Promise.resolve()).then(() => catalogRequest<ProviderSearchPage>(pin, `search?${params}`))
  const tail = request.then(() => undefined, () => undefined)
  searchQueues.set(scope, tail)
  void tail.then(() => {
    if (searchQueues.get(scope) === tail) { searchQueues.delete(scope) }
  })

  return request
}

export function previewProvider(profile: ProfileScope, identifier: string, version?: string) {
  const params = new URLSearchParams({ identifier })

  if (version) { params.set('version', version) }

  return catalogRequest<CatalogPreview>(profile, `preview?${params}`)
}

export function reviewProvider(profile: ProfileScope, identifier: string, action: CatalogAction, version?: string) {
  return catalogRequest<CatalogReview>(profile, 'review', { identifier, action, ...(action !== 'uninstall' && version ? { version } : {}) })
}

export function confirmProvider(profile: ProfileScope, review: CatalogReview, acknowledgements: string[]) {
  return catalogRequest<CatalogOperation>(profile, 'confirm', { identifier: review.identifier, action: review.action, review_token: review.review_token, acknowledgements })
}

export function readProviderOperation(profile: ProfileScope, identifier: string, operationId: string) {
  return catalogRequest<CatalogOperation>(profile, `operation?${new URLSearchParams({ identifier, operation_id: operationId })}`)
}
