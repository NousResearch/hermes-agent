import { capabilityScoped, type ProfileScope } from '@/api/client'
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

export function searchProvider(profile: ProfileScope, source: string, query: string, cursor: string | null) {
  const params = new URLSearchParams({ source, q: query, limit: '20' })

  if (cursor) { params.set('cursor', cursor) }

  return catalogRequest<ProviderSearchPage>(profile, `search?${params}`)
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
