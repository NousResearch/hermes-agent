import { hermesApi, profileScoped } from './client'
import type { LocalModelsScope } from './local-models'

export interface LocalModelSettingsRequest {
  expected_fingerprint: string
  expected_revision: number
  overrides: Record<string, unknown>
}

export interface LocalModelResetRequest {
  expected_fingerprint: string
  expected_revision: number
  keys: string[] | null
}

/** Read model-bound settings without letting the renderer choose a file or profile root. */
export function getLocalModelSettings(modelId: string, scope?: LocalModelsScope): Promise<unknown> {
  return hermesApi<unknown>({ ...(scope ?? profileScoped()), path: `/api/local-models/settings/${encodeURIComponent(modelId)}` })
}

/** Replace saved overrides for one exact model artifact revision. */
export function saveLocalModelSettings(modelId: string, request: LocalModelSettingsRequest, scope?: LocalModelsScope): Promise<unknown> {
  return hermesApi<unknown>({
    ...(scope ?? profileScoped()),
    body: request,
    method: 'PUT',
    path: `/api/local-models/settings/${encodeURIComponent(modelId)}`
  })
}

/** Delete all saved overrides, or the selected keys, for one exact model artifact revision. */
export function resetLocalModelSettings(modelId: string, request: LocalModelResetRequest, scope?: LocalModelsScope): Promise<unknown> {
  return hermesApi<unknown>({
    ...(scope ?? profileScoped()),
    body: request,
    method: 'POST',
    path: `/api/local-models/settings/${encodeURIComponent(modelId)}/reset`
  })
}
