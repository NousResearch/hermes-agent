import { capabilityScoped, hermesApi, type ProfileScope } from './client'

export type FoundryCredential =
  | { credential_source: 'default'; scope: string }
  | ({ credential_source: 'azure_cli'; scope: string } & (
      | { tenant_id?: never; subscription?: never }
      | { tenant_id: string; subscription?: never }
      | { tenant_id?: never; subscription: string }
    ))

export type FoundryAuth = { kind: 'api_key'; api_key?: string } | { kind: 'entra_id'; entra: FoundryCredential }

export interface FoundryUpdate {
  base_url: string
  model: string
  api_mode: 'chat_completions' | 'codex_responses' | 'anthropic_messages'
  auth: FoundryAuth
}

export interface FoundrySettings extends FoundryUpdate {
  api_key_set: boolean
  is_current: boolean
}

export interface FoundryValidation {
  ok: boolean
  reachable: boolean
  message: string
  resolved_base_url?: string
  transport_checked?: FoundryUpdate['api_mode']
}

const path = '/api/providers/azure-foundry'

export function getFoundrySettings(scope?: ProfileScope): Promise<FoundrySettings> {
  return hermesApi<FoundrySettings>({ ...capabilityScoped(scope), path })
}

export function saveFoundrySettings(body: FoundryUpdate, scope?: ProfileScope): Promise<FoundrySettings> {
  return hermesApi<FoundrySettings>({ ...capabilityScoped(scope), path, method: 'PUT', body })
}

export function validateFoundrySettings(body: FoundryUpdate, scope?: ProfileScope): Promise<FoundryValidation> {
  return hermesApi<FoundryValidation>({ ...capabilityScoped(scope), path: `${path}/validate`, method: 'POST', body })
}

export function discoverFoundryDeployments(
  body: FoundryUpdate,
  scope?: ProfileScope
): Promise<{ deployments: string[] }> {
  return hermesApi<{ deployments: string[] }>({
    ...capabilityScoped(scope),
    path: `${path}/deployments`,
    method: 'POST',
    body
  })
}
