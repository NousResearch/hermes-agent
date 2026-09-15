import type {
  ConfigSchemaResponse,
  CustomEndpointsResponse,
  CustomEndpointUpdate,
  CustomEndpointValidationResponse,
  EnvVarInfo,
  HermesConfig,
  HermesConfigRecord,
  LogsResponse,
  OAuthPollResponse,
  OAuthProvidersResponse,
  OAuthStartResponse,
  OAuthSubmitResponse,
  StatusResponse
} from '@/types/hermes'

import { capabilityScoped, hermesApi, type ProfileScope, profileScoped, STARTUP_REQUEST_TIMEOUT_MS } from './client'

const configReadOrigins = new WeakMap<object, { connectionId?: string; profile?: string }>()

/** Snapshot the `(connectionId, profile)` that served a config GET. */
export function bindConfigReadOrigin(
  record: object,
  origin: { connectionId?: string; profile?: string }
): void {
  configReadOrigins.set(record, origin)
}

export function peekConfigReadOrigin(
  record: object | undefined | null
): { connectionId?: string; profile?: string } | undefined {
  return record ? configReadOrigins.get(record) : undefined
}

/**
 * Route a config write to the identity that served the matching read.
 * An explicit `{ connectionId, profile }` pin wins. A GET-derived record
 * keeps its captured origin even after the registry primary changes.
 * Unbound writes (no captured origin, no object pin) keep the live ambient
 * capability scope — e.g. reset-to-defaults on the current connection.
 */
export function resolveConfigWriteScope(
  record: object | undefined,
  requestScope?: ProfileScope
): { connectionId?: string; profile?: string } {
  if (requestScope && typeof requestScope === 'object') {
    return capabilityScoped(requestScope)
  }

  const captured = peekConfigReadOrigin(record)

  if (captured) {
    const profile =
      typeof requestScope === 'string' && requestScope.trim() ? requestScope.trim() : captured.profile

    return {
      ...(profile ? { profile } : {}),
      ...(captured.connectionId ? { connectionId: captured.connectionId } : {})
    }
  }

  return capabilityScoped(requestScope)
}

export function getStatus(): Promise<StatusResponse> {
  return hermesApi<StatusResponse>({
    ...profileScoped(),
    path: '/api/status'
  })
}

export function getLogs(params: {
  component?: string
  file?: string
  level?: string
  lines?: number
  search?: string
}): Promise<LogsResponse> {
  const query = new URLSearchParams()

  if (params.file) {
    query.set('file', params.file)
  }

  if (typeof params.lines === 'number') {
    query.set('lines', String(params.lines))
  }

  if (params.level && params.level !== 'ALL') {
    query.set('level', params.level)
  }

  if (params.component && params.component !== 'all') {
    query.set('component', params.component)
  }

  if (params.search) {
    query.set('search', params.search)
  }

  const suffix = query.toString()

  return hermesApi<LogsResponse>({
    ...profileScoped(),
    path: suffix ? `/api/logs?${suffix}` : '/api/logs'
  })
}

export function getHermesConfig(profile?: string): Promise<HermesConfig> {
  return hermesApi<HermesConfig>({
    ...profileScoped(profile),
    path: '/api/config',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export async function getHermesConfigRecord(
  profile?: ProfileScope,
  { includeDefaults = true }: { includeDefaults?: boolean } = {}
): Promise<HermesConfigRecord> {
  const origin = capabilityScoped(profile)
  const record = await window.hermesDesktop.api<HermesConfigRecord>({
    ...origin,
    path: includeDefaults ? '/api/config' : '/api/config?include_defaults=false'
  })

  if (record && typeof record === 'object') {
    bindConfigReadOrigin(record, origin)
  }

  return record
}

export function getHermesConfigDefaults(): Promise<HermesConfigRecord> {
  return hermesApi<HermesConfigRecord>({
    ...profileScoped(),
    path: '/api/config/defaults',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function getHermesConfigSchema(profile?: null | string): Promise<ConfigSchemaResponse> {
  return hermesApi<ConfigSchemaResponse>({
    ...profileScoped(profile),
    path: '/api/config/schema'
  })
}

export function saveHermesConfig(
  config: HermesConfigRecord,
  profile?: ProfileScope,
  { preserveLanguage = false }: { preserveLanguage?: boolean } = {}
): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...resolveConfigWriteScope(config, profile),
    path: preserveLanguage ? '/api/config?preserve_language=true' : '/api/config',
    method: 'PUT',
    body: { config }
  })
}

/** Capability-scoped counterpart of saveHermesConfig — writes the config of
 *  the profile/connection the Capabilities scope selector points at (possibly
 *  on another registered gateway), mirroring getHermesConfigRecord. */
export function saveHermesConfigRecord(config: HermesConfigRecord, profile?: ProfileScope): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...resolveConfigWriteScope(config, profile),
    path: '/api/config',
    method: 'PUT',
    body: { config }
  })
}

export function getEnvVars(profile?: null | string): Promise<Record<string, EnvVarInfo>> {
  return hermesApi<Record<string, EnvVarInfo>>({
    ...profileScoped(profile),
    path: '/api/env'
  })
}

export function setEnvVar(key: string, value: string, profile?: ProfileScope): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/env',
    method: 'PUT',
    body: { key, value }
  })
}

export function deleteEnvVar(key: string, profile?: ProfileScope): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/env',
    method: 'DELETE',
    body: { key }
  })
}

export function revealEnvVar(key: string, profile?: ProfileScope): Promise<{ key: string; value: string }> {
  return window.hermesDesktop.api<{ key: string; value: string }>({
    ...capabilityScoped(profile),
    path: '/api/env/reveal',
    method: 'POST',
    body: { key }
  })
}

export function validateProviderCredential(
  key: string,
  value: string,
  apiKey?: string
): Promise<{ ok: boolean; reachable: boolean; message: string; models?: string[] }> {
  return hermesApi<{ ok: boolean; reachable: boolean; message: string; models?: string[] }>({
    ...profileScoped(),
    path: '/api/providers/validate',
    method: 'POST',
    body: { key, value, api_key: apiKey ?? '' }
  })
}

export function getCustomEndpoints(): Promise<CustomEndpointsResponse> {
  return hermesApi<CustomEndpointsResponse>({
    ...profileScoped(),
    path: '/api/providers/custom-endpoints'
  })
}

export function saveCustomEndpoint(endpoint: CustomEndpointUpdate): Promise<CustomEndpointsResponse> {
  return hermesApi<CustomEndpointsResponse>({
    ...profileScoped(),
    path: '/api/providers/custom-endpoints',
    method: 'POST',
    body: endpoint
  })
}

export function validateCustomEndpoint(endpoint: CustomEndpointUpdate): Promise<CustomEndpointValidationResponse> {
  return hermesApi<CustomEndpointValidationResponse>({
    path: '/api/providers/custom-endpoints/validate',
    method: 'POST',
    body: endpoint
  })
}

export function activateCustomEndpoint(id: string): Promise<{ ok: boolean; provider: string; model: string }> {
  return hermesApi<{ ok: boolean; provider: string; model: string }>({
    ...profileScoped(),
    path: `/api/providers/custom-endpoints/${encodeURIComponent(id)}/activate`,
    method: 'POST'
  })
}

export function deleteCustomEndpoint(id: string): Promise<CustomEndpointsResponse> {
  return hermesApi<CustomEndpointsResponse>({
    ...profileScoped(),
    path: `/api/providers/custom-endpoints/${encodeURIComponent(id)}`,
    method: 'DELETE'
  })
}

export function listOAuthProviders(profile?: null | string): Promise<OAuthProvidersResponse> {
  return hermesApi<OAuthProvidersResponse>({
    ...profileScoped(profile),
    path: '/api/providers/oauth'
  })
}

export function disconnectOAuthProvider(
  providerId: string,
  profile?: null | string
): Promise<{ ok: boolean; provider: string }> {
  return hermesApi<{ ok: boolean; provider: string }>({
    ...profileScoped(profile),
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}`,
    method: 'DELETE'
  })
}

export function startOAuthLogin(providerId: string, profile?: ProfileScope): Promise<OAuthStartResponse> {
  return window.hermesDesktop.api<OAuthStartResponse>({
    ...capabilityScoped(profile),
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/start`,
    method: 'POST',
    body: {}
  })
}

export function submitOAuthCode(
  providerId: string,
  sessionId: string,
  code: string,
  profile?: null | string
): Promise<OAuthSubmitResponse> {
  return hermesApi<OAuthSubmitResponse>({
    ...profileScoped(profile),
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/submit`,
    method: 'POST',
    body: { session_id: sessionId, code }
  })
}

export function pollOAuthSession(
  providerId: string,
  sessionId: string,
  profile?: ProfileScope
): Promise<OAuthPollResponse> {
  return window.hermesDesktop.api<OAuthPollResponse>({
    ...capabilityScoped(profile),
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/poll/${encodeURIComponent(sessionId)}`
  })
}

export function cancelOAuthSession(sessionId: string, profile?: null | string): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(profile),
    path: `/api/providers/oauth/sessions/${encodeURIComponent(sessionId)}`,
    method: 'DELETE'
  })
}
