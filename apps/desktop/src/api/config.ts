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

type ConfigReadOrigin = { connectionId?: null | string; priority?: 'foreground'; profile?: string }

const configReadOrigins = new WeakMap<object, ConfigReadOrigin>()
// Every origin object ever bound, so resolveConfigWriteScope can tell a
// captured read origin handed back as `writeScope` apart from a fresh
// `{ connectionId, profile }` pin from the scope selector.
const knownConfigReadOrigins = new WeakSet<object>()

/** Snapshot the `(connectionId, profile)` that served a config GET. */
export function bindConfigReadOrigin(record: object, origin: ConfigReadOrigin): void {
  configReadOrigins.set(record, origin)
  knownConfigReadOrigins.add(origin)
}

export function peekConfigReadOrigin(record: object | undefined | null): ConfigReadOrigin | undefined {
  return record ? configReadOrigins.get(record) : undefined
}

/** Carry `source`'s read origin onto a record derived from it, so the next write still routes to the gateway that served the GET. */
export function retainConfigReadOrigin<T extends object>(next: T, source: object | null | undefined): T {
  const origin = peekConfigReadOrigin(source)

  if (origin) {
    bindConfigReadOrigin(next, origin)
  }

  return next
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
): { connectionId?: null | string; priority?: 'foreground'; profile?: string } {
  if (requestScope && typeof requestScope === 'object') {
    // A captured read origin (the hook's `writeScope`) is already a
    // capabilityScoped() result. Spread it exactly like the WeakMap branch
    // below instead of re-running capabilityScoped, which would stamp
    // `priority: 'foreground'` onto an AMBIENT origin that never carried it —
    // the two branches must yield the same tag for the same read.
    return knownConfigReadOrigins.has(requestScope)
      ? { ...(requestScope as ConfigReadOrigin) }
      : capabilityScoped(requestScope)
  }

  const captured = peekConfigReadOrigin(record)

  if (captured) {
    const profile = typeof requestScope === 'string' ? requestScope.trim() : ''

    // Spread, don't rebuild: the captured origin is a capabilityScoped()
    // result and may carry `priority: 'foreground'`; an explicit profile
    // string gets the same foreground priority profileScoped(string) grants.
    return {
      ...captured,
      ...(profile ? { profile, priority: 'foreground' as const } : {})
    }
  }

  return capabilityScoped(requestScope)
}

export function getStatus(profile?: ProfileScope): Promise<StatusResponse> {
  return hermesApi<StatusResponse>({
    ...capabilityScoped(profile),
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

/** GET a config record on the capability scope and bind the serving
 *  `(connectionId, profile)` to it so the matching write routes back there. */
async function fetchBoundConfigRecord(
  profile: ProfileScope,
  request: { path: string; timeoutMs?: number }
): Promise<HermesConfigRecord> {
  const origin = capabilityScoped(profile ?? undefined)

  const record = await window.hermesDesktop.api<HermesConfigRecord>({ ...origin, ...request })

  if (record && typeof record === 'object') {
    bindConfigReadOrigin(record, origin)
  }

  return record
}

export function getHermesConfigRecord(
  profile?: ProfileScope,
  { includeDefaults = true }: { includeDefaults?: boolean } = {}
): Promise<HermesConfigRecord> {
  return fetchBoundConfigRecord(profile, {
    path: includeDefaults ? '/api/config' : '/api/config?include_defaults=false'
  })
}

export function getHermesConfigDefaults(profile?: ProfileScope): Promise<HermesConfigRecord> {
  return fetchBoundConfigRecord(profile, {
    path: '/api/config/defaults',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function getHermesConfigSchema(profile?: ProfileScope): Promise<ConfigSchemaResponse> {
  return hermesApi<ConfigSchemaResponse>({
    ...capabilityScoped(profile),
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

export function getEnvVars(profile?: ProfileScope): Promise<Record<string, EnvVarInfo>> {
  return hermesApi<Record<string, EnvVarInfo>>({
    ...capabilityScoped(profile),
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
  apiKey?: string,
  profile?: ProfileScope
): Promise<{ ok: boolean; reachable: boolean; message: string; models?: string[]; resolved_base_url?: string }> {
  return hermesApi<{
    ok: boolean
    reachable: boolean
    message: string
    models?: string[]
    resolved_base_url?: string
  }>({
    ...capabilityScoped(profile),
    path: '/api/providers/validate',
    method: 'POST',
    body: { key, value, api_key: apiKey ?? '' }
  })
}

export function getCustomEndpoints(profile?: ProfileScope): Promise<CustomEndpointsResponse> {
  return hermesApi<CustomEndpointsResponse>({
    ...capabilityScoped(profile),
    path: '/api/providers/custom-endpoints'
  })
}

export function saveCustomEndpoint(endpoint: CustomEndpointUpdate, profile?: ProfileScope): Promise<CustomEndpointsResponse> {
  return hermesApi<CustomEndpointsResponse>({
    ...capabilityScoped(profile),
    path: '/api/providers/custom-endpoints',
    method: 'POST',
    body: endpoint
  })
}

export function validateCustomEndpoint(
  endpoint: CustomEndpointUpdate,
  profile?: ProfileScope
): Promise<CustomEndpointValidationResponse> {
  return hermesApi<CustomEndpointValidationResponse>({
    ...capabilityScoped(profile),
    path: '/api/providers/custom-endpoints/validate',
    method: 'POST',
    body: endpoint
  })
}

export function activateCustomEndpoint(
  id: string,
  profile?: ProfileScope
): Promise<{ ok: boolean; provider: string; model: string }> {
  return hermesApi<{ ok: boolean; provider: string; model: string }>({
    ...capabilityScoped(profile),
    path: `/api/providers/custom-endpoints/${encodeURIComponent(id)}/activate`,
    method: 'POST'
  })
}

export function deleteCustomEndpoint(id: string, profile?: ProfileScope): Promise<CustomEndpointsResponse> {
  return hermesApi<CustomEndpointsResponse>({
    ...capabilityScoped(profile),
    path: `/api/providers/custom-endpoints/${encodeURIComponent(id)}`,
    method: 'DELETE'
  })
}

export function listOAuthProviders(profile?: ProfileScope): Promise<OAuthProvidersResponse> {
  return hermesApi<OAuthProvidersResponse>({
    ...capabilityScoped(profile),
    path: '/api/providers/oauth'
  })
}

export function disconnectOAuthProvider(
  providerId: string,
  profile?: ProfileScope
): Promise<{ ok: boolean; provider: string }> {
  return hermesApi<{ ok: boolean; provider: string }>({
    ...capabilityScoped(profile),
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
  profile?: ProfileScope
): Promise<OAuthSubmitResponse> {
  return hermesApi<OAuthSubmitResponse>({
    ...capabilityScoped(profile),
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

export function cancelOAuthSession(sessionId: string, profile?: ProfileScope): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...capabilityScoped(profile),
    path: `/api/providers/oauth/sessions/${encodeURIComponent(sessionId)}`,
    method: 'DELETE'
  })
}
