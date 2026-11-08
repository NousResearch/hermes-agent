import { capabilityScoped, type ProfileScope } from './client'

/** `delegation.backend` values the Settings editors know how to edit (any other plugin backend is a plain string). */
export type DelegationMode = 'default' | 'model_map' | 'profile_routes'

/** One backend row from `GET /api/delegation/routing`: `default` plus every plugin-registered backend. */
export interface DelegationBackendInfo {
  name: string
  label: string
  description: string
  available: boolean
  reason: string
  builtin: boolean
}

/** One entry of `delegation.backends.model_map.options`. Unknown keys round-trip untouched. */
export interface DelegationModelRoute {
  name: string
  description: string
  provider?: string
  model: string
  base_url?: string
  reasoning_effort?: string
  service_tier?: string
  /** Display-only preview when set; sending it back keeps the stored key. */
  api_key?: string
  fallback_providers?: unknown[]
  [key: string]: unknown
}

export interface DelegationProfileRoute {
  name: string
  description?: string
}

export interface DelegationModelMap {
  options: DelegationModelRoute[]
  default: null | string
}

export interface DelegationProfileRoutes {
  profiles: DelegationProfileRoute[]
  default: null | string
}

/** Shape of `GET /api/delegation/routing`. */
export interface DelegationRouting {
  /** The saved `delegation.backend` (normalized; `default` when unset). */
  configured: string
  /** Why the saved backend cannot run here (plugin missing, off or unavailable); `null` when it runs.
   *  Core fails closed: while this is set every subagent is refused, it never quietly runs Default. */
  blocked_reason: null | string
  backends: DelegationBackendInfo[]
  model_map: DelegationModelMap
  profile_routes: DelegationProfileRoutes
  /** Profiles of this backend host a profile route may name. */
  profiles: string[]
}

export function getDelegationRouting(scope?: ProfileScope): Promise<DelegationRouting> {
  return window.hermesDesktop.api<DelegationRouting>({ ...capabilityScoped(scope), path: '/api/delegation/routing' })
}

export function selectDelegationBackend(
  backend: string,
  scope?: ProfileScope
): Promise<{ ok: boolean; backend: string }> {
  return window.hermesDesktop.api<{ ok: boolean; backend: string }>({
    ...capabilityScoped(scope),
    path: '/api/delegation/backend',
    method: 'PUT',
    body: { backend }
  })
}

export function saveDelegationModelRoutes(
  routes: DelegationModelMap,
  scope?: ProfileScope
): Promise<{ ok: boolean; model_map: DelegationModelMap }> {
  return window.hermesDesktop.api<{ ok: boolean; model_map: DelegationModelMap }>({
    ...capabilityScoped(scope),
    path: '/api/delegation/backends/model_map',
    method: 'PUT',
    body: routes
  })
}

export function saveDelegationProfileRoutes(
  routes: DelegationProfileRoutes,
  scope?: ProfileScope
): Promise<{ ok: boolean; profile_routes: DelegationProfileRoutes }> {
  return window.hermesDesktop.api<{ ok: boolean; profile_routes: DelegationProfileRoutes }>({
    ...capabilityScoped(scope),
    path: '/api/delegation/backends/profile_routes',
    method: 'PUT',
    body: routes
  })
}
