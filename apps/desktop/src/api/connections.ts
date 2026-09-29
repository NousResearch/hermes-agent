import { capabilityScoped, type ProfileScope } from './client'

/** `connected` / `missing`, or `unknown` when local state cannot tell (no badge is shown then). */
export type ConnectionState = 'connected' | 'missing' | 'unknown'

/** One cache entry for every screen that shows connection state, so one refresh updates them all. */
export const CONNECTION_STATUS_KEY = ['connections-status']

export type ConnectionStatusMap = Record<string, ConnectionState>

export const getConnectionStatus = (profile?: ProfileScope) =>
  window.hermesDesktop.api<ConnectionStatusMap>({ ...capabilityScoped(profile), path: '/api/connections/status' })
