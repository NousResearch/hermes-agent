import { getApiRequestProfile, type ProfileScope } from '@/api/client'
import type { McpTestResult } from '@/hermes'

import { type ConnectionScopeDescriptor, connectionScopeSuffix } from './connection-scoped'

// ---------------------------------------------------------------------------
// Shared MCP probe cache. Extracted from mcp-tab.tsx so the MCP page and the
// background health checker (store/mcp-health.ts) share ONE cache: a probe is
// a REAL connect/disconnect (stdio servers get spawned!), so neither surface
// may re-probe what the other just learned.
// ---------------------------------------------------------------------------

export const NEEDS_AUTH_RE = /\b(401|unauthorized|forbidden|invalid[_ ]?token|authentication|oauth)\b/i

// Probe results outlive any component: each probe is a real connect/disconnect,
// so re-entering the MCP page (or a background sweep) must not re-probe the
// fleet. Manual refresh / auth / toggle-on bypass the cache.
export const PROBE_TTL_MS = 5 * 60_000

export const probeCache = new Map<string, { at: number; result: McpTestResult }>()

// The unscoped primary backend is not the same cache row as a literal
// `profile=default` request. Keep that distinction even when the UI's active
// profile happens to be named differently during startup.
const PRIMARY_SLOT_KEY = '@primary-slot'

export interface McpOwnerScope {
  /** True when the request can carry a registered connection id explicitly. */
  exact: boolean
  /** Stable cache/status/snooze identity for (connection, profile). */
  key: string
  /** Request scope matching capabilityScoped() semantics. */
  request: { connectionId?: string; priority?: 'foreground'; profile?: string }
}

/** Resolve the one owner identity shared by the MCP page and health sweep.
 *
 * An explicit object scope mirrors capabilityScoped(): its connection field is
 * authoritative, including an explicit null/empty value, so it never silently
 * inherits the ambient registry connection. Legacy string/undefined scopes use
 * the actual ambient registry id when available, otherwise the descriptor's
 * remote base (or the local legacy bucket) as a cache-only identity. The latter
 * remains profile-only on the wire and must be fenced by the caller before a
 * stale action is issued. */
export function resolveMcpOwner(
  scope: ProfileScope | undefined,
  activeProfile: string,
  ambientConnectionId: null | string,
  connection: ConnectionScopeDescriptor | null | undefined
): McpOwnerScope | null {
  void activeProfile

  const explicitScope = scope && typeof scope === 'object' ? scope : null
  const ambientProfile = (getApiRequestProfile() ?? '').trim()
  let profile = ''
  let requestProfile: string | undefined

  if (explicitScope) {
    profile = String(explicitScope.profile ?? '').trim()
    requestProfile = profile || undefined
  } else if (typeof scope === 'string') {
    profile = scope.trim()
    requestProfile = profile || undefined
  } else if (scope === undefined) {
    // The API client's ambient profile is the wire source of truth. When it is
    // absent, the request targets the primary slot; do not alias that row to a
    // UI profile name or to the literal `default` profile.
    profile = ambientProfile
    requestProfile = ambientProfile || undefined
  }

  const profileKey = profile || PRIMARY_SLOT_KEY
  const requestedConnection = explicitScope ? String(explicitScope.connectionId ?? '').trim() : ''
  const connectionId = explicitScope ? requestedConnection : (ambientConnectionId ?? '').trim()

  // An explicit object without a connection id deliberately omits the ambient
  // registry tag in capabilityScoped(). Its request may therefore land on the
  // registry primary rather than the descriptor currently shown in the store;
  // do not mislabel that unknown owner or share its cache row.
  if (explicitScope && !connectionId) {
    return null
  }

  if (connectionId) {
    const request = {
      connectionId,
      ...(requestProfile ? { profile: requestProfile } : {}),
      ...(explicitScope || typeof scope === 'string' ? { priority: 'foreground' as const } : {})
    }

    return { exact: true, key: `${connectionId}::${profileKey}`, request }
  }

  if (!connection) {
    return null
  }

  const connectionKey = connectionScopeSuffix(connection, false) || 'local'

  return {
    exact: false,
    key: `${connectionKey}::${profileKey}`,
    request: {
      ...(requestProfile ? { profile: requestProfile } : {}),
      ...(typeof scope === 'string' ? { priority: 'foreground' as const } : {})
    }
  }
}

// A probe is only valid for one (profile, exact-config) pair. Keying the cache
// by a fingerprint of the connection-relevant fields — plus the active profile
// — means a same-name edit (url/command/env change) or a same-named server in
// another profile MISSES the cache instead of showing a stale probe.
export const serverFingerprint = (server: Record<string, unknown>): string =>
  JSON.stringify([server.url, server.command, server.args, server.env, server.headers, server.transport, server.auth])

export const probeKey = (name: string, server: Record<string, unknown> | undefined, profileKey: string): string =>
  `${profileKey}::${name}::${serverFingerprint(server ?? {})}`

/** Read a still-fresh cached probe result, or null (miss / expired). */
export function freshProbe(key: string, now = Date.now()): McpTestResult | null {
  const cached = probeCache.get(key)

  return cached && now - cached.at < PROBE_TTL_MS ? cached.result : null
}

/** Classify a finished probe the way the MCP page's status dot does. */
export function classifyProbe(result: McpTestResult): 'error' | 'needs-auth' | 'ok' {
  if (result.ok) {
    return 'ok'
  }

  return NEEDS_AUTH_RE.test(result.error ?? '') ? 'needs-auth' : 'error'
}
