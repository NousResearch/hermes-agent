import type { McpTestResult } from '@/hermes'

// ---------------------------------------------------------------------------
// Shared MCP probe cache for the MCP surfaces: a probe is a REAL
// connect/disconnect (stdio servers get spawned!), so no surface may re-probe
// what another just learned. The background health checker never probes; it
// reads the backend's cached runtime state (store/mcp-health.ts).
// ---------------------------------------------------------------------------

export const NEEDS_AUTH_RE = /\b(401|unauthorized|forbidden|invalid[_ ]?token|authentication|oauth)\b/i

// Probe results outlive any component: each probe is a real connect/disconnect,
// so re-entering the MCP page (or a background sweep) must not re-probe the
// fleet. Manual refresh / auth / toggle-on bypass the cache.
export const PROBE_TTL_MS = 5 * 60_000

export const probeCache = new Map<string, { at: number; result: McpTestResult }>()

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
