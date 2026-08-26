/**
 * oauth-partition.ts
 *
 * Per-connection cookie-jar isolation for cookie-authenticated (OAuth /
 * dashboard basic-auth) remote gateways (#92183).
 *
 * Historically every cookie-mode remote rode ONE Electron session partition
 * (`persist:hermes-remote-oauth`) — the jar was keyed on the auth *mode*, not
 * on the connection's identity. Chromium cookie jars scope by host and ignore
 * the port, so two registered gateways on the same host (the #92183 VPN
 * setup: one box, two dashboards) fought over the same `hermes_session*`
 * cookies: signing in to gateway B evicted gateway A's session, and A's
 * cookie was silently PRESENTED to B on every request — a cross-connection
 * credential leak.
 *
 * This module is the pure decision seam: given a request/base URL and a
 * snapshot of the v2 connections registry, decide which session partition the
 * request must ride. Rules:
 *
 *   - A NON-primary v2 registry `remote` entry with cookie auth
 *     (`authMode: 'oauth'`, which covers dashboard basic/password providers —
 *     they authenticate via session cookies too) gets its own partition
 *     derived from the connection id. Fail closed: its requests can never see
 *     another connection's cookies, and its login window can never evict them.
 *   - The registry PRIMARY and the v1 single-connection remote stay on the
 *     LEGACY shared partition, so existing signed-in users are not signed out
 *     by the upgrade.
 *   - `cloud` entries stay on the legacy partition: the silent per-agent
 *     cascade deliberately shares one jar with the Nous Portal session.
 *   - Token-auth remotes, portal URLs, and anything unmatched or malformed
 *     fall back to the legacy partition (cookie-free flows are unaffected).
 *
 * Kept free of `electron` imports so it unit-tests in the electron vitest
 * project; main.ts owns session.fromPartition() and injects nothing here.
 */

export const LEGACY_OAUTH_PARTITION = 'persist:hermes-remote-oauth'

const CONNECTION_PARTITION_PREFIX = `${LEGACY_OAUTH_PARTITION}:conn:`

export interface PartitionRegistrySnapshot {
  primary?: unknown
  connections?: unknown
}

export interface ResolveOauthPartitionOptions {
  registry?: PartitionRegistrySnapshot | null
  /** v1 single-connection remote URL (connection.json `remote.url`), when set. */
  v1RemoteUrl?: unknown
}

/**
 * Normalize a URL for base-url matching: lowercased scheme+host, explicit
 * default ports elided (URL does this), trailing slashes trimmed, query and
 * fragment dropped. Returns null for anything that is not a plain http(s) URL.
 */
function normalizeForMatch(raw: unknown): string | null {
  if (typeof raw !== 'string' || !raw.trim()) {
    return null
  }

  try {
    const parsed = new URL(raw.trim())

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null
    }

    const path = parsed.pathname.replace(/\/+$/, '')

    return `${parsed.protocol}//${parsed.host}${path}`
  } catch {
    return null
  }
}

/**
 * True when `requestNorm` is the entry base URL itself or a path underneath
 * it. Both sides are pre-normalized; the '/' requirement is what stops
 * `https://gw.example.com.evil.tld` from matching `https://gw.example.com`.
 */
function matchesBase(requestNorm: string, baseNorm: string): boolean {
  return requestNorm === baseNorm || requestNorm.startsWith(`${baseNorm}/`)
}

/** Partition names must stay printable/simple; connection ids are user data. */
function sanitizePartitionComponent(id: string): string {
  return encodeURIComponent(id).replace(/%/g, '_')
}

/**
 * Decide the Electron session partition a cookie-authenticated request against
 * `requestUrl` must ride. See the module header for the rules.
 */
export function resolveOauthPartition(requestUrl: unknown, opts: ResolveOauthPartitionOptions = {}): string {
  const requestNorm = normalizeForMatch(requestUrl)

  if (!requestNorm) {
    return LEGACY_OAUTH_PARTITION
  }

  const registry = opts.registry

  if (!registry || typeof registry !== 'object' || !Array.isArray(registry.connections)) {
    return LEGACY_OAUTH_PARTITION
  }

  const primaryId = typeof registry.primary === 'string' ? registry.primary : ''
  const v1Norm = normalizeForMatch(opts.v1RemoteUrl)

  let best: { baseNorm: string; id: string } | null = null

  for (const entry of registry.connections) {
    if (!entry || typeof entry !== 'object') {
      continue
    }

    const id = typeof (entry as any).id === 'string' ? (entry as any).id.trim() : ''

    // Only NON-primary v2 `remote` entries with cookie-flow auth get their own
    // jar. Cloud entries need the shared portal jar; token entries never use
    // cookies; the primary (and the v1 remote it migrated from) keeps the
    // legacy jar so an upgrade does not sign the user out.
    if (!id || id === primaryId) {
      continue
    }

    if ((entry as any).kind !== 'remote' || (entry as any).authMode !== 'oauth') {
      continue
    }

    const baseNorm = normalizeForMatch((entry as any).url)

    if (!baseNorm || (v1Norm && baseNorm === v1Norm)) {
      continue
    }

    if (!matchesBase(requestNorm, baseNorm)) {
      continue
    }

    // Longest base-url prefix wins (sub-path gateways behind one proxy);
    // identical URLs tie-break on the lexicographically smallest id so the
    // choice is deterministic across processes and launches.
    if (
      !best ||
      baseNorm.length > best.baseNorm.length ||
      (baseNorm.length === best.baseNorm.length && id < best.id)
    ) {
      best = { baseNorm, id }
    }
  }

  if (!best) {
    return LEGACY_OAUTH_PARTITION
  }

  return `${CONNECTION_PARTITION_PREFIX}${sanitizePartitionComponent(best.id)}`
}
