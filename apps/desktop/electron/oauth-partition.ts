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
 *   - Login flows may name the connection explicitly (`connectionId`): the
 *     registry editor can sign a draft in BEFORE it is persisted, and the
 *     login window must write to the jar the saved entry will read from.
 *     Identity then decides — a known entry follows the rules above; an
 *     unknown id is a pending non-primary oauth remote (the editor's save
 *     path never promotes a fresh entry to primary) and gets its own
 *     partition up front.
 *
 * Kept free of `electron` imports so it unit-tests in the electron vitest
 * project; main.ts owns session.fromPartition() and injects nothing here.
 */

export const LEGACY_OAUTH_PARTITION = 'persist:hermes-remote-oauth'

// Colon-free ON PURPOSE: Electron escapes ':' in a partition name to '%3A' in
// the on-disk profile folder, and a Windows profile folder whose name contains
// '%3A' gets a cookie store the network stack can neither read nor write (a
// jar seeded into it reads back zero cookies, `cookies.set()` never reaches
// disk, and every cookie-authenticated request 401s as `no_cookie`). A jar
// the app cannot see means the dial always looks signed-out and the user is
// asked to sign in again on every connect. The whole path component (prefix
// AND the sanitized id below) must stay within the characters Electron never
// percent-escapes — the invariant test pins "no ':' and no '%'".
//
// The name used to be `${LEGACY_OAUTH_PARTITION}:conn:<id>` (#92183). That
// jar never worked on Windows; on macOS/Linux a non-primary remote signed in
// under the old name is re-prompted ONCE after this change (the old
// `Partitions/hermes-remote-oauth%3Aconn%3A<id>` folder is left on disk,
// inert). One name on every platform beats a per-OS partition scheme.
const CONNECTION_PARTITION_PREFIX = `${LEGACY_OAUTH_PARTITION}-conn-`

export interface PartitionRegistrySnapshot {
  primary?: unknown
  connections?: unknown
}

export interface ResolveOauthPartitionOptions {
  registry?: PartitionRegistrySnapshot | null
  /** v1 single-connection remote URL (connection.json `remote.url`), when set. */
  v1RemoteUrl?: unknown
  /**
   * Connection id for login flows that run before the draft is persisted.
   * When set, identity decides the jar (module header); URL matching below
   * is skipped entirely.
   */
  connectionId?: unknown
  /**
   * Draft entry shape for the pre-save login: the `kind`/`authMode` the save
   * will persist. The unknown-id shortcut below grants a private jar only
   * when the saved entry would earn one (a non-primary cookie-auth remote);
   * cloud and token drafts keep the legacy shared jar, which is what they
   * resolve to after the save. Absent/invalid values fail closed to the
   * legacy jar — never a private one a saved entry would not read.
   */
  pendingAuthMode?: unknown
  pendingKind?: unknown
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

/** Read one property off an untrusted registry entry (parsed JSON) without a cast. */
function entryField(entry: unknown, key: string): unknown {
  if (!entry || typeof entry !== 'object' || !(key in entry)) {
    return undefined
  }

  return Object.getOwnPropertyDescriptor(entry, key)?.value
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

  // Identity-keyed resolution for pre-save logins. A pending draft has no
  // persisted entry to URL-match against; without this branch its sign-in
  // would fall through to the legacy shared jar — where the session is both
  // unreadable by the saved connection AND able to evict a same-host primary's
  // cookie (the #92183 leak the per-connection jars exist to stop).
  const wantedId = typeof opts.connectionId === 'string' ? opts.connectionId.trim() : ''

  if (wantedId) {
    const own = (registry.connections as unknown[]).find(c => entryField(c, 'id') === wantedId)

    if (!own) {
      // No persisted entry yet: gate the up-front private jar on the SAME
      // eligibility the URL-match path applies after the save, so the jar the
      // login writes is always the jar the saved entry reads. Only a
      // non-primary cookie-auth remote earns its own partition (module
      // header): cloud drafts must share the portal jar (the silent per-agent
      // cloud cascade depends on it) and token drafts never ride cookies, so
      // both sign in on the legacy jar — which is also what they resolve to
      // once saved. A draft whose URL IS the v1 remote's lands on legacy for
      // the same reason a saved v1-migrated entry does. An unknown id cannot
      // be the registry primary (the primary is, by definition, persisted), so
      // no primary check is needed here. Missing/invalid pending shape (an
      // older renderer) fails closed to the legacy jar, never a private one.
      const pendingKind = typeof opts.pendingKind === 'string' ? opts.pendingKind : ''
      const pendingAuthMode = typeof opts.pendingAuthMode === 'string' ? opts.pendingAuthMode : ''

      return pendingKind === 'remote' && pendingAuthMode === 'oauth' && !(v1Norm && requestNorm === v1Norm)
        ? `${CONNECTION_PARTITION_PREFIX}${sanitizePartitionComponent(wantedId)}`
        : LEGACY_OAUTH_PARTITION
    }

    const ownBaseNorm = normalizeForMatch(entryField(own, 'url'))

    if (
      wantedId !== primaryId &&
      entryField(own, 'kind') === 'remote' &&
      entryField(own, 'authMode') === 'oauth' &&
      ownBaseNorm &&
      !(v1Norm && ownBaseNorm === v1Norm)
    ) {
      return `${CONNECTION_PARTITION_PREFIX}${sanitizePartitionComponent(wantedId)}`
    }

    return LEGACY_OAUTH_PARTITION
  }

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
    if (!best || baseNorm.length > best.baseNorm.length || (baseNorm.length === best.baseNorm.length && id < best.id)) {
      best = { baseNorm, id }
    }
  }

  if (!best) {
    return LEGACY_OAUTH_PARTITION
  }

  return `${CONNECTION_PARTITION_PREFIX}${sanitizePartitionComponent(best.id)}`
}
