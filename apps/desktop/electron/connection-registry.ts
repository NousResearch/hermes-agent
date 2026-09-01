/**
 * connection-registry.ts
 *
 * Pure, electron-free helpers for the desktop's multi-connection registry —
 * the v2 successor to the single global `mode` + `remote` block in
 * connection.json. The registry is a named list of agent SOURCES (local
 * runtime, remote gateways, Hermes Cloud instances, SSH hosts) that are all
 * registered at once; routing/pooling changes that consume the registry land
 * separately, so this module is deliberately storage-shaped, not
 * transport-shaped.
 *
 * Design rules (agreed with Teknium, Aug 2026):
 *  - Every connection carries a REQUIRED, registry-unique `label` (the
 *    "device name"). Uniqueness is case-insensitive so `Homelab` and
 *    `homelab` can't coexist and produce two identical badges.
 *  - When two sources expose the same profile name, surfaces disambiguate as
 *    `@<profile>-<label-slug>` — `agentHandle()` is the one place that rule
 *    lives.
 *  - The registry ALWAYS contains exactly one `local` connection (the app's
 *    own runtime). It cannot be removed; it is the default primary.
 *  - `primary` designates the connection that owns the window backend (boot
 *    overlay, install/update machinery). Removing the primary retargets to
 *    the local entry rather than leaving a dangling id.
 *
 * Kept standalone (no `import 'electron'`) so it unit-tests with `node --test`
 * — same pattern as connection-config.ts / backend-probes.ts. main.ts wires
 * these into the IPC layer and owns file I/O + secret encryption.
 */

import {
  hostLabelFromBaseUrl,
  modeIsRemoteLike,
  normalizeRemoteBaseUrl,
  normalizeRemoteHeaders,
  normalizeSshConfig,
  normAuthMode
} from './connection-config'
import { matchingConnectionId, type StoredRoute } from './connection-route-identity'

export const REGISTRY_VERSION = 2

export const LOCAL_CONNECTION_ID = 'local'

/** Connection kinds. 'cloud' is remote-shaped (see modeIsRemoteLike) but keeps
 * its provenance so the UI can render the right card and updates can skip
 * platform-managed instances. */
export type ConnectionKind = 'cloud' | 'local' | 'remote' | 'ssh'

export interface RegistryConnection {
  id: string
  kind: ConnectionKind
  /** Required, unique (case-insensitive) display name — the "device name". */
  label: string
  /** remote/cloud: normalized base URL. */
  url?: string
  /** remote/cloud: 'token' | 'oauth'. */
  authMode?: 'oauth' | 'token'
  /** remote: encrypted token envelope (opaque here; main.ts encrypts/decrypts). */
  token?: unknown
  /** remote/cloud: extra gateway headers (Cloudflare Access etc.). Secret
   * envelopes, same shape as `token`; names pre-filtered through
   * normalizeRemoteHeaders. Optional and additive — v2 registries written
   * before this field keep loading unchanged. */
  headers?: Record<string, unknown>
  /** cloud: portal org slug/id the instance was discovered under. */
  org?: string
  /** ssh fields (normalizeSshConfig shapes). */
  host?: string
  user?: string
  port?: number
  keyPath?: string
  remoteHermesPath?: string
  remoteProfile?: string
}

/**
 * A registry entry that failed normalization (#94246). The raw entry is USER
 * DATA — it is preserved verbatim here (and re-persisted on every write)
 * instead of being silently dropped, so a malformed/corrupt entry never
 * requires "delete connections.json" recovery and never loses the user's
 * connection material.
 */
export interface QuarantinedRegistryEntry {
  reason: string
  entry: unknown
}

/** Upper bound on preserved quarantine entries so a pathological file cannot
 * grow the registry without limit. Oldest-first within one load pass. */
export const REGISTRY_QUARANTINE_CAP = 20

export interface ConnectionRegistry {
  version: typeof REGISTRY_VERSION
  /** id of the connection that owns the window/primary backend. */
  primary: string
  /** Which saved source Sessions should restore when the app launches. */
  launchMode: 'last-used' | 'primary'
  /** Last source the Sessions workspace successfully opened. Additive in v2
   * so registries written before multi-source switching still normalize. */
  lastUsed: string
  connections: RegistryConnection[]
  /** Entries preserved from a malformed load — absent when empty. */
  quarantined?: QuarantinedRegistryEntry[]
}

// ── Labels and ids ──────────────────────────────────────────────────────────

const LABEL_MAX = 64

/** Canonical comparison key for label uniqueness. */
export function labelKey(label: string): string {
  return String(label || '')
    .trim()
    .toLowerCase()
}

/**
 * Derive a registry-unique label from a candidate: clamps to LABEL_MAX (a
 * migrated URL host can exceed it, which would fail validation on any later
 * edit) and suffixes " 2" / " 3" / … on collision. The single home of the
 * label-dedup rule — normalizeRegistry and the migration both use it.
 */
export function uniqueLabel(candidate: string, taken: Iterable<string>): string {
  const used = new Set([...taken].map(labelKey))

  // Reserve room for a collision suffix so the suffixed form stays in-bounds.
  const base = String(candidate || '')
    .trim()
    .slice(0, LABEL_MAX - 4)

  if (!used.has(labelKey(base))) {
    return base
  }

  for (let n = 2; ; n += 1) {
    const suffixed = `${base} ${n}`

    if (!used.has(labelKey(suffixed))) {
      return suffixed
    }
  }
}

/** Kebab-slug of a label for ids and @handles. Never empty for a non-empty label. */
export function labelSlug(label: string): string {
  const slug = String(label || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)

  return slug || 'connection'
}

/**
 * The one place the duplicate-agent naming rule lives: a profile that exists
 * on several registered sources renders as `@<profile>-<label-slug>`;
 * a profile unique across the roster keeps its bare name.
 */
export function agentHandle(profile: string, connectionLabel: string, duplicated: boolean): string {
  const name = String(profile || '').trim() || 'default'

  return duplicated ? `${name}-${labelSlug(connectionLabel)}` : name
}

/**
 * Pool key for a backend serving (connection, profile). The local/primary
 * connection keeps the BARE profile key so every legacy pool entry, reaper
 * log line, and touch call stays byte-identical for single-source users;
 * non-local connections get an unambiguous composite (`conn:<id>::<profile>`)
 * that cannot collide with a plain profile name (colons are invalid in
 * profile names).
 *
 * NOTE: the renderer's socket registry uses the twin implementation in
 * apps/shared/src/backend-scope.ts (`@hermes/shared`) — tsconfig project
 * boundaries prevent a single physical module here. The two are pinned
 * byte-identical by the cross-copy contract test in
 * connection-registry.test.ts; change BOTH or that test fails.
 */
export function backendScopeKey(connectionId: null | string | undefined, profile: null | string | undefined): string {
  const profileKey = String(profile ?? '').trim() || 'default'
  const connection = String(connectionId ?? '').trim()

  if (!connection || connection === LOCAL_CONNECTION_ID) {
    return profileKey
  }

  return `conn:${connection}::${profileKey}`
}

/**
 * Inverse of backendScopeKey(): recover (connectionId, profile) from a pool
 * key. A bare profile key (the local/primary scope) maps to a null
 * connectionId. Used by the post-resume rebuild path (#93910) to re-dial a
 * retired pool entry through the same claim-guarded ensure path a renderer
 * would use.
 */
export function parseBackendScopeKey(key: string): { connectionId: null | string; profile: string } {
  const value = String(key ?? '').trim()
  const match = /^conn:(.+?)::(.+)$/.exec(value)

  if (!match) {
    return { connectionId: null, profile: value || 'default' }
  }

  return { connectionId: match[1], profile: match[2] }
}

/** All pool keys owned by a connection share this prefix (used to stop them on remove). */
export function backendScopePrefix(connectionId: string): string {
  return `conn:${String(connectionId).trim()}::`
}

export interface RegistryLocalRoute {
  /** Reuse the legacy v1 ensureBackend path — it already resolves to the
   * app's own local runtime, so single-source behavior stays byte-identical. */
  delegate: boolean
  /** Pool key for the forced-local child when not delegating. */
  poolKey: string
}

export interface ResolvedConnectionSshDescriptor {
  effectiveConfigFingerprint?: string
  host?: string
  keyPath?: string
  port?: number
  remoteHermesPath?: string
  remoteProfile?: string
  user?: string
}

export interface ResolvedConnectionDescriptor {
  authMode?: unknown
  baseUrl?: string
  /** Property presence means this descriptor claims registry qualification.
   * Invalid or retired claims fail closed; only descriptors with no such
   * property may enter the legacy compatibility resolver. */
  connectionId?: unknown
  headers?: Record<string, unknown>
  mode?: 'local' | 'remote'
  org?: unknown
  remoteHost?: string
  remoteKind?: 'cloud' | 'ssh' | 'url'
  ssh?: ResolvedConnectionSshDescriptor
  token?: unknown
}

/**
 * Recover registry identity for a descriptor resolved through the legacy v1
 * profile path. Registry-scoped routes already carry `connectionId`; that
 * exact identity is authoritative only while it names a current registry
 * entry. Only genuinely unqualified descriptors may use compatibility
 * inference, which keeps migrated per-profile remotes truthful until v1 is
 * retired without letting malformed qualification fall through to a weaker
 * endpoint-shaped identity.
 */
export function resolvedConnectionId(
  registry: ConnectionRegistry,
  descriptor: ResolvedConnectionDescriptor
): null | string {
  if (Object.prototype.hasOwnProperty.call(descriptor, 'connectionId')) {
    const explicitConnectionId = descriptor.connectionId

    // Presence is authoritative even when the value is unusable. Never turn
    // a malformed, blank, unknown, or retired registry claim into permission
    // to infer a different source from mutable endpoint metadata.
    if (typeof explicitConnectionId !== 'string' || !explicitConnectionId.trim()) {
      return null
    }

    return registry.connections.some(connection => connection.id === explicitConnectionId) ? explicitConnectionId : null
  }

  if (descriptor.mode === 'local') {
    const localConnections = registry.connections.filter(connection => connection.kind === 'local')

    return localConnections.length === 1 ? localConnections[0].id : null
  }

  if (descriptor.mode !== 'remote') {
    return null
  }

  if (descriptor.remoteKind === 'ssh') {
    if (Object.prototype.hasOwnProperty.call(descriptor, 'ssh')) {
      if (!descriptor.ssh || typeof descriptor.ssh !== 'object') {
        return null
      }

      return matchingConnectionId(registry, { ...descriptor.ssh, kind: 'ssh' }, 'unique') ?? null
    }

    // Old descriptors expose only user@host after the tunnel has discarded
    // port/key/path/profile. That weak shape is compatible only when exactly
    // one registered SSH source even shares the target, and the defaulted
    // route still satisfies the canonical #88922 full-envelope matcher.
    const ssh = normalizeSshConfig({ mode: 'ssh', host: descriptor.remoteHost })

    if (!ssh) {
      return null
    }

    const target = normalizedSshTarget(ssh)

    const coarseMatches = registry.connections.filter(
      connection => connection.kind === 'ssh' && normalizedSshTarget(connection) === target
    )

    if (!target || coarseMatches.length !== 1) {
      return null
    }

    return matchingConnectionId(registry, { kind: 'ssh', ...ssh }, 'unique') ?? null
  }

  const kind = descriptor.remoteKind === 'cloud' ? 'cloud' : descriptor.remoteKind === 'url' ? 'remote' : null

  if (!kind) {
    return null
  }

  let url = ''

  try {
    url = normalizeRemoteBaseUrl(descriptor.baseUrl)
  } catch {
    return null
  }

  const authMode = normAuthMode(descriptor.authMode)

  const route: StoredRoute = {
    authMode,
    headers: descriptor.headers,
    kind,
    org: descriptor.org,
    token: descriptor.token,
    url
  }

  const hasExactEnvelope =
    Object.prototype.hasOwnProperty.call(descriptor, 'authMode') &&
    Object.prototype.hasOwnProperty.call(descriptor, 'headers') &&
    (authMode === 'oauth' || Object.prototype.hasOwnProperty.call(descriptor, 'token')) &&
    (kind === 'remote' || Object.prototype.hasOwnProperty.call(descriptor, 'org'))

  if (!hasExactEnvelope) {
    // A URL alone cannot choose among legal registrations that differ by auth,
    // headers, Cloud organization, or account. Require one coarse candidate
    // before the same full-envelope matcher is allowed to accept the legacy
    // defaults; otherwise zero/multiple candidates fail closed.
    const coarseMatches = registry.connections.filter(connection => {
      if (connection.kind !== kind) {
        return false
      }

      try {
        return normalizeRemoteBaseUrl(connection.url) === url
      } catch {
        return false
      }
    })

    if (coarseMatches.length !== 1) {
      return null
    }
  }

  return matchingConnectionId(registry, route, 'unique') ?? null
}

export interface ReuseMatchingPrimarySshBackendOptions {
  connectionId: null | string | undefined
  effectiveFingerprint: (source: RegistryConnection) => Promise<string>
  ensurePrimary: () => Promise<ResolvedConnectionDescriptor>
  profile: null | string | undefined
  registry: ConnectionRegistry
  source: RegistryConnection
}

/**
 * Reuse the v1 window SSH backend only when its actual dialing identity matches
 * the registry primary. Resolving that descriptor may boot the primary; a
 * mismatch returns null without reusing it so the caller continues with its
 * separately scoped registry backend. A matching descriptor is returned
 * unchanged and the caller may re-stamp routing fields such as profile and
 * connectionId. Guards run before either async dependency so secondary
 * profiles and sources never bootstrap the primary.
 */
export async function reuseMatchingPrimarySshBackend({
  connectionId,
  effectiveFingerprint,
  ensurePrimary,
  profile,
  registry,
  source
}: ReuseMatchingPrimarySshBackendOptions): Promise<null | ResolvedConnectionDescriptor> {
  const id = String(connectionId ?? '').trim()
  const profileKey = String(profile ?? '').trim() || 'default'

  if (profileKey !== 'default' || !id || id !== registry.primary || source.id !== id || source.kind !== 'ssh') {
    return null
  }

  let sourceFingerprint

  try {
    sourceFingerprint = String(await effectiveFingerprint(source)).trim()
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : String(cause)

    throw new Error(
      `Could not resolve effective SSH config for connection "${source.label}" (${source.id}) via ssh -G: ${detail}`,
      { cause }
    )
  }

  const descriptor = await ensurePrimary()
  const activeSsh = descriptor.mode === 'remote' && descriptor.remoteKind === 'ssh' ? descriptor.ssh : null
  const rootProfile = (value: unknown) => String(value || '').trim() || 'default'

  if (
    !sourceFingerprint ||
    !activeSsh ||
    sourceFingerprint !== String(activeSsh.effectiveConfigFingerprint || '').trim() ||
    String(source.remoteHermesPath || '').trim() !== String(activeSsh.remoteHermesPath || '').trim() ||
    rootProfile(source.remoteProfile) !== rootProfile(activeSsh.remoteProfile)
  ) {
    return null
  }

  return descriptor
}

/**
 * Whether a registry-scoped request names the already-running primary backend.
 * Main uses this before opening a pooled registry backend so the registry's
 * primary SSH/remote source cannot spawn a second isolated server for the same
 * descriptor.
 */
export function registrySourceOwnsPrimaryBackend(
  registry: ConnectionRegistry,
  connectionId: null | string | undefined,
  descriptor: ResolvedConnectionDescriptor
): boolean {
  const id = String(connectionId ?? '').trim()

  return Boolean(id) && id === registry.primary && resolvedConnectionId(registry, descriptor) === id
}

function normalizedSshTarget(route: { host?: unknown; port?: unknown; user?: unknown }): null | string {
  const ssh = normalizeSshConfig({ ...route, mode: 'ssh' })

  if (!ssh) {
    return null
  }

  const host = String(ssh.host || '')
    .trim()
    .toLowerCase()

  const user = String(ssh.user || '')
    .trim()
    .toLowerCase()

  return user ? `${user}@${host}` : host
}

/**
 * How the registry's 'local' entry resolves a backend for `profile`.
 *
 * The 'local' entry means THIS machine's runtime — always. The legacy
 * ensureBackend() path instead follows the v1 connection.json routing table,
 * where a global remote mode (or a per-profile remote override) resolves to a
 * REMOTE descriptor. A migrated user whose v1 global mode was remote gets that
 * remote as the registry primary AND keeps the mandatory 'local' entry, so
 * delegating 'local' to the v1 route made the roster's "This device" rows
 * enumerate and dial the remote box: every profile appeared twice (forcing
 * -slug handles) and "local" agents talked to the remote.
 *
 * When the v1 route is already local we delegate (legacy path, byte-identical
 * pool keys). When v1 says remote, the local entry spawns its own genuinely
 * local child under a composite pool key: backendScopeKey('local', p) maps to
 * the BARE profile key by design, and that slot may already hold the v1
 * route's REMOTE descriptor — so the forced-local child pools under the
 * `conn:local::<profile>` form instead (colons are invalid in profile names,
 * so it cannot collide).
 */
export function resolveRegistryLocalRoute(
  profile: null | string | undefined,
  opts: { globalRemote?: boolean; profileRemoteOverride?: boolean } = {}
): RegistryLocalRoute {
  const profileKey = String(profile ?? '').trim() || 'default'

  if (opts.globalRemote || opts.profileRemoteOverride) {
    return { delegate: false, poolKey: `${backendScopePrefix(LOCAL_CONNECTION_ID)}${profileKey}` }
  }

  return { delegate: true, poolKey: profileKey }
}

/**
 * Whether the roster enumeration should SKIP the registry's local entry as
 * connect-on-demand. True when the local source is the forced-local route
 * (primary resolves remote — enumerating would spawn a local backend the
 * user never asked for, minting a phantom `default` agent and forcing
 * -device handles onto the real one) AND no forced-local child is already
 * pooled. Pure — main.ts feeds it the live route + pool keys.
 */
export function shouldDeferLocalEnumeration(
  route: RegistryLocalRoute,
  poolKeys: Iterable<string>,
  connectionId: string = LOCAL_CONNECTION_ID
): boolean {
  if (route.delegate) {
    return false
  }

  const prefix = backendScopePrefix(connectionId)

  return ![...poolKeys].some(key => String(key).startsWith(prefix))
}

// ── Union agent roster ──────────────────────────────────────────────────────

export interface ConnectionAgents {
  connection: RegistryConnection
  /** Profile names enumerated from the connection, or null when unreachable /
   * connect-on-demand (ssh not yet dialed). */
  profiles: null | string[]
  /** Credential-free profile metadata from the same connection. Kept separate
   * from `profiles` so old enumerators can continue returning names only. */
  profileMetadata?: Record<string, RosterProfileMetadata>
  /** Present when profiles is null: why enumeration was skipped. */
  error?: string
  /** Stable backend identity from the connection's /api/status (`install_id`).
   * Two connections reporting the same id are the SAME physical install
   * registered under two addresses (hostname + Tailscale IP), so the roster
   * collapses their rows. Absent on older backends → no collapse (fully
   * backward compatible). */
  installId?: string
}

export interface RosterAgent {
  connectionId: string
  connectionKind: ConnectionKind
  connectionLabel: string
  profile: string
  /** Backend profile when the registry route maps the Desktop profile name. */
  targetProfile?: string
  /** Bare profile name, or `<profile>-<label-slug>` when the profile name
   * exists on more than one registered source (the @name-device rule). */
  handle: string
  /** Rich metadata for this exact connection + profile, when enumerated. */
  profileMetadata?: RosterProfileMetadata
}

export interface RosterProfileMetadata {
  display_name?: string
  title?: string
  ui_meta?: Record<string, unknown>
  has_avatar?: boolean
}

/**
 * Roster enumeration skips undialed sources (connect-on-demand) and reports
 * unreachable ones with `profiles: null`. Reuse the last successful profile
 * list so Bot Mode does not go empty (or drop to a partial roster) the moment
 * a source is briefly unreachable — SSH tunnels drop on sleep/wake, and a
 * remote gateway bounce (VPS restart) otherwise erased its bots from the
 * roster until the next successful enumeration ("my 4 bots show as 2", Aug
 * 2026 bundle). Never-seen SSH sources still get a `default` seed so the
 * device is clickable; never-seen remote sources stay empty (no seed) since
 * an unreachable URL is not evidence a backend exists there.
 */
export function rememberSshEnumeration(
  enumeration: Pick<ConnectionAgents, 'error' | 'profiles'>,
  cached: null | string[] | undefined,
  kind: ConnectionKind
): Pick<ConnectionAgents, 'error' | 'profiles'> {
  if (enumeration.profiles && enumeration.profiles.length > 0) {
    return enumeration
  }

  if (kind === 'local') {
    return enumeration
  }

  if (cached && cached.length > 0) {
    return { profiles: cached, error: enumeration.error }
  }

  if (kind === 'ssh' && enumeration.error === 'connect-on-demand') {
    return { profiles: ['default'], error: 'connect-on-demand' }
  }

  return enumeration
}

/** Whether an undialed SSH source should be inventoried again. Cached
 *  successes never retry. Failures retry after `retryAfterMs` so a cold box
 *  does not stay seeded as `default` until the user hits Test. */
export function shouldRetrySshInventory(
  hasCache: boolean,
  lastAttemptMs: null | number | undefined,
  nowMs: number,
  retryAfterMs = 60_000
): boolean {
  if (hasCache) {
    return false
  }

  if (lastAttemptMs == null) {
    return true
  }

  return nowMs - lastAttemptMs >= retryAfterMs
}

const PROFILE_NAME_RE = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/

/** Turn `ls ~/.hermes/profiles` output into roster names. Always includes
 *  `default`. Drops rollback snapshots and junk lines. */
export function parseRemoteProfileListing(text: string): string[] {
  const names = new Set<string>(['default'])

  for (const raw of String(text || '').split(/\r?\n/)) {
    const name = raw.trim()

    if (!name || name.startsWith('.') || name.endsWith('.rollback-old')) {
      continue
    }

    if (!PROFILE_NAME_RE.test(name)) {
      continue
    }

    names.add(name)
  }

  return ['default', ...[...names].filter(name => name !== 'default').sort()]
}

/**
 * Flatten per-connection profile enumerations into the union roster, applying
 * the duplicate-handle rule ONCE across all sources. Pure so the disambiguation
 * policy is testable without IPC; main.ts feeds it live enumerations.
 */
export function buildAgentRoster(
  enumerations: ConnectionAgents[],
  opts: { primaryConnectionId?: string } = {}
): RosterAgent[] {
  // A connection can transiently report the same profile more than once (or
  // arrive twice while registry state is reconciling). A roster row represents
  // one routable identity, so collapse strictly by connection + profile before
  // counting names for @name-device disambiguation.
  const identities = new Map<
    string,
    {
      connection: RegistryConnection
      installId?: string
      order: number
      profile: string
      profileMetadata?: RosterProfileMetadata
    }
  >()

  let order = 0

  for (const { connection, installId, profiles, profileMetadata } of enumerations) {
    for (const profile of profiles || []) {
      const name = String(profile || '').trim() || 'default'
      const key = `${connection.id}\0${name}`

      if (!identities.has(key)) {
        identities.set(key, {
          connection,
          installId,
          order,
          profile: name,
          ...(profileMetadata?.[name] ? { profileMetadata: profileMetadata[name] } : {})
        })
      }
    }

    order += 1
  }

  // Backend-identity collapse: two connections reporting the same install_id
  // are the SAME physical install registered under two addresses, so their
  // (install, profile) rows are one bot, not two. Connections without an id
  // (older backends, undialed ssh) keep a per-connection key — no collapse.
  const backends = new Map<
    string,
    { connection: RegistryConnection; order: number; profile: string; profileMetadata?: RosterProfileMetadata }[]
  >()

  for (const { connection, installId, order: rank, profile, profileMetadata } of identities.values()) {
    const key = installId ? `id:${installId}\0${profile}` : `conn:${connection.id}\0${profile}`
    const group = backends.get(key)

    if (group) {
      group.push({ connection, order: rank, profile, profileMetadata })
    } else {
      backends.set(key, [{ connection, order: rank, profile, profileMetadata }])
    }
  }

  const rows = [...backends.values()].map(group => pickCanonicalConnection(group, opts.primaryConnectionId))

  // The @name-device duplicate-handle rule runs AFTER the collapse, so a
  // profile that only *looked* duplicated (one box, two addresses) keeps its
  // bare name once the rows are recognized as one backend.
  const counts = new Map<string, number>()

  for (const { profile } of rows) {
    counts.set(profile, (counts.get(profile) || 0) + 1)
  }

  const roster: RosterAgent[] = []

  for (const { connection, profile, profileMetadata } of rows) {
    roster.push({
      connectionId: connection.id,
      connectionKind: connection.kind,
      connectionLabel: connection.label,
      profile,
      targetProfile: connection.remoteProfile || profile,
      handle: agentHandle(profile, connection.label, (counts.get(profile) || 0) > 1),
      ...(profileMetadata ? { profileMetadata } : {})
    })
  }

  return roster
}

/** Deterministic route priority for same-backend rows: local is definitionally
 * this box; ssh beats HTTP remotes; cloud last. */
const CANONICAL_KIND_PRIORITY: Record<ConnectionKind, number> = { cloud: 3, local: 0, remote: 2, ssh: 1 }

/**
 * Which connection represents a collapsed same-backend roster row: the ACTIVE
 * (primary) connection when it is one of the candidates — the row should route
 * where the window already routes — else the highest kind priority, else the
 * earliest-registered (enumeration order follows registry order).
 */
function pickCanonicalConnection<T extends { connection: RegistryConnection; order: number }>(
  candidates: T[],
  primaryConnectionId?: string
): T {
  const active = primaryConnectionId ? candidates.find(c => c.connection.id === primaryConnectionId) : undefined

  if (active) {
    return active
  }

  return [...candidates].sort(
    (a, b) =>
      CANONICAL_KIND_PRIORITY[a.connection.kind] - CANONICAL_KIND_PRIORITY[b.connection.kind] || a.order - b.order
  )[0]
}

// ── Fan-out update eligibility ──────────────────────────────────────────────

export interface UpdateEligibility {
  eligible: boolean
  /** Present when not eligible: 'cloud-managed' (platform updates it). */
  reason?: 'cloud-managed'
}

/**
 * Whether "Update all instances" may drive this connection. Hermes Cloud
 * instances are platform-managed — we never run `hermes update` against them.
 * Local, remote, and ssh sources are all eligible (reachability and busy
 * checks happen at dispatch time, not here).
 */
export function updateEligibility(connection: RegistryConnection): UpdateEligibility {
  if (connection.kind === 'cloud') {
    return { eligible: false, reason: 'cloud-managed' }
  }

  return { eligible: true }
}

/** Mint a registry-unique id from a label (slug, then -2/-3… suffixes). */
export function connectionIdForLabel(label: string, taken: Iterable<string>): string {
  const used = new Set([...taken])
  const base = labelSlug(label)

  if (!used.has(base) && base !== LOCAL_CONNECTION_ID) {
    return base
  }

  for (let n = 2; ; n += 1) {
    const candidate = `${base}-${n}`

    if (!used.has(candidate) && candidate !== LOCAL_CONNECTION_ID) {
      return candidate
    }
  }
}

// ── Validation ──────────────────────────────────────────────────────────────

export interface ConnectionInput {
  id?: string
  kind: ConnectionKind
  label: string
  url?: string
  authMode?: string
  token?: unknown
  headers?: Record<string, unknown>
  org?: string
  host?: string
  user?: string
  port?: number | string
  keyPath?: string
  remoteHermesPath?: string
  remoteProfile?: string
}

/**
 * Validate + normalize a save payload into a RegistryConnection.
 * Throws with a user-facing message on any violation. `registry` supplies the
 * uniqueness context; when `input.id` matches an existing entry this is an
 * edit and that entry is excluded from the label-collision check.
 */
export function normalizeConnectionInput(input: ConnectionInput, registry: ConnectionRegistry): RegistryConnection {
  const label = String(input.label || '').trim()

  if (!label) {
    throw new Error('Every connection needs a name. Give this instance a device name (e.g. "Homelab", "Work laptop").')
  }

  if (label.length > LABEL_MAX) {
    throw new Error(`Connection name is too long (max ${LABEL_MAX} characters).`)
  }

  const key = labelKey(label)
  const collision = registry.connections.find(c => labelKey(c.label) === key && c.id !== input.id)

  if (collision) {
    throw new Error(`A connection named "${collision.label}" already exists. Connection names must be unique.`)
  }

  const kind = input.kind

  if (kind === 'local') {
    // The local entry is managed by the app; only its label is editable.
    return { id: LOCAL_CONNECTION_ID, kind: 'local', label }
  }

  // The reserved local id can never be claimed by a non-local entry — a
  // crafted IPC payload ({id:'local', kind:'remote', …}) would otherwise
  // replace the local entry via upsert and break the exactly-one-local
  // invariant. connectionIdForLabel never mints 'local'; reject it when
  // supplied, too.
  if (input.id === LOCAL_CONNECTION_ID) {
    throw new Error('The id "local" is reserved for the local connection.')
  }

  const id =
    input.id ||
    connectionIdForLabel(
      label,
      registry.connections.map(c => c.id)
    )

  if (kind === 'ssh') {
    const ssh = normalizeSshConfig({
      mode: 'ssh',
      host: input.host,
      user: input.user,
      port: input.port,
      keyPath: input.keyPath,
      remoteHermesPath: input.remoteHermesPath,
      remoteProfile: input.remoteProfile
    })

    if (!ssh) {
      throw new Error('SSH connections need a host.')
    }

    const { mode: _mode, ...sshFields } = ssh

    // Duplicate prevention (enforced here so a crafted IPC payload can't slip
    // past the editor's check): two ssh entries collide on the same
    // user@host:port + remote profile.
    const sshKey = (c: { host?: string; port?: number; remoteProfile?: string; user?: string }) =>
      `${(c.user || '').toLowerCase()}@${(c.host || '').toLowerCase()}:${c.port ?? 22}::${(c.remoteProfile || '').trim()}`

    const sshDupe = registry.connections.find(c => c.kind === 'ssh' && c.id !== id && sshKey(c) === sshKey(sshFields))

    if (sshDupe) {
      throw new Error(`A connection to this SSH host already exists ("${sshDupe.label}").`)
    }

    return { id, kind: 'ssh', label, ...sshFields }
  }

  if (kind === 'remote' || kind === 'cloud') {
    // normalizeRemoteBaseUrl throws its own user-facing message on bad input.
    const url = normalizeRemoteBaseUrl(input.url)

    // Duplicate prevention: remote/cloud entries collide on the normalized URL
    // (trimmed, trailing slashes stripped, lowercased) regardless of kind — a
    // cloud entry and a remote entry pointing at the same gateway are dupes.
    const urlKey = (value: string) => value.trim().replace(/\/+$/, '').toLowerCase()

    const urlDupe = registry.connections.find(
      c => (c.kind === 'remote' || c.kind === 'cloud') && c.id !== id && urlKey(c.url || '') === urlKey(url)
    )

    if (urlDupe) {
      throw new Error(`A connection to this gateway URL already exists ("${urlDupe.label}").`)
    }

    const authMode = normAuthMode(input.authMode)
    const entry: RegistryConnection = { id, kind, label, url, authMode }

    // A token is only meaningful for token-auth remotes. Dropping it here is
    // what clears the stale envelope when an entry is switched token→oauth
    // (or is a cloud entry, which authenticates via the portal session) —
    // otherwise dead secret material rides along on the edited entry.
    if (input.token !== undefined && kind === 'remote' && authMode === 'token') {
      entry.token = input.token
    }

    // Extra gateway headers (access-proxy credentials) apply to any
    // remote-shaped entry regardless of auth mode — Cloudflare Access sits in
    // front of both token- and OAuth-gated gateways. Normalization drops
    // transport-/Hermes-managed names; an empty result stores nothing.
    if (input.headers !== undefined) {
      const headers = normalizeRemoteHeaders(input.headers)

      if (Object.keys(headers).length > 0) {
        entry.headers = headers
      }
    }

    const org = String(input.org || '').trim()

    if (kind === 'cloud' && org) {
      entry.org = org
    }

    return entry
  }

  throw new Error(`Unknown connection kind: ${String(kind)}`)
}

/**
 * Merge a (possibly partial) edit payload over the stored entry so fields the
 * editor doesn't carry survive a save. Renaming a migrated cloud entry must
 * not drop its `org` (downstream update-fanout uses it to skip
 * platform-managed instances), and renaming an ssh entry must not drop
 * `remoteHermesPath`/`remoteProfile`. Only fields the payload explicitly
 * carries (non-undefined) override; `token` is deliberately NOT merged here —
 * the caller owns secret handling.
 */
export function mergeConnectionInput(input: ConnectionInput, existing?: null | RegistryConnection): ConnectionInput {
  if (!existing || existing.kind !== input.kind) {
    return input
  }

  const merged: ConnectionInput = { ...input }

  const inherit = (field: keyof ConnectionInput & keyof RegistryConnection) => {
    if (merged[field] === undefined && existing[field] !== undefined) {
      ;(merged as unknown as Record<string, unknown>)[field] = existing[field]
    }
  }

  inherit('url')
  inherit('authMode')
  inherit('org')
  inherit('host')
  inherit('keyPath')
  inherit('remoteHermesPath')
  inherit('remoteProfile')
  // Headers inherit like other dial fields: an edit payload that omits the
  // field keeps the stored set; an explicit payload (even {}) is
  // authoritative so the editor can clear them.
  inherit('headers')

  // ssh user/port: the editor shows ONE composite host field (user@host:port),
  // and normalizeSshConfig gives explicit user/port fields precedence over the
  // parsed host string. Inheriting stored user/port alongside a NEW host string
  // would resurrect the old values over what the user just typed — so when the
  // payload carries a host, the host string is authoritative and stored
  // user/port are NOT inherited.
  if (input.host === undefined || !String(input.host).trim()) {
    inherit('user')
    inherit('port')
  }

  return merged
}

/**
 * True when an edit changes how a connection is DIALED — endpoint, auth, or
 * ssh routing fields — as opposed to a cosmetic label rename. Callers use
 * this to decide whether live pooled backends / renderer sockets for the
 * connection must be recycled after a save: a label-only edit keeps traffic
 * flowing, while a url/token/host change means everything currently open
 * points at the OLD target and must be torn down and re-dialed.
 */
export function connectionDialFieldsChanged(before: RegistryConnection, after: RegistryConnection): boolean {
  if (before.kind !== after.kind) {
    return true
  }

  const fields: (keyof RegistryConnection)[] = [
    'url',
    'authMode',
    'org',
    'host',
    'user',
    'port',
    'keyPath',
    'remoteHermesPath',
    'remoteProfile'
  ]

  for (const field of fields) {
    if ((before[field] ?? null) !== (after[field] ?? null)) {
      return true
    }
  }

  // Token envelopes are opaque here (main.ts encrypts). An edit that carries
  // no new token inherits the stored envelope verbatim, so structural
  // equality is exact for the label-only case.
  if (JSON.stringify(before.token ?? null) !== JSON.stringify(after.token ?? null)) {
    return true
  }

  // Headers are dial material too: a changed access-proxy credential means
  // every open socket/backend authenticated with the OLD set.
  return JSON.stringify(before.headers ?? null) !== JSON.stringify(after.headers ?? null)
}

// ── Registry-level operations (all pure: return a new registry) ────────────

function localEntry(label = 'This device'): RegistryConnection {
  return { id: LOCAL_CONNECTION_ID, kind: 'local', label }
}

/**
 * Coerce arbitrary parsed JSON into a valid registry: version stamped, a
 * local entry guaranteed, labels de-duplicated defensively (suffix, never
 * drop), primary always pointing at an existing entry. A hand-edited or
 * corrupt file degrades to a minimal local-only registry rather than
 * throwing at boot.
 */
export function normalizeRegistry(raw: unknown): ConnectionRegistry {
  const parsed = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  const rawConnections = Array.isArray(parsed.connections) ? parsed.connections : []
  const seenLabels = new Set<string>()
  const seenIds = new Set<string>()
  const connections: RegistryConnection[] = []
  const quarantined: QuarantinedRegistryEntry[] = []

  const quarantine = (reason: string, entry: unknown) => {
    if (quarantined.length < REGISTRY_QUARANTINE_CAP) {
      quarantined.push({ reason, entry })
    }
  }

  // Entries quarantined by a previous load are user data too — carry them
  // through every subsequent normalize/write cycle rather than dropping them
  // the first time the file is rewritten.
  if (Array.isArray(parsed.quarantined)) {
    for (const item of parsed.quarantined) {
      if (item && typeof item === 'object' && 'entry' in (item as Record<string, unknown>)) {
        quarantine(
          String((item as Record<string, unknown>).reason || 'unknown'),
          (item as Record<string, unknown>).entry
        )
      }
    }
  }

  // Best-effort plain-data copy for entries that blew up mid-normalization —
  // the raw object may carry whatever poisoned it, so never persist it as-is.
  const safeEntryCopy = (item: unknown) => {
    try {
      return JSON.parse(JSON.stringify(item))
    } catch {
      return { unserializable: true }
    }
  }

  for (const item of rawConnections) {
    if (!item) {
      continue // null/false/'' carry no user data
    }

    if (typeof item !== 'object') {
      // A string/number here is usually a mangled hand-edit — still user data.
      quarantine('entry-malformed', item)

      continue
    }

    // One bad entry must never abort the whole registry load (#94246): any
    // unexpected throw quarantines THIS entry and the loop moves on.
    try {
      const entry = item as Record<string, unknown>
      const kind = entry.kind

      if (kind !== 'local' && kind !== 'remote' && kind !== 'cloud' && kind !== 'ssh') {
        quarantine('entry-unrecognized-kind', item)

        continue
      }

      let label = String(entry.label || '').trim()

      if (!label) {
        // Defensive: registry entries are always written with labels, but a
        // hand-edited file may drop one. Derive rather than discard.
        label =
          kind === 'ssh' ? String(entry.host || 'ssh') : hostLabelFromBaseUrl(String(entry.url || '')) || String(kind)
      }

      label = uniqueLabel(label, seenLabels)

      let id = kind === 'local' ? LOCAL_CONNECTION_ID : String(entry.id || '').trim()

      if (!id || (seenIds.has(id) && kind !== 'local')) {
        id = connectionIdForLabel(label, seenIds)
      }

      if (seenIds.has(id)) {
        continue // second 'local' entry — first one wins
      }

      seenLabels.add(labelKey(label))
      seenIds.add(id)

      const clean: RegistryConnection = { id, kind, label }

      if (kind === 'remote' || kind === 'cloud') {
        const url = String(entry.url || '').trim()

        if (!url) {
          quarantine('entry-missing-url', item)

          continue
        }

        clean.url = url
        clean.authMode = normAuthMode(entry.authMode)

        if (entry.token !== undefined) {
          clean.token = entry.token
        }

        const storedHeaders = normalizeRemoteHeaders(entry.headers)

        if (Object.keys(storedHeaders).length > 0) {
          clean.headers = storedHeaders
        }

        const org = String(entry.org || '').trim()

        if (kind === 'cloud' && org) {
          clean.org = org
        }
      } else if (kind === 'ssh') {
        const ssh = normalizeSshConfig({ ...entry, mode: 'ssh' })

        if (!ssh) {
          quarantine('entry-missing-ssh-host', item)

          continue
        }

        const { mode: _mode, ...sshFields } = ssh
        Object.assign(clean, sshFields)
      }

      connections.push(clean)
    } catch {
      quarantine('entry-normalization-failed', safeEntryCopy(item))
    }
  }

  if (!connections.some(c => c.kind === 'local')) {
    connections.unshift(localEntry())
  }

  const storedPrimary = String(parsed.primary || '').trim()
  const primary = connections.some(c => c.id === storedPrimary) ? storedPrimary : LOCAL_CONNECTION_ID
  const storedLastUsed = String(parsed.lastUsed || '').trim()

  const normalized: ConnectionRegistry = {
    version: REGISTRY_VERSION,
    primary,
    launchMode: parsed.launchMode === 'last-used' ? 'last-used' : 'primary',
    lastUsed: connections.some(c => c.id === storedLastUsed) ? storedLastUsed : primary,
    connections
  }

  if (quarantined.length > 0) {
    normalized.quarantined = quarantined
  }

  return normalized
}

/**
 * One-time import of the v1 connection.json shape (global `mode` + `remote`
 * block + per-profile `profiles` map) into a v2 registry. v1 had no labels,
 * so they are derived (URL host, SSH host, "This device") and uniqued by
 * suffixing. The active v1 global connection becomes the primary. The v1
 * file is left untouched by the caller — old builds keep working.
 *
 * Per-profile override entries become registry connections too (deduped by
 * URL/host against the global block), so a user who had `research` pinned to
 * a second gateway sees both sources registered on first launch.
 */
export function migrateV1ToRegistry(v1: unknown): ConnectionRegistry {
  const config = v1 && typeof v1 === 'object' ? (v1 as Record<string, any>) : {}
  const connections: RegistryConnection[] = [localEntry()]
  const byFingerprint = new Map<string, RegistryConnection>()

  const addRemoteLike = (block: Record<string, any>, kind: 'cloud' | 'remote'): null | RegistryConnection => {
    const url = String(block?.url || '').trim()

    if (!url) {
      return null
    }

    const fingerprint = `${kind}:${url}`
    const existing = byFingerprint.get(fingerprint)

    if (existing) {
      return existing
    }

    const label = uniqueLabel(
      hostLabelFromBaseUrl(url) || (kind === 'cloud' ? 'Hermes Cloud' : 'Remote gateway'),
      connections.map(c => c.label)
    )

    const entry: RegistryConnection = {
      id: connectionIdForLabel(
        label,
        connections.map(c => c.id)
      ),
      kind,
      label,
      url,
      authMode: normAuthMode(block.authMode)
    }

    if (block.token !== undefined) {
      entry.token = block.token
    }

    const v1Headers = normalizeRemoteHeaders(block.headers)

    if (Object.keys(v1Headers).length > 0) {
      entry.headers = v1Headers
    }

    const org = String(block.org || '').trim()

    if (kind === 'cloud' && org) {
      entry.org = org
    }

    connections.push(entry)
    byFingerprint.set(fingerprint, entry)

    return entry
  }

  const addSsh = (block: Record<string, any>): null | RegistryConnection => {
    const ssh = normalizeSshConfig({ ...block, mode: 'ssh' })

    if (!ssh) {
      return null
    }

    const fingerprint = `ssh:${ssh.user || ''}@${ssh.host}:${ssh.port || 22}`
    const existing = byFingerprint.get(fingerprint)

    if (existing) {
      return existing
    }

    const label = uniqueLabel(
      ssh.host,
      connections.map(c => c.label)
    )

    const { mode: _mode, ...sshFields } = ssh

    const entry: RegistryConnection = {
      id: connectionIdForLabel(
        label,
        connections.map(c => c.id)
      ),
      kind: 'ssh',
      label,
      ...sshFields
    }

    connections.push(entry)
    byFingerprint.set(fingerprint, entry)

    return entry
  }

  // Global connection → an entry + the primary designation.
  let primary = LOCAL_CONNECTION_ID
  const globalMode = config.mode

  if (modeIsRemoteLike(globalMode)) {
    const entry = addRemoteLike(config.remote || {}, globalMode === 'cloud' ? 'cloud' : 'remote')

    if (entry) {
      primary = entry.id
    }
  } else if (globalMode === 'ssh') {
    const entry = addSsh(config.remote || {})

    if (entry) {
      primary = entry.id
    }
  }

  // Per-profile overrides → additional registered sources (deduped).
  const profiles = config.profiles && typeof config.profiles === 'object' ? config.profiles : {}

  for (const block of Object.values(profiles) as Record<string, any>[]) {
    if (!block || typeof block !== 'object') {
      continue
    }

    if (modeIsRemoteLike(block.mode)) {
      addRemoteLike(block, block.mode === 'cloud' ? 'cloud' : 'remote')
    } else if (block.mode === 'ssh') {
      addSsh(block)
    } else if (block.mode === 'local' && block.savedSsh) {
      addSsh(block.savedSsh)
    }
  }

  return { version: REGISTRY_VERSION, primary, launchMode: 'primary', lastUsed: primary, connections }
}

/** Insert or replace by id. Input must already be normalized/validated. */
export function upsertConnection(registry: ConnectionRegistry, entry: RegistryConnection): ConnectionRegistry {
  const connections = registry.connections.some(c => c.id === entry.id)
    ? registry.connections.map(c => (c.id === entry.id ? entry : c))
    : [...registry.connections, entry]

  return { ...registry, connections }
}

/**
 * Remove a connection. The local entry is not removable; removing the
 * current primary retargets primary to local.
 */
export function removeConnection(registry: ConnectionRegistry, id: string): ConnectionRegistry {
  const target = registry.connections.find(c => c.id === id)

  if (!target) {
    return registry
  }

  if (target.kind === 'local') {
    throw new Error('The local connection cannot be removed.')
  }

  const primary = registry.primary === id ? LOCAL_CONNECTION_ID : registry.primary

  return {
    ...registry,
    primary,
    lastUsed: registry.lastUsed === id ? primary : registry.lastUsed,
    connections: registry.connections.filter(c => c.id !== id)
  }
}

/** Point the window/primary backend at another registered connection. */
export function setPrimaryConnection(registry: ConnectionRegistry, id: string): ConnectionRegistry {
  if (!registry.connections.some(c => c.id === id)) {
    throw new Error(`No connection with id "${id}".`)
  }

  return { ...registry, primary: id }
}

/** Remember the last source the Sessions workspace opened successfully. */
export function setLastUsedConnection(registry: ConnectionRegistry, id: string): ConnectionRegistry {
  if (!registry.connections.some(c => c.id === id)) {
    throw new Error(`No connection with id "${id}".`)
  }

  return { ...registry, lastUsed: id }
}

/**
 * Reconcile a successfully-coerced global v1 connection config into the v2
 * registry. Settings still writes connection.json for compatibility, but an
 * Apply must publish the same primary identity to connections.json in the
 * same transaction or the live remote descriptor has no connectionId.
 *
 * Remote-shaped entries are matched by normalized URL across remote/cloud so
 * changing provenance never duplicates a gateway. Existing identity and
 * user-chosen label win; a new entry derives both from the host. Switching to
 * local keeps registered remotes available while moving primary/last-used
 * back to This device.
 */
export function reconcileAppliedGlobalConnection(
  registry: ConnectionRegistry,
  config: Record<string, any>
): ConnectionRegistry {
  const mode = config?.mode

  if (!modeIsRemoteLike(mode)) {
    if (mode === 'local') {
      return { ...registry, primary: LOCAL_CONNECTION_ID, lastUsed: LOCAL_CONNECTION_ID }
    }

    // SSH registry identity is managed by its existing registry editor and
    // migration path. Do not reinterpret or delete it here.
    return registry
  }

  const block = config.remote && typeof config.remote === 'object' ? config.remote : {}
  const url = normalizeRemoteBaseUrl(block.url)

  const existing = registry.connections.find(connection => {
    if (connection.kind !== 'remote' && connection.kind !== 'cloud') {
      return false
    }

    try {
      return normalizeRemoteBaseUrl(connection.url) === url
    } catch {
      return false
    }
  })

  const kind: ConnectionKind = mode === 'cloud' ? 'cloud' : 'remote'

  const label =
    existing?.label ||
    uniqueLabel(
      hostLabelFromBaseUrl(url) || (kind === 'cloud' ? 'Hermes Cloud' : 'Remote gateway'),
      registry.connections.map(connection => connection.label)
    )

  const entry = normalizeConnectionInput(
    {
      id: existing?.id,
      kind,
      label,
      url,
      authMode: block.authMode,
      token: block.token,
      headers: block.headers,
      org: block.org
    },
    registry
  )

  return {
    ...upsertConnection(registry, entry),
    primary: entry.id,
    lastUsed: entry.id
  }
}

/**
 * Heal an already-created registry that never learned about the v1 route it is
 * supposed to be serving.
 *
 * `migrateV1ToRegistry` runs exactly once — only when connections.json does not
 * exist. A user who was local at that moment and configured a remote gateway
 * afterwards (Settings -> Gateway writes connection.json alone) ends up with a
 * live remote that the registry cannot name: `resolvedConnectionId` returns
 * null, `primary` still says `local`, and every launch force-switches the
 * window onto a fresh local backend seconds after boot. Deleting
 * connections.json by hand is the only recovery today.
 *
 * Deliberately narrow: heal ONLY when the v1 global route has no matching
 * registry entry at all. That is the drift state and nothing else. If the
 * route is already registered but `primary` names another source, the user
 * chose that in the Connections panel and we leave it alone.
 *
 * SSH drifts the same way remote does: a v1 global `mode:'ssh'` route (host,
 * no url) written by Settings after the one-shot migration has no registry
 * identity, so `resolvedConnectionId` returns null, `primary` stays `local`,
 * and every launch re-homes the window onto a local backend — and because the
 * heal used to skip SSH entirely, the two files re-drifted after every update
 * relaunch instead of converging once.
 */
export function reconcileRegistryDrift(
  registry: ConnectionRegistry,
  v1: unknown
): { changed: boolean; registry: ConnectionRegistry } {
  const config = v1 && typeof v1 === 'object' ? (v1 as Record<string, any>) : {}
  const unchanged = { changed: false, registry }

  if (config.mode === 'ssh') {
    const ssh = normalizeSshConfig({
      ...(config.remote && typeof config.remote === 'object' ? config.remote : {}),
      mode: 'ssh'
    })

    if (!ssh) {
      // A v1 SSH route without a usable host is not a route we can register.
      return unchanged
    }

    const target = normalizedSshTarget(ssh)

    const alreadyRegistered = registry.connections.some(
      connection =>
        connection.kind === 'ssh' &&
        normalizedSshTarget(connection) === target &&
        (connection.port ?? 22) === (ssh.port ?? 22)
    )

    if (alreadyRegistered) {
      // Route is known; if primary names another source, that is the user's
      // Connections-panel choice, not drift.
      return unchanged
    }

    const { mode: _mode, ...sshFields } = ssh

    let entry: RegistryConnection

    try {
      entry = normalizeConnectionInput(
        {
          kind: 'ssh',
          label: uniqueLabel(
            ssh.host,
            registry.connections.map(connection => connection.label)
          ),
          ...sshFields
        },
        registry
      )
    } catch {
      // Validation failure (e.g. a crafted collision) must not corrupt the
      // registry; the v1 path keeps failing the way it already does.
      return unchanged
    }

    return {
      changed: true,
      registry: { ...upsertConnection(registry, entry), primary: entry.id, lastUsed: entry.id }
    }
  }

  if (!modeIsRemoteLike(config.mode)) {
    return unchanged
  }

  const block = config.remote && typeof config.remote === 'object' ? config.remote : {}

  let url = ''

  try {
    url = normalizeRemoteBaseUrl(block.url)
  } catch {
    // An unparseable v1 URL is not a route we can register. The v1 path keeps
    // failing the way it already does; do not corrupt the registry over it.
    return unchanged
  }

  if (!url) {
    return unchanged
  }

  const alreadyRegistered = registry.connections.some(connection => {
    if (connection.kind !== 'remote' && connection.kind !== 'cloud') {
      return false
    }

    try {
      return normalizeRemoteBaseUrl(connection.url) === url
    } catch {
      return false
    }
  })

  if (alreadyRegistered) {
    return unchanged
  }

  return { changed: true, registry: reconcileAppliedGlobalConnection(registry, config) }
}

/** Choose whether launch restores the explicit primary or the last-used source. */
export function setConnectionLaunchMode(registry: ConnectionRegistry, launchMode: string): ConnectionRegistry {
  if (launchMode !== 'last-used' && launchMode !== 'primary') {
    throw new Error(`Unknown connection launch mode "${String(launchMode)}".`)
  }

  return { ...registry, launchMode }
}
