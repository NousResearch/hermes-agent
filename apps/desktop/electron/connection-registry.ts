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

export interface ConnectionRegistry {
  version: typeof REGISTRY_VERSION
  /** id of the connection that owns the window/primary backend. */
  primary: string
  connections: RegistryConnection[]
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

// ── Union agent roster ──────────────────────────────────────────────────────

export interface ConnectionAgents {
  connection: RegistryConnection
  /** Profile names enumerated from the connection, or null when unreachable /
   * connect-on-demand (ssh not yet dialed). */
  profiles: null | string[]
  /** Present when profiles is null: why enumeration was skipped. */
  error?: string
}

export interface RosterAgent {
  connectionId: string
  connectionKind: ConnectionKind
  connectionLabel: string
  profile: string
  /** Bare profile name, or `<profile>-<label-slug>` when the profile name
   * exists on more than one registered source (the @name-device rule). */
  handle: string
}

/**
 * Flatten per-connection profile enumerations into the union roster, applying
 * the duplicate-handle rule ONCE across all sources. Pure so the disambiguation
 * policy is testable without IPC; main.ts feeds it live enumerations.
 */
export function buildAgentRoster(enumerations: ConnectionAgents[]): RosterAgent[] {
  // A connection can transiently report the same profile more than once (or
  // arrive twice while registry state is reconciling). A roster row represents
  // one routable identity, so collapse strictly by connection + profile before
  // counting names for @name-device disambiguation.
  const identities = new Map<string, { connection: RegistryConnection; profile: string }>()

  for (const { connection, profiles } of enumerations) {
    for (const profile of profiles || []) {
      const name = String(profile || '').trim() || 'default'
      const key = `${connection.id}\0${name}`

      if (!identities.has(key)) {
        identities.set(key, { connection, profile: name })
      }
    }
  }

  const counts = new Map<string, number>()

  for (const { profile } of identities.values()) {
    counts.set(profile, (counts.get(profile) || 0) + 1)
  }

  const roster: RosterAgent[] = []

  for (const { connection, profile } of identities.values()) {
    roster.push({
      connectionId: connection.id,
      connectionKind: connection.kind,
      connectionLabel: connection.label,
      profile,
      handle: agentHandle(profile, connection.label, (counts.get(profile) || 0) > 1)
    })
  }

  return roster
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

    return { id, kind: 'ssh', label, ...sshFields }
  }

  if (kind === 'remote' || kind === 'cloud') {
    // normalizeRemoteBaseUrl throws its own user-facing message on bad input.
    const url = normalizeRemoteBaseUrl(input.url)
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

  for (const item of rawConnections) {
    if (!item || typeof item !== 'object') {
      continue
    }

    const entry = item as Record<string, unknown>
    const kind = entry.kind

    if (kind !== 'local' && kind !== 'remote' && kind !== 'cloud' && kind !== 'ssh') {
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
        continue
      }

      const { mode: _mode, ...sshFields } = ssh
      Object.assign(clean, sshFields)
    }

    connections.push(clean)
  }

  if (!connections.some(c => c.kind === 'local')) {
    connections.unshift(localEntry())
  }

  const primary = String(parsed.primary || '').trim()

  return {
    version: REGISTRY_VERSION,
    primary: connections.some(c => c.id === primary) ? primary : LOCAL_CONNECTION_ID,
    connections
  }
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

  return { version: REGISTRY_VERSION, primary, connections }
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

  return {
    ...registry,
    primary: registry.primary === id ? LOCAL_CONNECTION_ID : registry.primary,
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
