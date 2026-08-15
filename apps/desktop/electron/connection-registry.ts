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

import { hostLabelFromBaseUrl, modeIsRemoteLike, normalizeRemoteBaseUrl, normalizeSshConfig, normAuthMode } from './connection-config'

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

  const id = input.id || connectionIdForLabel(label, registry.connections.map(c => c.id))

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

    if (input.token !== undefined) {
      entry.token = input.token
    }

    const org = String(input.org || '').trim()

    if (kind === 'cloud' && org) {
      entry.org = org
    }

    return entry
  }

  throw new Error(`Unknown connection kind: ${String(kind)}`)
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

    while (seenLabels.has(labelKey(label))) {
      label = `${label} 2`
    }

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

    let label = hostLabelFromBaseUrl(url) || (kind === 'cloud' ? 'Hermes Cloud' : 'Remote gateway')

    while (connections.some(c => labelKey(c.label) === labelKey(label))) {
      label = `${label} 2`
    }

    const entry: RegistryConnection = {
      id: connectionIdForLabel(label, connections.map(c => c.id)),
      kind,
      label,
      url,
      authMode: normAuthMode(block.authMode)
    }

    if (block.token !== undefined) {
      entry.token = block.token
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

    let label = ssh.host

    while (connections.some(c => labelKey(c.label) === labelKey(label))) {
      label = `${label} 2`
    }

    const { mode: _mode, ...sshFields } = ssh

    const entry: RegistryConnection = {
      id: connectionIdForLabel(label, connections.map(c => c.id)),
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
