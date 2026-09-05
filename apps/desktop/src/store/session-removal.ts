import { atom } from 'nanostores'

/** The renderer-side identity needed to scope an optimistic removal.
 *
 * A profile name is only unique inside one gateway. Rows from the registry
 * therefore use the complete `(connection_id, profile, session id)` tuple;
 * rows from the legacy primary path may omit `connection_id` and are handled
 * conservatively below.
 */
export interface SessionRemovalTarget {
  _lineage_ids?: null | readonly (null | string | undefined)[]
  _lineage_root_id?: null | string
  connection_id?: null | string
  id: string
  profile?: null | string
}

export type SessionRemovalInput = null | SessionRemovalTarget | string | undefined

export interface SessionRemovalScope {
  connectionId?: null | string
  profile?: null | string
}

const OWNER_KEY_PREFIX = 'session-removal:v1:owner:'
const PROFILE_KEY_PREFIX = 'session-removal:v1:profile:'
const LEGACY_KEY_PREFIX = 'session-removal:v1:legacy:'

interface NormalizedRemovalTarget {
  connectionId: null | string
  ids: string[]
  profile: null | string
}

type ParsedRemovalKey =
  | { connectionId: string; id: string; kind: 'owner'; profile: string }
  | { id: string; kind: 'legacy' | 'raw' }
  | { id: string; kind: 'profile'; profile: string }

function normalizeId(id: null | string | undefined): null | string {
  const trimmed = id?.trim()

  return trimmed || null
}

function normalizedIds(target: SessionRemovalTarget): string[] {
  const ids = [target.id, target._lineage_root_id, ...(target._lineage_ids ?? [])]
  const seen = new Set<string>()

  for (const id of ids) {
    const normalized = normalizeId(id)

    if (normalized) {
      seen.add(normalized)
    }
  }

  return [...seen]
}

function normalizeTarget(input: SessionRemovalInput): NormalizedRemovalTarget | null {
  if (typeof input === 'string') {
    const id = normalizeId(input)

    return id ? { connectionId: null, ids: [id], profile: null } : null
  }

  if (!input) {
    return null
  }

  const ids = normalizedIds(input)

  if (!ids.length) {
    return null
  }

  const connectionId = normalizeId(input.connection_id)
  const rawProfile = normalizeId(input.profile)

  // A tagged row with no explicit profile is still an exact owner on the
  // backend's default profile. An untagged row with no profile is genuinely
  // legacy and must not be promoted into an invented default owner.
  return {
    connectionId,
    ids,
    profile: rawProfile || (connectionId ? 'default' : null)
  }
}

function ownerKey(connectionId: string, profile: string, id: string): string {
  return `${OWNER_KEY_PREFIX}${JSON.stringify([connectionId, profile, id])}`
}

function profileKey(profile: string, id: string): string {
  return `${PROFILE_KEY_PREFIX}${JSON.stringify([profile, id])}`
}

function legacyKey(id: string): string {
  return `${LEGACY_KEY_PREFIX}${JSON.stringify(id)}`
}

function producerKeys(input: SessionRemovalInput): string[] {
  const target = normalizeTarget(input)

  if (!target) {
    return []
  }

  return target.ids.map(id => {
    if (target.connectionId && target.profile) {
      return ownerKey(target.connectionId, target.profile, id)
    }

    if (target.profile) {
      return profileKey(target.profile, id)
    }

    return legacyKey(id)
  })
}

function parseRemovalKey(entry: string): ParsedRemovalKey | null {
  const trimmed = entry.trim()

  if (!trimmed) {
    return null
  }

  if (trimmed.startsWith(OWNER_KEY_PREFIX)) {
    try {
      const value: unknown = JSON.parse(trimmed.slice(OWNER_KEY_PREFIX.length))

      if (
        Array.isArray(value) &&
        value.length === 3 &&
        typeof value[0] === 'string' &&
        typeof value[1] === 'string' &&
        typeof value[2] === 'string' &&
        value[0].trim() &&
        value[1].trim() &&
        value[2].trim()
      ) {
        return {
          connectionId: value[0].trim(),
          id: value[2].trim(),
          kind: 'owner',
          profile: value[1].trim()
        }
      }
    } catch {
      return null
    }

    return null
  }

  if (trimmed.startsWith(PROFILE_KEY_PREFIX)) {
    try {
      const value: unknown = JSON.parse(trimmed.slice(PROFILE_KEY_PREFIX.length))

      if (
        Array.isArray(value) &&
        value.length === 2 &&
        typeof value[0] === 'string' &&
        typeof value[1] === 'string' &&
        value[0].trim() &&
        value[1].trim()
      ) {
        return { id: value[1].trim(), kind: 'profile', profile: value[0].trim() }
      }
    } catch {
      return null
    }

    return null
  }

  if (trimmed.startsWith(LEGACY_KEY_PREFIX)) {
    try {
      const value: unknown = JSON.parse(trimmed.slice(LEGACY_KEY_PREFIX.length))

      return typeof value === 'string' && value.trim() ? { id: value.trim(), kind: 'legacy' } : null
    } catch {
      return null
    }
  }

  // Pre-v1 atoms stored the bare id. Keep reading those entries so an update
  // cannot accidentally re-enable a queued resume from the previous build.
  return { id: trimmed, kind: 'raw' }
}

function matchesTarget(entry: string, input: SessionRemovalInput): boolean {
  const parsed = parseRemovalKey(entry)
  const target = normalizeTarget(input)

  if (!parsed || !target || !target.ids.includes(parsed.id)) {
    return false
  }

  if (target.connectionId && target.profile) {
    if (parsed.kind === 'owner') {
      return parsed.connectionId === target.connectionId && parsed.profile === target.profile
    }

    // A legacy/profile-only tombstone carries no contrary owner evidence. It
    // may have been created by this owner before the row was fully stamped, so
    // an explicitly targeted resume remains blocked by it.
    return parsed.kind === 'profile' || parsed.kind === 'legacy' || parsed.kind === 'raw'
  }

  if (target.profile) {
    return parsed.kind === 'owner' || parsed.kind === 'profile'
      ? parsed.profile === target.profile
      : parsed.kind === 'legacy' || parsed.kind === 'raw'
  }

  // A bare id is ambiguous by definition. Any owner-scoped entry for that id
  // blocks it instead of guessing which gateway the caller meant.
  return true
}

function matchesUntombstoneTarget(entry: string, input: SessionRemovalInput): boolean {
  const parsed = parseRemovalKey(entry)
  const target = normalizeTarget(input)

  if (!parsed || !target || !target.ids.includes(parsed.id)) {
    return false
  }

  // An explicit owner can only roll back the key it could have produced. A
  // legacy/profile-only entry may belong to a different same-id owner, so
  // leaving it in place is the safe outcome until that ambiguous producer
  // performs its own unscoped rollback.
  if (target.connectionId && target.profile) {
    return parsed.kind === 'owner' && parsed.connectionId === target.connectionId && parsed.profile === target.profile
  }

  if (target.profile) {
    return parsed.kind === 'profile' && parsed.profile === target.profile
  }

  return parsed.kind === 'legacy' || parsed.kind === 'raw'
}

function equivalentEntries(left: string, right: string): boolean {
  if (left === right) {
    return true
  }

  const a = parseRemovalKey(left)
  const b = parseRemovalKey(right)

  if (!a || !b || a.id !== b.id) {
    return false
  }

  if (a.kind === 'owner' && b.kind === 'owner') {
    return a.connectionId === b.connectionId && a.profile === b.profile
  }

  if (a.kind === 'profile' && b.kind === 'profile') {
    return a.profile === b.profile
  }

  return (a.kind === 'legacy' || a.kind === 'raw') && (b.kind === 'legacy' || b.kind === 'raw')
}

function entryBelongsToScope(entry: string, scope: SessionRemovalScope): boolean {
  const parsed = parseRemovalKey(entry)

  if (!parsed) {
    return false
  }

  const connectionId = normalizeId(scope.connectionId)
  const profile = normalizeId(scope.profile) || (connectionId ? 'default' : null)

  if (parsed.kind === 'owner') {
    return Boolean(connectionId && profile && parsed.connectionId === connectionId && parsed.profile === profile)
  }

  if (parsed.kind === 'profile') {
    // A profile-only row is safe to prune only on the legacy, connection-less
    // path. With a known registry connection it could belong to another source.
    return Boolean(!connectionId && profile && parsed.profile === profile)
  }

  // Raw/legacy keys are the old primary-backend contract. They are scoped only
  // when the current source is also connection-less.
  return !connectionId
}

/** Return every stored/lineage id represented by a removal target. */
export function sessionRemovalIds(input: SessionRemovalInput): string[] {
  return normalizeTarget(input)?.ids ?? []
}

/** True when a session row matches an optimistic removal entry exactly enough
 * to be hidden. Callers pass the whole row so same-id gateway twins do not
 * share a tombstone. Bare/legacy rows intentionally match any owner entry for
 * the same id and therefore fail closed. */
export function hasSessionRemovalKey(entries: ReadonlySet<string>, input: SessionRemovalInput): boolean {
  return [...entries].some(entry => matchesTarget(entry, input))
}

// Client-side cache eviction (Apollo-style optimistic layer): the encoded
// entries the user just deleted/archived. The backend tree is a snapshot that
// still lists them until its next refresh, so the render-time overlay strips
// these so the tree matches the live `$sessions` cache exactly.
export const $removedSessionIds = atom<Set<string>>(new Set())

export function tombstoneSessions(inputs: Array<SessionRemovalInput>): void {
  const next = new Set($removedSessionIds.get())
  const before = next.size

  for (const input of inputs) {
    for (const key of producerKeys(input)) {
      next.add(key)
    }
  }

  if (next.size !== before) {
    $removedSessionIds.set(next)
  }
}

export function untombstoneSessions(inputs: Array<SessionRemovalInput>): void {
  const current = $removedSessionIds.get()

  if (!current.size) {
    return
  }

  const next = new Set([...current].filter(entry => !inputs.some(input => matchesUntombstoneTarget(entry, input))))

  if (next.size !== current.size) {
    $removedSessionIds.set(next)
  }
}

// Ids whose delete/archive RPC is still in flight. These use the same owner
// key as the tombstone so a refresh on gateway A cannot release gateway B's
// mutation barrier when the stored ids happen to be identical.
export const $sessionMutationsInFlight = atom<Set<string>>(new Set())

function mutateInFlight(inputs: Array<SessionRemovalInput>, add: boolean): void {
  const current = $sessionMutationsInFlight.get()
  const next = new Set(current)

  for (const input of inputs) {
    for (const key of producerKeys(input)) {
      if (add) {
        next.add(key)
      } else {
        next.delete(key)

        // A string input may be an update from the pre-v1 producer. Clear the
        // raw representation too, but never clear an exact owner's key.
        const parsed = parseRemovalKey(key)

        if (parsed && (parsed.kind === 'legacy' || parsed.kind === 'raw')) {
          next.delete(parsed.id)
        }
      }
    }
  }

  if (next.size !== current.size) {
    $sessionMutationsInFlight.set(next)
  }
}

export const beginSessionMutation = (inputs: Array<SessionRemovalInput>): void => mutateInFlight(inputs, true)
export const endSessionMutation = (inputs: Array<SessionRemovalInput>): void => mutateInFlight(inputs, false)

/** Keep only entries that are still listed by a source in `scopedIds`, or are
 * still in flight. Exact entries owned by another known connection are left
 * untouched because a tree read from gateway A cannot prove gateway B's delete
 * has landed. With no scope (the all-profile reader), legacy entries can be
 * reconciled globally but exact entries remain conservative until their owner
 * is read. */
export function pruneSessionRemovalState(scopedIds: ReadonlySet<string>, scope?: SessionRemovalScope): Set<string> {
  const scoped = new Set([...scopedIds].map(id => normalizeId(id)).filter((id): id is string => Boolean(id)))
  const current = $removedSessionIds.get()
  const inFlight = $sessionMutationsInFlight.get()

  return new Set(
    [...current].filter(entry => {
      if ([...inFlight].some(inFlightEntry => equivalentEntries(entry, inFlightEntry))) {
        return true
      }

      const parsed = parseRemovalKey(entry)

      if (!parsed) {
        return true
      }

      // The all-profile tree has no single connection owner. Its id-only
      // scope cannot prove that an exact owner tombstone has landed, so keep
      // that key until the owning connection is read.
      if (!scope && parsed.kind === 'owner') {
        return true
      }

      if (scope && !entryBelongsToScope(entry, scope)) {
        return true
      }

      return scoped.has(parsed.id)
    })
  )
}

/** The session is on its way out: already tombstoned, or its delete/archive
 * RPC is still in flight. Either way, nothing may resume it. */
export function isSessionRemovalPending(input: SessionRemovalInput): boolean {
  return (
    hasSessionRemovalKey($removedSessionIds.get(), input) ||
    hasSessionRemovalKey($sessionMutationsInFlight.get(), input)
  )
}
