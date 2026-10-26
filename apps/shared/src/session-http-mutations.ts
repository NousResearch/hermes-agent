export interface SessionMutationSnapshot {
  session_id: string
  exists: boolean
  runtime_revision: number
  runtime_generation: number | null
}

export interface SessionMutationIdentity {
  request_id: string
  expected_revision: number
  expected_generation?: number
}

/** Keep an ambiguous write's identity until the same intent is acknowledged.
 * Keys include the captured destination and payload; neither retries nor a
 * profile switch may silently refresh the precondition of a possibly committed edit.
 * A key is a JSON array whose first two entries are the destination scope and the session id:
 * an acknowledged write retires that session's other retained identities (their revision is
 * superseded, so replaying them could only conflict or return a stale receipt).
 *
 * `absent` opts a write into treating "the session no longer exists" as committed
 * state rather than a precondition failure. A DELETE raced by another surface's
 * delete (retention purge, a second window) used to throw the snapshot error and
 * leave the caller's optimistic rollback re-listing a row no backend owns — the
 * undeletable ghost. The server's own DELETE endpoint already answers a genuinely
 * absent id with idempotent success; `absent` mirrors that contract client-side
 * without paying for the round trip: the snapshot read and the delete share one
 * destination scope, so its `exists: false` is the same verdict the DELETE would get.
 */
export function createSessionMutationClient() {
  const pending = new Map<string, Promise<SessionMutationIdentity | null>>()

  /** Retire this key's retained identity and every other retained identity of the
   * same session: their revision is superseded by the committed write (or, for the
   * absent path, by the deletion that already happened), so replaying them could
   * only conflict or return a stale receipt. */
  const retireRetainedIdentities = (key: string): void => {
    const owner = sessionOwner(key)

    for (const other of [...pending.keys()]) {
      if (other === key || (owner && sessionOwner(other) === owner)) {
        pending.delete(other)
      }
    }
  }

  return async function mutate<T>(key: string, read: () => Promise<SessionMutationSnapshot>,
    send: (identity: SessionMutationIdentity) => Promise<T>, importing = false,
    absent?: () => T): Promise<T> {
    let prepared = pending.get(key)
    if (!prepared) {
      prepared = read().then(snapshot => {
        if (typeof snapshot?.exists !== 'boolean' || !Number.isSafeInteger(snapshot.runtime_revision)
          || snapshot.runtime_revision < 0
          || (!importing && !snapshot.exists && !absent)
          || (snapshot.exists && (!Number.isSafeInteger(snapshot.runtime_generation) || snapshot.runtime_generation! < 0))) {
          throw new Error('Session mutation snapshot unavailable; refresh before trying again')
        }
        // An absent target opted into the `absent` path resolves to a null identity:
        // the deletion this write wanted is already committed, so no write is sent.
        // Imports (importing, no `absent`) keep the historic shape — an identity
        // without expected_generation for an anchor that does not exist yet.
        if (!snapshot.exists && absent) {
          return null
        }
        return { request_id: newMutationRequestId(), expected_revision: snapshot.runtime_revision,
          ...(snapshot.exists ? { expected_generation: snapshot.runtime_generation! } : {}) }
      })
      pending.set(key, prepared)
    }
    let identity: SessionMutationIdentity | null
    try { identity = await prepared } catch (error) { pending.delete(key); throw error }
    if (identity === null) {
      // The deletion this write wanted is already committed on the destination
      // scope: same retirement bookkeeping as an acknowledged write, no send.
      retireRetainedIdentities(key)

      return absent!()
    }
    try {
      const result = await send(identity)
      retireRetainedIdentities(key)
      return result
    } catch (error) {
      // These responses definitively rejected the write. Show the conflict;
      // only a later explicit user action may read a new precondition.
      if (/revision_conflict|stale_generation|permission_denied|profile_mismatch|invalid_params|admission_conflict|session_busy|retained_receipts|runtime_coordination_required|unknown_execution|not_found|mutation_identity_required/.test(String(error))) {
        pending.delete(key)
      }
      throw error
    }
  }
}

function newMutationRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    crypto.getRandomValues(bytes)
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256)
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

function sessionOwner(key: string): string | null {
  try {
    const parsed = JSON.parse(key)
    return Array.isArray(parsed) && parsed.length > 2 ? JSON.stringify(parsed.slice(0, 2)) : null
  } catch { return null }
}
