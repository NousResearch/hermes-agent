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
 */
export function createSessionMutationClient() {
  const pending = new Map<string, Promise<SessionMutationIdentity>>()

  return async function mutate<T>(key: string, read: () => Promise<SessionMutationSnapshot>,
    send: (identity: SessionMutationIdentity) => Promise<T>, importing = false): Promise<T> {
    let prepared = pending.get(key)
    if (!prepared) {
      prepared = read().then(snapshot => {
        if (typeof snapshot?.exists !== 'boolean' || !Number.isSafeInteger(snapshot.runtime_revision)
          || snapshot.runtime_revision < 0 || (!importing && !snapshot.exists)
          || (snapshot.exists && (!Number.isSafeInteger(snapshot.runtime_generation) || snapshot.runtime_generation! < 0))) {
          throw new Error('Session mutation snapshot unavailable; refresh before trying again')
        }
        return { request_id: crypto.randomUUID(), expected_revision: snapshot.runtime_revision,
          ...(snapshot.exists ? { expected_generation: snapshot.runtime_generation! } : {}) }
      })
      pending.set(key, prepared)
    }
    let identity: SessionMutationIdentity
    try { identity = await prepared } catch (error) { pending.delete(key); throw error }
    try {
      const result = await send(identity)
      pending.delete(key)
      return result
    } catch (error) {
      // These responses definitively rejected the write. Show the conflict;
      // only a later explicit user action may read a new precondition.
      if (/revision_conflict|stale_generation|permission_denied|profile_mismatch|invalid_params|admission_conflict|session_busy|retained_receipts|runtime_coordination_required/.test(String(error))) {
        pending.delete(key)
      }
      throw error
    }
  }
}
