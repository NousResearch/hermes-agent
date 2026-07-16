export interface DurablePinnedSessions {
  exists: boolean
  pinned_session_ids: string[]
}

export interface PinnedSessionReconciliation {
  pinnedSessionIds: string[]
  shouldPersist: boolean
}

type SavePinnedSessions = (pinnedSessionIds: string[]) => Promise<unknown>
type WaitForRetry = (attempt: number) => Promise<void>

function normalize(ids: unknown): string[] {
  if (!Array.isArray(ids)) {
    return []
  }

  const seen = new Set<string>()
  const normalized: string[] = []

  for (const id of ids) {
    if (typeof id !== 'string') {
      continue
    }

    const value = id.trim()

    if (!value || seen.has(value)) {
      continue
    }

    seen.add(value)
    normalized.push(value)
  }

  return normalized
}

/**
 * Resolve the one-time startup handoff from legacy localStorage to the
 * machine-owned backend record. Once the backend record exists it is canonical,
 * including an intentionally empty list after a user unpinned every chat.
 */
export function reconcilePinnedSessions(
  localPinsAtRequest: unknown,
  localPinsNow: unknown,
  durable: DurablePinnedSessions
): PinnedSessionReconciliation {
  const baseline = normalize(localPinsAtRequest)
  const current = normalize(localPinsNow)
  const remote = normalize(durable.pinned_session_ids)

  if (!durable.exists) {
    return { pinnedSessionIds: current, shouldPersist: true }
  }

  const changedDuringRequest =
    baseline.length !== current.length || baseline.some((id, index) => id !== current[index])

  if (!changedDuringRequest) {
    return { pinnedSessionIds: remote, shouldPersist: false }
  }

  // Apply the local user's in-flight delta to the recovered durable list. This
  // preserves remote-only pins after a localStorage wipe while still honoring
  // removals, additions, and reorderings made before the GET completed.
  const baselineSet = new Set(baseline)
  const currentSet = new Set(current)
  const removed = new Set(baseline.filter(id => !currentSet.has(id)))
  const remoteSurvivors = remote.filter(id => !removed.has(id))
  const remoteSurvivorSet = new Set(remoteSurvivors)
  const reorderedKnown = current.filter(id => baselineSet.has(id) && remoteSurvivorSet.has(id))
  const remoteOnly = remoteSurvivors.filter(id => !baselineSet.has(id))
  const localOnly = current.filter(id => !baselineSet.has(id) && !remoteSurvivorSet.has(id))

  return { pinnedSessionIds: normalize([...reorderedKnown, ...remoteOnly, ...localOnly]), shouldPersist: true }
}

const defaultWaitForRetry: WaitForRetry = attempt =>
  new Promise(resolve => window.setTimeout(resolve, 250 * 2 ** (attempt - 1)))

/** Serialize writes and retry transient failures without reordering snapshots. */
export function createPinnedSessionWriter(
  save: SavePinnedSessions,
  waitForRetry: WaitForRetry = defaultWaitForRetry,
  maxAttempts = 3
): (pinnedSessionIds: readonly string[]) => Promise<void> {
  let queue = Promise.resolve()

  return pinnedSessionIds => {
    const snapshot = normalize(pinnedSessionIds)

    queue = queue
      .catch(() => undefined)
      .then(async () => {
        for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
          try {
            await save(snapshot)

            return
          } catch (error) {
            if (attempt === maxAttempts) {
              throw error
            }

            await waitForRetry(attempt)
          }
        }
      })

    return queue
  }
}
