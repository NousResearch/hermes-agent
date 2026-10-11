interface ExecutionAuthority {
  epoch: string
  generation: number
  terminal: boolean
  retiredEpochs: Set<string>
}

/**
 * The owner's wire stamp: `SessionEvents.publish` spreads the claimed
 * execution onto the event params, beside `type`/`payload`, never inside the
 * payload. `authority_epoch` is the integer runtime epoch; the fence keys on
 * its string form because epochs are opaque identities, never ordered.
 */
export interface ExecutionStampedEvent {
  admission_id?: unknown
  authority_epoch?: unknown
  execution_generation?: unknown
  payload?: unknown
}

const LIFECYCLE_TYPES = ['session.info', 'message.start', 'message.complete', 'message.error']

/** The event's opaque epoch (string form) and generation, or null when it is unversioned. */
function executionStamp(event?: ExecutionStampedEvent): { epoch: string; generation: number } | null {
  const rawEpoch = event?.authority_epoch
  const epoch = typeof rawEpoch === 'number' && Number.isSafeInteger(rawEpoch) ? String(rawEpoch) : rawEpoch
  const generation = event?.execution_generation
  const versioned = typeof epoch === 'string' && epoch.length > 0 && typeof generation === 'number' && Number.isSafeInteger(generation) && generation >= 0

  return versioned ? { epoch, generation } : null
}

/** Whether a versioned lifecycle frame is fenced off by the previous owner (retiring its epoch on a valid takeover). */
function fencedByPrevious(previous: ExecutionAuthority, type: string, epoch: string, generation: number, payload: Record<string, unknown> | undefined): boolean {
  if (previous.retiredEpochs.has(epoch)) {return true}

  if (previous.epoch === epoch) {
    if (generation < previous.generation) {return true}

    return generation === previous.generation && previous.terminal && (type === 'message.start' || payload?.running === true)
  }

  // Only an owner snapshot/start can establish a new epoch, not a late finalizer.
  if (type !== 'session.info' && type !== 'message.start') {return true}
  previous.retiredEpochs.add(previous.epoch)

  return false
}

/** An unstamped lifecycle frame: no execution is claimed, so it changes nothing the fence tracks. */
function acceptUnstampedLifecycle(previous: ExecutionAuthority | undefined, type: string, event?: ExecutionStampedEvent): boolean {
  // An admission-only frame (a queued row's cancellation: the owner's epoch and the row's
  // admission, no execution generation) is never a turn's lifecycle, before or after one.
  if (event?.authority_epoch !== undefined && typeof event.admission_id === 'string') {return false}

  if (!previous) {return true}
  const payload = event?.payload as Record<string, unknown> | undefined
  // The owner's idle snapshot (queue, revision) carries the session's last generation in its
  // payload; only an older owner's is stale.
  const generation = payload?.execution_generation

  return type === 'session.info' && payload?.running !== true && typeof generation === 'number' && generation >= previous.generation
}

/** Each transport consumer owns its map; epochs are opaque, never ordered. */
export function acceptExecutionEvent(
  authorities: Map<string, ExecutionAuthority>,
  key: string,
  type: string,
  event?: ExecutionStampedEvent
): boolean {
  const lifecycle = LIFECYCLE_TYPES.includes(type)
  const previous = authorities.get(key)
  const payload = event?.payload as Record<string, unknown> | undefined
  const stamp = executionStamp(event) ?? (type === 'session.info' ? executionStamp(payload) : null)

  // Output may follow the current owner, but cannot establish another one.
  // Keep legacy unversioned output compatible; fence stamped late frames.
  if (!lifecycle) {
    return !stamp || !previous || (previous.epoch === stamp.epoch && previous.generation === stamp.generation && !previous.terminal)
  }

  if (!stamp) {return acceptUnstampedLifecycle(previous, type, event)}
  const { epoch, generation } = stamp
  const terminal = type === 'message.complete' || type === 'message.error' || payload?.running === false

  if (previous && fencedByPrevious(previous, type, epoch, generation, payload)) {return false}

  authorities.set(key, { epoch, generation, terminal: terminal || Boolean(previous?.epoch === epoch && previous.generation === generation && previous.terminal), retiredEpochs: previous?.retiredEpochs ?? new Set() })

  return true
}
