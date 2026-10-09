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
  const stamp = executionStamp(event)

  // Output may follow the current owner, but cannot establish another one.
  // Keep legacy unversioned output compatible; fence stamped late frames.
  if (!lifecycle) {
    return !stamp || !previous || (previous.epoch === stamp.epoch && previous.generation === stamp.generation && !previous.terminal)
  }

  if (!stamp) {return !previous}
  const { epoch, generation } = stamp
  const terminal = type === 'message.complete' || type === 'message.error' || payload?.running === false

  if (previous && fencedByPrevious(previous, type, epoch, generation, payload)) {return false}

  authorities.set(key, { epoch, generation, terminal: terminal || Boolean(previous?.epoch === epoch && previous.generation === generation && previous.terminal), retiredEpochs: previous?.retiredEpochs ?? new Set() })

  return true
}
