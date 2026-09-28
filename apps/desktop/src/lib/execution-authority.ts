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

/** Each transport consumer owns its map; epochs are opaque, never ordered. */
export function acceptExecutionEvent(
  authorities: Map<string, ExecutionAuthority>,
  key: string,
  type: string,
  event?: ExecutionStampedEvent
): boolean {
  const lifecycle = ['session.info', 'message.start', 'message.complete', 'message.error'].includes(type)
  const previous = authorities.get(key)
  const payload = event?.payload as Record<string, unknown> | undefined
  const rawEpoch = event?.authority_epoch
  const epoch = typeof rawEpoch === 'number' && Number.isSafeInteger(rawEpoch) ? String(rawEpoch) : rawEpoch
  const generation = event?.execution_generation
  const versioned = typeof epoch === 'string' && epoch.length > 0 && typeof generation === 'number' && Number.isSafeInteger(generation) && generation >= 0

  // Output may follow the current owner, but cannot establish another one.
  // Keep legacy unversioned output compatible; fence stamped late frames.
  if (!lifecycle) {
    return !versioned || !previous || (previous.epoch === epoch && previous.generation === generation && !previous.terminal)
  }

  if (!versioned) {return !previous}
  const terminal = type === 'message.complete' || type === 'message.error' || payload?.running === false

  if (previous) {
    if (previous.retiredEpochs.has(epoch)) {return false}

    if (previous.epoch === epoch) {
      if (generation < previous.generation) {return false}

      if (generation === previous.generation && previous.terminal && (type === 'message.start' || payload?.running === true)) {return false}
    } else {
      // Only an owner snapshot/start can establish a new epoch, not a late finalizer.
      if (type !== 'session.info' && type !== 'message.start') {return false}
      previous.retiredEpochs.add(previous.epoch)
    }
  }

  authorities.set(key, { epoch, generation, terminal: terminal || Boolean(previous?.epoch === epoch && previous.generation === generation && previous.terminal), retiredEpochs: previous?.retiredEpochs ?? new Set() })

  return true
}
