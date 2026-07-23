import { atom, type ReadableAtom } from 'nanostores'

import type { PetState } from './pet'

export type PetSignalState = 'working' | 'thinking' | 'blocked' | 'done' | 'failed'

/**
 * A provider-owned pet signal. The `(source, id)` pair is its stable identity.
 * `expiresAt` is an absolute epoch timestamp; omit it only when the owner will
 * explicitly clear the signal (for example, during plugin disposal).
 */
export interface PetSignal {
  /** Event timestamp used for arbitration; replacements must strictly increase it. */
  readonly createdAt: number
  readonly expiresAt?: number
  readonly id: string
  /** Higher values win arbitration. */
  readonly priority: number
  readonly source: string
  readonly state: PetSignalState
}

const $petSignalsWritable = atom<readonly PetSignal[]>([])
export const $petSignals: ReadableAtom<readonly PetSignal[]> = $petSignalsWritable

const MAX_TIMER_DELAY_MS = 2_147_483_647
let expiryTimer: ReturnType<typeof setTimeout> | undefined

// Tombstones live for the renderer process lifetime so delayed provider work
// cannot resurrect a cleared or expired identity. Provider IDs should be bounded.
const petSignalCreatedAtHighWatermarks = new Map<string, Map<string, number>>()

const PET_SIGNAL_STATE: Record<PetSignalState, PetState> = {
  working: 'run',
  thinking: 'review',
  blocked: 'waiting',
  done: 'wave',
  failed: 'failed'
}

function isPetSignalCurrent(signal: PetSignal, now: number): boolean {
  return signal.expiresAt === undefined || (Number.isFinite(signal.expiresAt) && signal.expiresAt > now)
}

function isPetSignalValid(signal: PetSignal, now: number): boolean {
  return (
    Number.isFinite(signal.createdAt) &&
    Number.isFinite(signal.priority) &&
    Object.hasOwn(PET_SIGNAL_STATE, signal.state) &&
    isPetSignalCurrent(signal, now)
  )
}

function getPetSignalCreatedAtHighWatermark(source: string, id: string): number | undefined {
  return petSignalCreatedAtHighWatermarks.get(source)?.get(id)
}

function recordPetSignalCreatedAt(signal: Pick<PetSignal, 'createdAt' | 'id' | 'source'>) {
  let sourceHighWatermarks = petSignalCreatedAtHighWatermarks.get(signal.source)

  if (!sourceHighWatermarks) {
    sourceHighWatermarks = new Map()
    petSignalCreatedAtHighWatermarks.set(signal.source, sourceHighWatermarks)
  }

  const currentHighWatermark = sourceHighWatermarks.get(signal.id)

  if (currentHighWatermark === undefined || signal.createdAt > currentHighWatermark) {
    sourceHighWatermarks.set(signal.id, signal.createdAt)
  }
}

function snapshotPetSignal(signal: PetSignal): PetSignal {
  return { ...signal }
}

function armPetSignalExpiry() {
  clearTimeout(expiryTimer)
  expiryTimer = undefined

  let expiresAt = Number.POSITIVE_INFINITY

  for (const signal of $petSignalsWritable.get()) {
    if (signal.expiresAt !== undefined && signal.expiresAt < expiresAt) {
      expiresAt = signal.expiresAt
    }
  }

  if (!Number.isFinite(expiresAt)) {
    return
  }

  const delay = Math.min(MAX_TIMER_DELAY_MS, Math.max(0, expiresAt - Date.now()))
  expiryTimer = setTimeout(() => {
    expiryTimer = undefined

    const now = Date.now()
    const current = $petSignalsWritable.get()
    const next = current.filter(signal => isPetSignalCurrent(signal, now))

    if (next.length !== current.length) {
      $petSignalsWritable.set(next)
    }

    armPetSignalExpiry()
  }, delay)
}

/** Insert one identity or apply a strictly newer valid replacement. */
export function upsertPetSignal(signal: PetSignal) {
  if (!isPetSignalValid(signal, Date.now())) {
    return
  }

  const highWatermark = getPetSignalCreatedAtHighWatermark(signal.source, signal.id)

  if (highWatermark !== undefined && highWatermark >= signal.createdAt) {
    return
  }

  const storedSignal = snapshotPetSignal(signal)
  const current = $petSignalsWritable.get()
  const index = current.findIndex(candidate => candidate.source === signal.source && candidate.id === signal.id)

  if (index >= 0 && current[index].createdAt >= storedSignal.createdAt) {
    return
  }

  const next = [...current]

  if (index < 0) {
    next.push(storedSignal)
  } else {
    next[index] = storedSignal
  }

  recordPetSignalCreatedAt(storedSignal)
  $petSignalsWritable.set(next)
  armPetSignalExpiry()
}

/** Remove every signal owned by a provider source. */
export function clearPetSignals(source: string) {
  const current = $petSignalsWritable.get()
  const next = current.filter(signal => signal.source !== source)

  if (next.length === current.length) {
    return
  }

  $petSignalsWritable.set(next)
  armPetSignalExpiry()
}

/**
 * Remove one `(source, id)` signal without affecting its siblings. Pass the
 * expected creation time when a delayed callback must not clear a replacement.
 */
export function clearPetSignal(source: string, id: string, expectedCreatedAt?: number) {
  const current = $petSignalsWritable.get()

  const next = current.filter(
    signal =>
      signal.source !== source ||
      signal.id !== id ||
      (expectedCreatedAt !== undefined && signal.createdAt !== expectedCreatedAt)
  )

  if (next.length === current.length) {
    return
  }

  $petSignalsWritable.set(next)
  armPetSignalExpiry()
}

function petSignalOutranks(candidate: PetSignal, selected: PetSignal): boolean {
  if (candidate.priority !== selected.priority) {
    return candidate.priority > selected.priority
  }

  if (candidate.createdAt !== selected.createdAt) {
    return candidate.createdAt > selected.createdAt
  }

  if (candidate.source !== selected.source) {
    return candidate.source < selected.source
  }

  return candidate.id < selected.id
}

/**
 * Select active state by priority, then recency, then lexical source/id for a
 * stable final tie-break.
 */
export function selectPetStateSignal(signals: readonly PetSignal[], now = Date.now()): PetSignal | undefined {
  let selected: PetSignal | undefined

  for (const signal of signals) {
    if (!isPetSignalValid(signal, now)) {
      continue
    }

    if (!selected || petSignalOutranks(signal, selected)) {
      selected = signal
    }
  }

  return selected
}

export function derivePetSignalState(signals: readonly PetSignal[], now = Date.now()): PetState {
  const signal = selectPetStateSignal(signals, now)

  return signal ? PET_SIGNAL_STATE[signal.state] : 'idle'
}
