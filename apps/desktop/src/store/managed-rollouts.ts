import { atom } from 'nanostores'

export interface ManagedRolloutSnapshot {
  revision: number
  rolloutId: string | null
  phase: string
  data: Record<string, unknown>
}

export interface ManagedRolloutsState {
  status: 'idle' | 'loading' | 'ready' | 'reconnecting' | 'unsupported' | 'error'
  revision: number | null
  snapshot: ManagedRolloutSnapshot | null
  error: string | null
}

export interface RolloutResponse {
  revision: number
  snapshot: ManagedRolloutSnapshot | null
}

export interface ManagedRolloutsBridge {
  capabilities: () => Promise<{ available: boolean; reason: string | null }>
  activeRevision: () => Promise<number | null>
  read: (sinceRevision: number | null) => Promise<RolloutResponse>
  command: (payload: Record<string, unknown>) => Promise<unknown>
}

const initial: ManagedRolloutsState = {
  status: 'idle',
  revision: null,
  snapshot: null,
  error: null
}

export const $managedRollouts = atom<ManagedRolloutsState>(initial)

let bridgeOverride: ManagedRolloutsBridge | null = null
let pollTimer: ReturnType<typeof setTimeout> | undefined
let pollGeneration = 0
let pollInFlight: Promise<void> | null = null
let commandInFlight: { key: string; promise: Promise<unknown> } | null = null
let pollingActive = false
let pollingInterval = 5_000
let retryDelay = 1_000
let capabilityState: 'unknown' | 'available' | 'unsupported' = 'unknown'

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.keys(value as Record<string, unknown>).sort().map(key => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function bridge(): ManagedRolloutsBridge | null {
  if (bridgeOverride) return bridgeOverride
  if (typeof window === 'undefined') return null
  return window.hermesDesktop?.connections?.managedRollouts ?? null
}

function setUnsupported(reason: string): void {
  const current = $managedRollouts.get()
  $managedRollouts.set({ ...current, status: 'unsupported', error: reason })
}

function schedulePoll(delayMs: number, generation = pollGeneration): void {
  if (!pollingActive) return
  if (pollTimer !== undefined) clearTimeout(pollTimer)
  pollTimer = setTimeout(() => {
    if (generation === pollGeneration && pollingActive) void pollManagedRollouts()
  }, Math.max(0, delayMs))
}

function validResponse(value: unknown): value is RolloutResponse {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Partial<RolloutResponse>
  const revision = candidate.revision
  if (typeof revision !== 'number' || !Number.isInteger(revision) || revision < 0) return false
  if (candidate.snapshot === null) return true
  const snapshot = candidate.snapshot
  return Boolean(
    snapshot &&
      typeof snapshot === 'object' &&
      Number.isInteger(snapshot.revision) &&
      snapshot.revision === candidate.revision &&
      (typeof snapshot.rolloutId === 'string' || snapshot.rolloutId === null) &&
      typeof snapshot.phase === 'string' &&
      snapshot.data !== null &&
      typeof snapshot.data === 'object' &&
      !Array.isArray(snapshot.data)
  )
}

async function capabilityAvailable(requestBridge: ManagedRolloutsBridge): Promise<boolean> {
  if (capabilityState === 'available') return true

  try {
    const result = await requestBridge.capabilities()
    if (!result || result.available !== true) {
      capabilityState = 'unsupported'
      setUnsupported(result?.reason ?? 'managed-rollouts-unavailable')
      return false
    }
    capabilityState = 'available'
    return true
  } catch (error: unknown) {
    capabilityState = 'unsupported'
    setUnsupported(error instanceof Error ? error.message : String(error))
    return false
  }
}

export async function pollManagedRollouts(): Promise<void> {
  if (pollInFlight) return pollInFlight
  const requestBridge = bridge()
  const generation = pollGeneration
  const run = (async () => {
    if (!requestBridge || !(await capabilityAvailable(requestBridge))) {
      if (pollingActive && generation === pollGeneration) schedulePoll(pollingInterval, generation)
      return
    }

    const previous = $managedRollouts.get()
    $managedRollouts.set({
      ...previous,
      status: previous.snapshot ? 'reconnecting' : 'loading',
      error: null
    })
    try {
      const activeRevision = await requestBridge.activeRevision()
      if (activeRevision !== null && (!Number.isSafeInteger(activeRevision) || activeRevision < 0)) {
        throw new Error('managed-rollouts-invalid-active-revision')
      }
      const current = $managedRollouts.get()
      if (activeRevision === current.revision && current.revision !== null) {
        $managedRollouts.set({ ...current, status: 'ready', error: null })
        retryDelay = 1_000
        schedulePoll(pollingInterval, generation)
        return
      }
      if (activeRevision === null && current.revision === null && current.snapshot === null) {
        $managedRollouts.set({ ...current, status: 'ready', error: null })
        retryDelay = 1_000
        schedulePoll(pollingInterval, generation)
        return
      }
      const response = await requestBridge.read(current.revision)
      if (generation !== pollGeneration) return
      if (!validResponse(response)) throw new Error('managed-rollouts-invalid-response')

      const currentAfterRead = $managedRollouts.get()
      if (currentAfterRead.revision !== null && response.revision < currentAfterRead.revision) {
        $managedRollouts.set({ ...currentAfterRead, status: 'ready', error: null })
        schedulePoll(pollingInterval, generation)
        return
      }
      if (response.revision === currentAfterRead.revision && response.snapshot === null) {
        // A terminal null is an acknowledgement for this revision, not proof
        // that the last settled snapshot never existed.
        $managedRollouts.set({ ...currentAfterRead, status: 'ready', error: null })
        retryDelay = 1_000
        schedulePoll(pollingInterval, generation)
        return
      }
      if (response.revision === currentAfterRead.revision && response.snapshot !== null && currentAfterRead.snapshot !== null) {
        $managedRollouts.set({ ...currentAfterRead, status: 'ready', error: null })
        retryDelay = 1_000
        schedulePoll(pollingInterval, generation)
        return
      }
      $managedRollouts.set({
        status: 'ready',
        revision: response.revision,
        snapshot: response.snapshot,
        error: null
      })
      retryDelay = 1_000
      schedulePoll(pollingInterval, generation)
    } catch (error: unknown) {
      if (generation !== pollGeneration) return
      const message = error instanceof Error ? error.message : String(error)
      $managedRollouts.set({ ...$managedRollouts.get(), status: 'error', error: message })
      schedulePoll(retryDelay, generation)
      retryDelay = Math.min(retryDelay * 2, 30_000)
    }
  })()
  pollInFlight = run.finally(() => {
    pollInFlight = null
  })
  return pollInFlight
}

export async function sendManagedRolloutCommand(command: Record<string, unknown>): Promise<unknown> {
  const requestBridge = bridge()
  if (!requestBridge?.command) throw new Error('managed-rollouts-command-unavailable')
  const requestId = typeof command.requestId === 'string' ? command.requestId : ''
  if (!requestId) throw new Error('managed-rollouts-request-id-required')
  const key = `${requestId}:${canonical(command)}`
  if (commandInFlight) {
    if (commandInFlight.key !== key) throw new Error('managed-rollouts-command-conflict')
    return commandInFlight.promise
  }
  const promise = requestBridge.command(command).finally(() => {
    if (commandInFlight?.key === key) commandInFlight = null
  })
  commandInFlight = { key, promise }
  return promise
}

export function startManagedRolloutPolling(intervalMs = 5_000): () => void {
  stopManagedRolloutPolling()
  pollingActive = true
  pollingInterval = Math.max(250, intervalMs)
  retryDelay = 1_000
  capabilityState = 'unknown'
  pollGeneration += 1
  void pollManagedRollouts()
  return stopManagedRolloutPolling
}

export function stopManagedRolloutPolling(): void {
  pollingActive = false
  pollGeneration += 1
  if (pollTimer !== undefined) clearTimeout(pollTimer)
  pollTimer = undefined
}

/** @internal */
export function _setManagedRolloutsBridgeForTests(next: ManagedRolloutsBridge | null): void {
  bridgeOverride = next
  capabilityState = 'unknown'
}

/** @internal */
export function _resetManagedRolloutsForTests(): void {
  stopManagedRolloutPolling()
  pollInFlight = null
  commandInFlight = null
  bridgeOverride = null
  capabilityState = 'unknown'
  $managedRollouts.set(initial)
}
