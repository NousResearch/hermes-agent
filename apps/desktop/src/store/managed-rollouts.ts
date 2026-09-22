import { atom } from 'nanostores'

export interface ManagedRolloutSnapshot {
  revision: number
  rolloutId: string | null
  phase: string
  data: Record<string, unknown>
}

export interface ManagedRolloutsState {
  status: 'idle' | 'loading' | 'ready' | 'reconnecting' | 'error'
  revision: number | null
  snapshot: ManagedRolloutSnapshot | null
  error: string | null
}

type RolloutResponse = { revision: number; snapshot: ManagedRolloutSnapshot | null }
export interface ManagedRolloutsBridge {
  read: (sinceRevision: number | null) => Promise<RolloutResponse>
  command: (payload: Record<string, unknown>) => Promise<unknown>
}

const initial: ManagedRolloutsState = { status: 'idle', revision: null, snapshot: null, error: null }
export const $managedRollouts = atom<ManagedRolloutsState>(initial)

let bridgeOverride: ManagedRolloutsBridge | null = null
let pollTimer: ReturnType<typeof setTimeout> | undefined
let pollGeneration = 0
let pollInFlight: Promise<void> | null = null
let commandInFlight: { key: string; promise: Promise<unknown> } | null = null

function bridge(): ManagedRolloutsBridge | null {
  const desktop = typeof window === 'undefined'
    ? undefined
    : (window as Window & { hermesDesktop?: { managedRollouts?: ManagedRolloutsBridge } }).hermesDesktop
  return bridgeOverride ?? desktop?.managedRollouts ?? null
}

function schedulePoll(delayMs: number): void {
  if (pollTimer !== undefined) clearTimeout(pollTimer)
  const generation = pollGeneration
  pollTimer = setTimeout(() => {
    if (generation === pollGeneration) void pollManagedRollouts()
  }, Math.max(0, delayMs))
}

export async function pollManagedRollouts(): Promise<void> {
  if (pollInFlight) return pollInFlight
  const requestBridge = bridge()
  if (!requestBridge) return
  const generation = pollGeneration
  const previous = $managedRollouts.get()
  $managedRollouts.set({ ...previous, status: previous.snapshot ? 'reconnecting' : 'loading', error: null })
  pollInFlight = requestBridge.read(previous.revision)
    .then(response => {
      if (generation !== pollGeneration) return
      const current = $managedRollouts.get()
      if (response.revision < (current.revision ?? -1)) return
      if (response.revision === current.revision && response.snapshot === null) {
        $managedRollouts.set({ ...$managedRollouts.get(), status: 'ready', error: null })
        return
      }
      $managedRollouts.set({
        status: 'ready',
        revision: response.revision,
        snapshot: response.snapshot,
        error: null
      })
    })
    .catch((error: unknown) => {
      if (generation !== pollGeneration) return
      $managedRollouts.set({ ...$managedRollouts.get(), status: 'error', error: error instanceof Error ? error.message : String(error) })
      schedulePoll(5_000)
    })
    .finally(() => {
      pollInFlight = null
    })
  return pollInFlight
}

export async function sendManagedRolloutCommand(command: Record<string, unknown>): Promise<unknown> {
  const requestBridge = bridge()
  if (!requestBridge) throw new Error('managed-rollouts-bridge-unavailable')
  const requestId = typeof command.requestId === 'string' ? command.requestId : ''
  if (!requestId) throw new Error('managed-rollouts-request-id-required')
  const key = `${requestId}:${JSON.stringify(command, Object.keys(command).sort())}`
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
  pollGeneration += 1
  const generation = pollGeneration
  void pollManagedRollouts()
  pollTimer = setTimeout(function tick() {
    if (generation !== pollGeneration) return
    void pollManagedRollouts()
    pollTimer = setTimeout(tick, Math.max(250, intervalMs))
  }, Math.max(250, intervalMs))
  return stopManagedRolloutPolling
}

export function stopManagedRolloutPolling(): void {
  pollGeneration += 1
  if (pollTimer !== undefined) clearTimeout(pollTimer)
  pollTimer = undefined
}

/** @internal */
export function _setManagedRolloutsBridgeForTests(next: ManagedRolloutsBridge | null): void {
  bridgeOverride = next
}

/** @internal */
export function _resetManagedRolloutsForTests(): void {
  stopManagedRolloutPolling()
  pollInFlight = null
  commandInFlight = null
  bridgeOverride = null
  $managedRollouts.set(initial)
}
