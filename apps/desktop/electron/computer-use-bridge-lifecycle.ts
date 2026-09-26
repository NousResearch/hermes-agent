interface BridgeSocketState {
  closedByDesktop?: boolean
  ws?: { close: () => void }
}

type BridgeTimer = ReturnType<typeof setTimeout>

/**
 * Mutable state for Desktop's profile-scoped reverse bridge connections.
 *
 * Keeping the cancellation generation beside the socket/timer/promise maps is
 * important: deleting a pool backend must also invalidate a connector that is
 * still awaiting the local sidecar or an OAuth ticket, otherwise that stale
 * async continuation can recreate the WebSocket after the profile was stopped.
 */
class ScopedComputerUseBridgeLifecycle<State extends BridgeSocketState = BridgeSocketState> {
  readonly connections = new Map<string, State>()
  readonly connectionPromises = new Map<string, Promise<State | null>>()
  readonly reconnectTimers = new Map<string, BridgeTimer>()

  private readonly generations = new Map<string, number>()
  private readonly owners = new Map<string, Set<string>>()
  private readonly clearTimer: (timer: BridgeTimer) => void

  constructor(clearTimer: (timer: BridgeTimer) => void = clearTimeout) {
    this.clearTimer = clearTimer
  }

  generation(key: string): number {
    return this.generations.get(key) || 0
  }

  isCurrent(key: string, generation: number): boolean {
    return this.generation(key) === generation
  }

  acquire(key: string, owner: string): void {
    if (!key || !owner) {
      return
    }

    let owners = this.owners.get(key)

    if (!owners) {
      owners = new Set()
      this.owners.set(key, owners)
    }

    owners.add(owner)
  }

  hasOwners(key: string): boolean {
    return Boolean(this.owners.get(key)?.size)
  }

  hasScopedActivity(): boolean {
    return Boolean(
      this.owners.size || this.connections.size || this.connectionPromises.size || this.reconnectTimers.size
    )
  }

  release(key: string, owner: string): boolean {
    const owners = this.owners.get(key)

    if (!owners?.delete(owner)) {
      return false
    }

    if (owners.size > 0) {
      return false
    }

    this.owners.delete(key)
    this.cancel(key)

    return true
  }

  closeSockets(key: string | null = null, allowReconnect = false): void {
    const keys = key ? [key] : [...this.connections.keys()]

    for (const scopedKey of keys) {
      const state = this.connections.get(scopedKey)
      this.connections.delete(scopedKey)

      if (!state?.ws) {
        continue
      }

      state.closedByDesktop = !allowReconnect

      try {
        state.ws.close()
      } catch {
        // The peer may already be gone.
      }
    }
  }

  cancel(key: string): void {
    this.generations.set(key, this.generation(key) + 1)

    const timer = this.reconnectTimers.get(key)

    if (timer !== undefined) {
      this.clearTimer(timer)
      this.reconnectTimers.delete(key)
    }

    this.connectionPromises.delete(key)
    this.closeSockets(key)
  }

  cancelAll(): void {
    const keys = new Set([
      ...this.owners.keys(),
      ...this.connections.keys(),
      ...this.connectionPromises.keys(),
      ...this.reconnectTimers.keys()
    ])

    for (const key of keys) {
      this.owners.delete(key)
      this.cancel(key)
    }
  }
}

interface BridgeReconnectGuard<State extends BridgeSocketState = BridgeSocketState> {
  lifecycle: ScopedComputerUseBridgeLifecycle<State>
  remoteKey: string
  stopping: boolean
  enabled: boolean
  capturedGlobalGeneration: number
  currentGlobalGeneration: number
  capturedScopedGeneration: number
  scheduleReconnect: () => void
}

interface BridgeOwnerRelease<State extends BridgeSocketState = BridgeSocketState> {
  lifecycle: ScopedComputerUseBridgeLifecycle<State>
  remoteKey: string
  owner: string
  stopSidecar: () => void
}

/** Release one scope and stop the shared sidecar only after the last scope is gone. */
function releaseBridgeOwnerAndStopSidecarIfIdle<State extends BridgeSocketState>({
  lifecycle,
  remoteKey,
  owner,
  stopSidecar
}: BridgeOwnerRelease<State>): boolean {
  const releasedFinalOwner = lifecycle.release(remoteKey, owner)

  if (!releasedFinalOwner || lifecycle.hasScopedActivity()) {
    return false
  }

  stopSidecar()

  return true
}

/** Schedule recovery only while the connection attempt still owns its scope. */
function scheduleBridgeReconnectIfCurrent<State extends BridgeSocketState>({
  lifecycle,
  remoteKey,
  stopping,
  enabled,
  capturedGlobalGeneration,
  currentGlobalGeneration,
  capturedScopedGeneration,
  scheduleReconnect
}: BridgeReconnectGuard<State>): boolean {
  if (
    stopping ||
    !enabled ||
    capturedGlobalGeneration !== currentGlobalGeneration ||
    !lifecycle.isCurrent(remoteKey, capturedScopedGeneration) ||
    !lifecycle.hasOwners(remoteKey)
  ) {
    return false
  }

  scheduleReconnect()

  return true
}

export { releaseBridgeOwnerAndStopSidecarIfIdle, scheduleBridgeReconnectIfCurrent, ScopedComputerUseBridgeLifecycle }
export type { BridgeOwnerRelease, BridgeReconnectGuard, BridgeSocketState, BridgeTimer }
