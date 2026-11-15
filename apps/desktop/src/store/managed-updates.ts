import { atom } from 'nanostores'

import type { DesktopManagedConnectionUpdateResult, DesktopManagedUpdateReceipt } from '@/global'

/** Renderer-side lifecycle of one managed SSH update (#93042 renderer unit,
 * slimmed to what #95942's engine actually exposes today: a single
 * `connections.updateManaged(id)` promise that resolves with a correlated
 * result + receipt. There is no streaming progress channel, so the in-flight
 * phase is one `updating` state and every finer-grained outcome (partial
 * restore, refusal, per-profile recovery) comes from the terminal result). */
export type ManagedUpdateStatus = 'failed' | 'idle' | 'partial' | 'refused' | 'updated' | 'updating'

export interface ManagedUpdateState {
  /** True when the refusal was the busy gate: another managed update already
   * owns this connection (the `managed-update-in-progress` envelope). */
  alreadyRunning: boolean
  connectionId: string
  finishedAt: number | null
  message: string | null
  receipt: DesktopManagedUpdateReceipt | null
  scopes: DesktopManagedConnectionUpdateResult['scopes']
  status: ManagedUpdateStatus
}

export const $managedUpdates = atom<Record<string, ManagedUpdateState>>({})

const inFlight = new Map<string, Promise<ManagedUpdateState>>()

// The main process signals the busy gate two ways: a `refused` result whose
// message says an update is already in progress, and (from sibling IPC paths
// it pauses) a thrown error carrying code `managed-update-in-progress`.
const IN_PROGRESS_PATTERN = /managed[\s-]update[\s\w]*\bin[\s-]progress|managed-update-in-progress|already in progress/i

function managedUpdater(): ((id: string) => Promise<DesktopManagedConnectionUpdateResult>) | null {
  if (typeof window === 'undefined') {
    return null
  }

  return window.hermesDesktop?.connections?.updateManaged ?? null
}

/** Whether this Electron main exposes the transactional SSH update bridge.
 * Older mains without it hide the section entirely (fail closed, no fake
 * button that could mutate a live serve process). */
export function managedUpdatesSupported(): boolean {
  return Boolean(managedUpdater())
}

export function isManagedUpdateBusyMessage(message: string | null | undefined): boolean {
  return Boolean(message && IN_PROGRESS_PATTERN.test(message))
}

function publish(state: ManagedUpdateState): void {
  $managedUpdates.set({ ...$managedUpdates.get(), [state.connectionId]: state })
}

function blankState(connectionId: string): ManagedUpdateState {
  return {
    alreadyRunning: false,
    connectionId,
    finishedAt: null,
    message: null,
    receipt: null,
    scopes: [],
    status: 'idle'
  }
}

function terminalState(connectionId: string, result: DesktopManagedConnectionUpdateResult): ManagedUpdateState {
  const message = result.message || result.error || result.receipt?.stopReason || null
  const alreadyRunning = result.outcome === 'refused' && isManagedUpdateBusyMessage(message)

  const status: ManagedUpdateStatus = result.ok
    ? 'updated'
    : result.outcome === 'refused'
      ? 'refused'
      : result.updateOk && !result.restoreOk
        ? 'partial'
        : 'failed'

  return {
    alreadyRunning,
    connectionId,
    finishedAt: Date.now(),
    message,
    receipt: result.receipt ?? null,
    scopes: result.scopes ?? [],
    status
  }
}

/** Run one managed update. A repeat click while it is in flight joins the
 * existing promise — the main process gate would refuse a second start, and
 * the UI must not turn that self-collision into a scary failure row. */
export function runManagedUpdate(connectionId: string): Promise<ManagedUpdateState> {
  const existing = inFlight.get(connectionId)

  if (existing) {
    return existing
  }

  const updateManaged = managedUpdater()

  if (!updateManaged) {
    const state: ManagedUpdateState = {
      ...blankState(connectionId),
      finishedAt: Date.now(),
      status: 'failed'
    }

    publish(state)

    return Promise.resolve(state)
  }

  publish({ ...blankState(connectionId), status: 'updating' })

  const run = (async (): Promise<ManagedUpdateState> => {
    try {
      const result = await updateManaged(connectionId)
      const state = terminalState(connectionId, result)
      publish(state)

      return state
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)

      const alreadyRunning =
        (error as { code?: unknown } | null)?.code === 'managed-update-in-progress' ||
        isManagedUpdateBusyMessage(message)

      const state: ManagedUpdateState = {
        ...blankState(connectionId),
        alreadyRunning,
        finishedAt: Date.now(),
        message,
        status: alreadyRunning ? 'refused' : 'failed'
      }

      publish(state)

      return state
    } finally {
      inFlight.delete(connectionId)
    }
  })()

  inFlight.set(connectionId, run)

  return run
}

/** @internal */
export function _resetManagedUpdatesForTests(): void {
  inFlight.clear()
  $managedUpdates.set({})
}
